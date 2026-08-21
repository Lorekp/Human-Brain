"""Loopback-only file transport between WSL evaluation and a Windows model server.

The SWE runner lives in WSL while the CUDA llama.cpp server runs on Windows.  A
shared-directory queue keeps both HTTP listeners on their respective loopback
interfaces, avoiding an inbound Windows firewall exception.
"""

from __future__ import annotations

import argparse
import base64
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time
from typing import Any
import urllib.error
import urllib.request
import uuid


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)


def _forward_headers(headers: Any) -> dict[str, str]:
    return {
        str(name): str(value)
        for name, value in headers.items()
        if str(name).lower() not in HOP_BY_HOP_HEADERS | {"host", "content-length"}
    }


def _claim_request(request_path: Path) -> Path | None:
    request_id = request_path.name.removesuffix(".request.json")
    claimed_path = request_path.with_name(f"{request_id}.working.json")
    try:
        os.replace(request_path, claimed_path)
    except FileNotFoundError:
        return None
    return claimed_path


def process_one_request(queue: Path, upstream: str) -> bool:
    """Forward one queued request, returning whether work was found."""
    request_paths = sorted(queue.glob("*.request.json"))
    if not request_paths:
        return False
    claimed_path = _claim_request(request_paths[0])
    if claimed_path is None:
        return True

    request_id = claimed_path.name.removesuffix(".working.json")
    response_path = queue / f"{request_id}.response.json"
    try:
        queued = json.loads(claimed_path.read_text(encoding="utf-8"))
        body = base64.b64decode(queued["body_b64"])
        method = str(queued["method"]).upper()
        request = urllib.request.Request(
            url=f"{upstream.rstrip('/')}{queued['path']}",
            data=body if body or method not in {"GET", "HEAD"} else None,
            headers=_forward_headers(queued.get("headers", {})),
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=3600) as response:
                status = response.status
                headers = _forward_headers(response.headers)
                response_body = response.read()
        except urllib.error.HTTPError as error:
            status = error.code
            headers = _forward_headers(error.headers)
            response_body = error.read()
        payload = {
            "status": status,
            "headers": headers,
            "body_b64": base64.b64encode(response_body).decode("ascii"),
        }
    except Exception as error:
        body = json.dumps(
            {"error": f"{type(error).__name__}: {error}"},
            separators=(",", ":"),
        ).encode("utf-8")
        payload = {
            "status": HTTPStatus.BAD_GATEWAY,
            "headers": {"Content-Type": "application/json"},
            "body_b64": base64.b64encode(body).decode("ascii"),
        }
    _atomic_write_json(response_path, payload)
    claimed_path.unlink(missing_ok=True)
    return True


def serve_worker(
    queue: Path,
    upstream: str,
    *,
    poll_seconds: float = 0.02,
    stop_event: threading.Event | None = None,
) -> None:
    """Service queued HTTP requests until interrupted or ``stop_event`` is set."""
    queue.mkdir(parents=True, exist_ok=True)
    while stop_event is None or not stop_event.is_set():
        if not process_one_request(queue, upstream):
            time.sleep(poll_seconds)


def make_proxy_handler(queue: Path, timeout_seconds: float) -> type[BaseHTTPRequestHandler]:
    class FileProxyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _proxy(self) -> None:
            request_id = uuid.uuid4().hex
            request_path = queue / f"{request_id}.request.json"
            response_path = queue / f"{request_id}.response.json"
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length) if content_length else b""
            _atomic_write_json(
                request_path,
                {
                    "method": self.command,
                    "path": self.path,
                    "headers": _forward_headers(self.headers),
                    "body_b64": base64.b64encode(body).decode("ascii"),
                },
            )

            deadline = time.monotonic() + timeout_seconds
            while not response_path.exists():
                if time.monotonic() >= deadline:
                    request_path.unlink(missing_ok=True)
                    self.send_error(HTTPStatus.GATEWAY_TIMEOUT, "model bridge timed out")
                    return
                time.sleep(0.02)

            response = json.loads(response_path.read_text(encoding="utf-8"))
            response_path.unlink(missing_ok=True)
            response_body = base64.b64decode(response["body_b64"])
            self.send_response(int(response["status"]))
            for name, value in response.get("headers", {}).items():
                if name.lower() not in HOP_BY_HOP_HEADERS | {"content-length"}:
                    self.send_header(name, value)
            self.send_header("Content-Length", str(len(response_body)))
            self.send_header("Connection", "close")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(response_body)

        do_DELETE = _proxy
        do_GET = _proxy
        do_HEAD = _proxy
        do_OPTIONS = _proxy
        do_PATCH = _proxy
        do_POST = _proxy
        do_PUT = _proxy

        def log_message(self, format: str, *args: Any) -> None:
            return

    return FileProxyHandler


def build_proxy_server(
    queue: Path,
    host: str,
    port: int,
    timeout_seconds: float,
) -> ThreadingHTTPServer:
    queue.mkdir(parents=True, exist_ok=True)
    return ThreadingHTTPServer(
        (host, port),
        make_proxy_handler(queue, timeout_seconds),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--queue", type=Path, required=True)
    worker_parser.add_argument("--upstream", default="http://127.0.0.1:8080")
    worker_parser.add_argument("--poll-seconds", type=float, default=0.02)

    proxy_parser = subparsers.add_parser("proxy")
    proxy_parser.add_argument("--queue", type=Path, required=True)
    proxy_parser.add_argument("--host", default="127.0.0.1")
    proxy_parser.add_argument("--port", type=int, default=8081)
    proxy_parser.add_argument("--timeout-seconds", type=float, default=3600.0)

    args = parser.parse_args()
    if args.mode == "worker":
        print(
            json.dumps(
                {"mode": "worker", "queue": str(args.queue), "upstream": args.upstream}
            ),
            flush=True,
        )
        serve_worker(args.queue, args.upstream, poll_seconds=args.poll_seconds)
        return

    server = build_proxy_server(args.queue, args.host, args.port, args.timeout_seconds)
    print(
        json.dumps(
            {
                "mode": "proxy",
                "queue": str(args.queue),
                "url": f"http://{args.host}:{args.port}",
            }
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
