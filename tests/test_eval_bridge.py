from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import urllib.request

from humanbrain.eval_bridge import build_proxy_server, serve_worker


class EchoHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"]))
        response = json.dumps(
            {"path": self.path, "payload": json.loads(body)}
        ).encode("utf-8")
        self.send_response(201)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: object) -> None:
        return


def _start_server(server: ThreadingHTTPServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def test_file_bridge_forwards_get_and_json_post(tmp_path: Path) -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), EchoHandler)
    upstream_thread = _start_server(upstream)

    stop_worker = threading.Event()
    worker_thread = threading.Thread(
        target=serve_worker,
        args=(tmp_path, f"http://127.0.0.1:{upstream.server_port}"),
        kwargs={"poll_seconds": 0.001, "stop_event": stop_worker},
        daemon=True,
    )
    worker_thread.start()
    proxy = build_proxy_server(tmp_path, "127.0.0.1", 0, 5.0)
    proxy_thread = _start_server(proxy)

    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{proxy.server_port}/health", timeout=5
        ) as response:
            assert response.status == 200
            assert json.loads(response.read()) == {"status": "ok"}

        request = urllib.request.Request(
            f"http://127.0.0.1:{proxy.server_port}/v1/chat/completions",
            data=b'{"model":"gemma"}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 201
            assert json.loads(response.read()) == {
                "path": "/v1/chat/completions",
                "payload": {"model": "gemma"},
            }
    finally:
        proxy.shutdown()
        upstream.shutdown()
        stop_worker.set()
        proxy_thread.join(timeout=2)
        upstream_thread.join(timeout=2)
        worker_thread.join(timeout=2)

    assert list(tmp_path.iterdir()) == []
