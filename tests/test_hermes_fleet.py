from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools" / "hermes" / "run_fleet.ps1"
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    assert POWERSHELL is not None, "PowerShell is required to test the PowerShell launcher"
    return subprocess.run(
        [POWERSHELL, "-NoLogo", "-NoProfile", "-File", str(LAUNCHER), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def _clean_repository(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Hermes Fleet Test")
    _git(repo, "config", "user.email", "hermes-fleet@example.invalid")
    (repo / "README.md").write_text("test repository\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "test baseline")
    return repo, _git(repo, "rev-parse", "HEAD")


def _task_record(repository_sha: str, **overrides: object) -> dict[str, object]:
    task: dict[str, object] = {
        "id": "audit-cell-dynamics",
        "title": "Audit the public cell dynamics",
        "objective": "Inspect public source and report one bounded improvement.",
        "base_sha": repository_sha,
        "allowed_paths": ["src/humanbrain/recurrent.py"],
        "acceptance": ["Return evidence and do not modify files outside allowed_paths."],
        "resource_class": "cpu",
        "public_source_only": True,
        "no_hidden_eval": True,
    }
    task.update(overrides)
    return task


def _write_tasks(tmp_path: Path, tasks: list[dict[str, object]]) -> Path:
    manifest = tmp_path / "tasks.jsonl"
    manifest.write_text(
        "".join(json.dumps(task) + "\n" for task in tasks), encoding="utf-8"
    )
    return manifest


def _write_manifest(tmp_path: Path, repository_sha: str, **overrides: object) -> Path:
    return _write_tasks(tmp_path, [_task_record(repository_sha, **overrides)])


def _write_fake_hermes(tmp_path: Path) -> tuple[Path, Path, Path]:
    capture = tmp_path / "hermes-calls.jsonl"
    prompt_capture = tmp_path / "attached-prompt.md"
    fake = tmp_path / "fake-hermes.ps1"
    fake.write_text(
        "[System.IO.File]::AppendAllText($env:HERMES_FAKE_CAPTURE, "
        "(($args | ConvertTo-Json -Compress) + [Environment]::NewLine))\n"
        "if ($args -contains '--version') { Write-Output 'Hermes Agent v0.20.5 (test)'; return }\n"
        "$scriptArgs = @($args)\n"
        "function Value-After([string]$flag) { $i = [Array]::IndexOf($scriptArgs, $flag); if ($i -lt 0) { return $null }; return $scriptArgs[$i + 1] }\n"
        "if ($args -contains 'create') { $createIndex = [Array]::IndexOf($args, 'create'); [ordered]@{ id='t_mock-card'; title=$args[$createIndex + 1]; body=(Value-After '--body'); assignee=(Value-After '--assignee'); status='blocked'; workspace_kind='worktree'; workspace_path=$env:HERMES_FAKE_REPOSITORY; branch_name=(Value-After '--branch'); project_id=$null; max_retries=1; model_override=(Value-After '--model'); provider_override=(Value-After '--provider') } | ConvertTo-Json -Compress; return }\n"
        "$attachIndex = [Array]::IndexOf($args, 'attach')\n"
        "if ($attachIndex -ge 0) { [System.IO.File]::Copy($args[$attachIndex + 2], $env:HERMES_FAKE_PROMPT_CAPTURE, $true); Write-Output 'Attached prompt'; return }\n"
        "if ($args -contains 'attachments') { if (-not (Test-Path -LiteralPath $env:HERMES_FAKE_PROMPT_CAPTURE)) { Write-Output '[]'; return }; $item = [ordered]@{ id=1; filename=$env:HERMES_FAKE_ATTACHMENT_NAME; content_type='text/markdown'; size=(Get-Item -LiteralPath $env:HERMES_FAKE_PROMPT_CAPTURE).Length; uploaded_by='human-brain-fleet'; stored_path=$env:HERMES_FAKE_PROMPT_CAPTURE; created_at=1 }; ConvertTo-Json -InputObject @($item) -Compress; return }\n"
        "if ($args -contains 'promote') { [System.IO.File]::WriteAllText($env:HERMES_FAKE_PROMOTED, 'yes'); Write-Output '{\"task_id\":\"t_mock-card\",\"promoted\":true,\"dry_run\":false,\"forced\":false,\"reason\":\"verified\",\"error\":null}'; return }\n"
        "if ($args -contains 'list') { if (($args -contains '--archived') -and $env:HERMES_FAKE_OTHER_READY -eq '1') { Write-Output '[{\"id\":\"t_other\",\"title\":\"Other\",\"status\":\"ready\"}]' } elseif (($args -contains '--status') -and (Test-Path -LiteralPath $env:HERMES_FAKE_PROMOTED)) { Write-Output '[{\"id\":\"t_mock-card\"}]' } elseif (($args -contains '--status') -and $env:HERMES_FAKE_OTHER_READY -eq '1') { Write-Output '[{\"id\":\"t_other\"}]' } else { Write-Output '[]' }; return }\n"
        "if ($args -contains 'dispatch') { Write-Output '{\"spawned\":[{\"task_id\":\"t_mock-card\",\"assignee\":\"humanbrainox\",\"workspace\":\"mock\"}]}'; return }\n"
        "throw 'unexpected fake Hermes invocation'\n",
        encoding="utf-8",
    )
    return fake, capture, prompt_capture


@pytest.mark.parametrize("authorization", [(), ("-Execute",), ("-AuthorizeSourceUpload",)])
def test_any_missing_authorization_is_plan_only_and_never_invokes_hermes(
    tmp_path: Path, authorization: tuple[str, ...]
) -> None:
    repo, base_sha = _clean_repository(tmp_path)
    manifest = _write_manifest(tmp_path, base_sha)
    fake_hermes, capture, prompt_capture = _write_fake_hermes(tmp_path)
    env = os.environ | {
        "HERMES_FAKE_CAPTURE": str(capture),
        "HERMES_FAKE_PROMPT_CAPTURE": str(prompt_capture),
    }

    result = _run(
        "-ManifestPath",
        str(manifest),
        "-RepositoryPath",
        str(repo),
        "-HermesExecutable",
        str(fake_hermes),
        *authorization,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "PLAN ONLY" in result.stdout
    assert not capture.exists()


def test_manifest_without_hidden_eval_guard_fails_before_hermes(tmp_path: Path) -> None:
    repo, base_sha = _clean_repository(tmp_path)
    manifest = _write_manifest(tmp_path, base_sha, no_hidden_eval=False)
    fake_hermes, capture, prompt_capture = _write_fake_hermes(tmp_path)
    env = os.environ | {
        "HERMES_FAKE_CAPTURE": str(capture),
        "HERMES_FAKE_PROMPT_CAPTURE": str(prompt_capture),
    }

    result = _run(
        "-ManifestPath",
        str(manifest),
        "-RepositoryPath",
        str(repo),
        "-HermesExecutable",
        str(fake_hermes),
        "-Execute",
        "-AuthorizeSourceUpload",
        env=env,
    )

    assert result.returncode != 0
    assert "no_hidden_eval" in (result.stdout + result.stderr)
    assert not capture.exists()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"id": "../../unsafe"}, "id"),
        ({"title": ""}, "title"),
        ({"acceptance": []}, "acceptance"),
        ({"public_source_only": False}, "public_source_only"),
        ({"resource_class": "tpu"}, "resource_class"),
        ({"allowed_paths": ["../Fly Brain/private.txt"]}, "allowed_paths"),
        ({"base_sha": "0" * 40}, "base_sha"),
    ],
)
def test_invalid_task_record_fails_before_hermes(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    repo, base_sha = _clean_repository(tmp_path)
    manifest = _write_manifest(tmp_path, base_sha, **overrides)
    fake_hermes, capture, prompt_capture = _write_fake_hermes(tmp_path)
    env = os.environ | {
        "HERMES_FAKE_CAPTURE": str(capture),
        "HERMES_FAKE_PROMPT_CAPTURE": str(prompt_capture),
    }

    result = _run(
        "-ManifestPath",
        str(manifest),
        "-RepositoryPath",
        str(repo),
        "-HermesExecutable",
        str(fake_hermes),
        "-Execute",
        "-AuthorizeSourceUpload",
        env=env,
    )

    assert result.returncode != 0
    assert message in (result.stdout + result.stderr)
    assert not capture.exists()


@pytest.mark.parametrize(
    ("case", "message"),
    [("duplicate", "duplicate"), ("two-gpu", "GPU"), ("too-many", "128")],
)
def test_manifest_wide_limits_fail_before_hermes(
    tmp_path: Path, case: str, message: str
) -> None:
    repo, base_sha = _clean_repository(tmp_path)
    if case == "duplicate":
        tasks = [_task_record(base_sha), _task_record(base_sha)]
    elif case == "two-gpu":
        tasks = [
            _task_record(base_sha, id="gpu-one", resource_class="gpu"),
            _task_record(base_sha, id="gpu-two", resource_class="gpu"),
        ]
    else:
        tasks = [
            _task_record(base_sha, id=f"task-{index}") for index in range(1, 130)
        ]
    manifest = _write_tasks(tmp_path, tasks)
    fake_hermes, capture, prompt_capture = _write_fake_hermes(tmp_path)
    env = os.environ | {
        "HERMES_FAKE_CAPTURE": str(capture),
        "HERMES_FAKE_PROMPT_CAPTURE": str(prompt_capture),
    }

    result = _run(
        "-ManifestPath",
        str(manifest),
        "-RepositoryPath",
        str(repo),
        "-HermesExecutable",
        str(fake_hermes),
        "-Execute",
        "-AuthorizeSourceUpload",
        env=env,
    )

    assert result.returncode != 0
    assert message in (result.stdout + result.stderr)
    assert not capture.exists()


@pytest.mark.parametrize(
    "objective",
    [
        "Inspect ../Fly Brain and compare its implementation.",
        "Read .runtime/private-eval.json before proposing the change.",
    ],
)
def test_forbidden_project_or_private_runtime_reference_fails_closed(
    tmp_path: Path, objective: str
) -> None:
    repo, base_sha = _clean_repository(tmp_path)
    manifest = _write_manifest(tmp_path, base_sha, objective=objective)
    fake_hermes, capture, prompt_capture = _write_fake_hermes(tmp_path)
    env = os.environ | {
        "HERMES_FAKE_CAPTURE": str(capture),
        "HERMES_FAKE_PROMPT_CAPTURE": str(prompt_capture),
    }

    result = _run(
        "-ManifestPath",
        str(manifest),
        "-RepositoryPath",
        str(repo),
        "-HermesExecutable",
        str(fake_hermes),
        "-Execute",
        "-AuthorizeSourceUpload",
        env=env,
    )

    assert result.returncode != 0
    assert "forbidden" in (result.stdout + result.stderr)
    assert not capture.exists()


def test_dirty_repository_fails_before_hermes(tmp_path: Path) -> None:
    repo, base_sha = _clean_repository(tmp_path)
    manifest = _write_manifest(tmp_path, base_sha)
    (repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    fake_hermes, capture, prompt_capture = _write_fake_hermes(tmp_path)
    env = os.environ | {
        "HERMES_FAKE_CAPTURE": str(capture),
        "HERMES_FAKE_PROMPT_CAPTURE": str(prompt_capture),
    }

    result = _run(
        "-ManifestPath",
        str(manifest),
        "-RepositoryPath",
        str(repo),
        "-HermesExecutable",
        str(fake_hermes),
        "-Execute",
        "-AuthorizeSourceUpload",
        env=env,
    )

    assert result.returncode != 0
    assert "clean" in (result.stdout + result.stderr)
    assert not capture.exists()


def test_existing_ready_card_blocks_board_wide_dispatch(tmp_path: Path) -> None:
    repo, base_sha = _clean_repository(tmp_path)
    manifest = _write_manifest(tmp_path, base_sha)
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    fake_hermes, capture, prompt_capture = _write_fake_hermes(tmp_path)
    env = os.environ | {
        "HERMES_FAKE_CAPTURE": str(capture),
        "HERMES_FAKE_PROMPT_CAPTURE": str(prompt_capture),
        "HERMES_FAKE_REPOSITORY": str(repo),
        "HERMES_FAKE_ATTACHMENT_NAME": f"human-brain-ox-fleet-prompt-{manifest_hash[:12]}.md",
        "HERMES_FAKE_PROMOTED": str(tmp_path / "promoted.txt"),
        "HERMES_FAKE_OTHER_READY": "1",
    }

    result = _run(
        "-ManifestPath",
        str(manifest),
        "-RepositoryPath",
        str(repo),
        "-HermesExecutable",
        str(fake_hermes),
        "-Execute",
        "-AuthorizeSourceUpload",
        env=env,
    )

    assert result.returncode != 0
    assert "ready card" in (result.stdout + result.stderr)
    calls = [json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines()]
    assert calls == [
        ["--profile", "humanbrainox", "--version"],
        [
            "--profile",
            "humanbrainox",
            "kanban",
            "--board",
            "human-brain-ox",
            "list",
            "--archived",
            "--json",
        ],
    ]


def test_authorized_execute_uses_exact_safe_hermes_argv(tmp_path: Path) -> None:
    repo, base_sha = _clean_repository(tmp_path)
    injection_marker = tmp_path / "shell-interpolation-must-not-run.txt"
    objective = (
        'Audit this literal input; Set-Content -LiteralPath "'
        + str(injection_marker)
        + '" -Value pwned; $(Write-Output injected)'
    )
    manifest = _write_manifest(tmp_path, base_sha, objective=objective)
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    fake_hermes, capture, prompt_capture = _write_fake_hermes(tmp_path)
    promoted_marker = tmp_path / "promoted.txt"
    attachment_name = f"human-brain-ox-fleet-prompt-{manifest_hash[:12]}.md"
    env = os.environ | {
        "HERMES_FAKE_CAPTURE": str(capture),
        "HERMES_FAKE_PROMPT_CAPTURE": str(prompt_capture),
        "HERMES_FAKE_REPOSITORY": str(repo),
        "HERMES_FAKE_ATTACHMENT_NAME": attachment_name,
        "HERMES_FAKE_PROMOTED": str(promoted_marker),
    }

    result = _run(
        "-ManifestPath",
        str(manifest),
        "-RepositoryPath",
        str(repo),
        "-HermesExecutable",
        str(fake_hermes),
        "-Execute",
        "-AuthorizeSourceUpload",
        env=env,
    )

    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines()]
    assert calls[0] == ["--profile", "humanbrainox", "--version"]
    assert calls[1] == [
        "--profile",
        "humanbrainox",
        "kanban",
        "--board",
        "human-brain-ox",
        "list",
        "--archived",
        "--json",
    ]
    create = calls[2]
    body_index = create.index("--body") + 1
    body = create[body_index]
    assert body == (
        f"Read the attached {attachment_name} before acting. "
        f"Its declared manifest SHA-256 is {manifest_hash}. "
        "Stop if the attachment is missing or disagrees with this card."
    )
    assert create == [
        "--profile",
        "humanbrainox",
        "kanban",
        "--board",
        "human-brain-ox",
        "create",
        f"Human Brain Ox fleet {manifest_hash[:12]}",
        "--body",
        body,
        "--assignee",
        "humanbrainox",
        "--workspace",
        f"worktree:{repo}",
        "--branch",
        f"hermes/ox-fleet-{manifest_hash[:12]}",
        "--idempotency-key",
        f"human-brain-ox-fleet-{manifest_hash}",
        "--max-runtime",
        "2h",
        "--max-retries",
        "1",
        "--model",
        "stealth/ox-alpha",
        "--provider",
        "nous",
        "--goal",
        "--goal-max-turns",
        "20",
        "--created-by",
        "human-brain-fleet",
        "--initial-status",
        "blocked",
        "--json",
    ]
    assert calls[3] == [
        "--profile",
        "humanbrainox",
        "kanban",
        "--board",
        "human-brain-ox",
        "attachments",
        "t_mock-card",
        "--json",
    ]
    attach = calls[4]
    prompt_path = Path(attach[7])
    normalized_attach = attach.copy()
    normalized_attach[7] = "<PROMPT_FILE>"
    assert normalized_attach == [
        "--profile",
        "humanbrainox",
        "kanban",
        "--board",
        "human-brain-ox",
        "attach",
        "t_mock-card",
        "<PROMPT_FILE>",
        "--name",
        attachment_name,
        "--content-type",
        "text/markdown",
        "--author",
        "human-brain-fleet",
    ]
    assert not prompt_path.exists()
    assert calls[5] == [
        "--profile",
        "humanbrainox",
        "kanban",
        "--board",
        "human-brain-ox",
        "attachments",
        "t_mock-card",
        "--json",
    ]
    assert calls[6] == [
        "--profile",
        "humanbrainox",
        "kanban",
        "--board",
        "human-brain-ox",
        "list",
        "--status",
        "ready",
        "--json",
    ]
    assert calls[7] == [
        "--profile",
        "humanbrainox",
        "kanban",
        "--board",
        "human-brain-ox",
        "promote",
        "t_mock-card",
        f"verified fleet prompt {manifest_hash}",
        "--json",
    ]
    assert calls[8] == [
        "--profile",
        "humanbrainox",
        "kanban",
        "--board",
        "human-brain-ox",
        "list",
        "--status",
        "ready",
        "--json",
    ]
    assert calls[9] == [
        "--profile",
        "humanbrainox",
        "kanban",
        "--board",
        "human-brain-ox",
        "dispatch",
        "--max",
        "1",
        "--json",
    ]
    attached_prompt = prompt_capture.read_text(encoding="utf-8")
    assert "Set-Content -LiteralPath" in attached_prompt
    assert injection_marker.name in attached_prompt
    assert "$(Write-Output injected)" in attached_prompt
    assert "delegate_task" in attached_prompt
    assert "background=true" in attached_prompt
    assert not injection_marker.exists()
    assert _git(repo, "rev-parse", f"refs/heads/hermes/ox-fleet-{manifest_hash[:12]}") == base_sha
    assert manifest_hash in result.stdout
    assert base_sha in result.stdout
    assert "Hermes Agent v0.20.5 (test)" in result.stdout
    assert "humanbrainox" in result.stdout
    assert "nous:stealth/ox-alpha" in result.stdout
    assert "t_mock-card" in result.stdout
