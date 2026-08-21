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
CONTRACT = ROOT / "tools" / "hermes" / "WORKER_CONTRACT.md"
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")


def _run(
    *args: str,
    env: dict[str, str] | None = None,
    launcher_repository: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    assert POWERSHELL is not None, "PowerShell is required to test the PowerShell launcher"
    repository_index = args.index("-RepositoryPath")
    repository = Path(args[repository_index + 1])
    launcher_root = launcher_repository or repository
    launcher = launcher_root / "tools" / "hermes" / "run_fleet.ps1"
    return subprocess.run(
        [POWERSHELL, "-NoLogo", "-NoProfile", "-File", str(launcher), *args],
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


def _clean_repository(tmp_path: Path, name: str = "repo") -> tuple[Path, str]:
    repo = tmp_path / name
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Hermes Fleet Test")
    _git(repo, "config", "user.email", "hermes-fleet@example.invalid")
    (repo / "README.md").write_text("test repository\n", encoding="utf-8")
    hermes_tools = repo / "tools" / "hermes"
    hermes_tools.mkdir(parents=True)
    shutil.copy2(LAUNCHER, hermes_tools / LAUNCHER.name)
    shutil.copy2(CONTRACT, hermes_tools / CONTRACT.name)
    _git(repo, "add", ".")
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
    return _write_tasks(
        tmp_path,
        [
            _task_record(repository_sha, **overrides),
            _task_record(
                repository_sha,
                id="verify-fleet-contract",
                title="Verify the public worker contract",
                objective="Independently verify the bounded public-source acceptance evidence.",
            ),
        ],
    )


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
        "if ($args -contains 'config') { $key = Value-After 'get'; if ($key -eq 'model') { if ($env:HERMES_FAKE_CONFIG_DRIFT -eq 'model') { Write-Output '{\"default\":\"wrong-model\",\"provider\":\"nous\"}' } else { Write-Output '{\"default\":\"stealth/ox-alpha\",\"provider\":\"nous\"}' }; return }; if ($key -eq 'delegation') { $isolation = $true; if ($env:HERMES_FAKE_CONFIG_DRIFT -eq 'isolation') { $isolation = $false }; [ordered]@{ model='stealth/ox-alpha'; provider='nous'; max_concurrent_children=128; max_spawn_depth=2; subagent_auto_approve=$false; worktree_isolation=$isolation } | ConvertTo-Json -Compress; return }; if ($key -eq 'tool_output') { $lineLength = 2000; if ($env:HERMES_FAKE_CONFIG_DRIFT -eq 'line-length') { $lineLength = 500 }; [ordered]@{ max_bytes=50000; max_lines=2000; max_line_length=$lineLength } | ConvertTo-Json -Compress; return }; if ($key -eq 'file_read_max_chars') { Write-Output '100000'; return }; throw 'unexpected config key' }\n"
        "if ($args -contains 'create') { $createIndex = [Array]::IndexOf($args, 'create'); $card = [ordered]@{ id='t_mock-card'; title=$args[$createIndex + 1]; body=(Value-After '--body'); assignee=(Value-After '--assignee'); status='blocked'; workspace_kind='worktree'; workspace_path=$env:HERMES_FAKE_REPOSITORY; branch_name=(Value-After '--branch'); project_id='p_board-inherited'; created_by='human-brain-fleet'; max_retries=1; model_override=(Value-After '--model'); provider_override=(Value-After '--provider') }; $cardJson = $card | ConvertTo-Json -Compress; if (-not [string]::IsNullOrWhiteSpace($env:HERMES_FAKE_CREATED)) { [System.IO.File]::WriteAllText($env:HERMES_FAKE_CREATED, $cardJson) }; Write-Output $cardJson; return }\n"
        "$attachIndex = [Array]::IndexOf($args, 'attach')\n"
        "if ($attachIndex -ge 0) { [System.IO.File]::Copy($args[$attachIndex + 2], $env:HERMES_FAKE_PROMPT_CAPTURE, $true); Write-Output 'Attached prompt'; return }\n"
        "if ($args -contains 'attachments') { if (-not (Test-Path -LiteralPath $env:HERMES_FAKE_PROMPT_CAPTURE)) { Write-Output '[]'; return }; $item = [ordered]@{ id=1; filename=$env:HERMES_FAKE_ATTACHMENT_NAME; content_type='text/markdown'; size=(Get-Item -LiteralPath $env:HERMES_FAKE_PROMPT_CAPTURE).Length; uploaded_by='human-brain-fleet'; stored_path=$env:HERMES_FAKE_PROMPT_CAPTURE; created_at=1 }; ConvertTo-Json -InputObject @($item) -Compress; return }\n"
        "if ($args -contains 'promote') { [System.IO.File]::WriteAllText($env:HERMES_FAKE_PROMOTED, 'yes'); Write-Output '{\"task_id\":\"t_mock-card\",\"promoted\":true,\"dry_run\":false,\"forced\":false,\"reason\":\"verified\",\"error\":null}'; return }\n"
        "if ($args -contains 'list') { if (($args -contains '--archived') -and $env:HERMES_FAKE_OTHER_READY -eq '1') { Write-Output '[{\"id\":\"t_other\",\"title\":\"Other\",\"status\":\"ready\"}]' } elseif (($args -contains '--archived') -and -not [string]::IsNullOrWhiteSpace($env:HERMES_FAKE_CREATED) -and (Test-Path -LiteralPath $env:HERMES_FAKE_CREATED)) { $created = Get-Content -LiteralPath $env:HERMES_FAKE_CREATED -Raw | ConvertFrom-Json; if ($env:HERMES_FAKE_DUPLICATE_AFTER_CREATE -eq '1') { $duplicate = [ordered]@{ id='t_duplicate'; title=$created.title; body=$created.body; created_by=$created.created_by; status='blocked' }; ConvertTo-Json -InputObject @($created, $duplicate) -Compress } else { ConvertTo-Json -InputObject @($created) -Compress } } elseif (($args -contains '--status') -and (Test-Path -LiteralPath $env:HERMES_FAKE_PROMOTED)) { Write-Output '[{\"id\":\"t_mock-card\"}]' } elseif (($args -contains '--status') -and $env:HERMES_FAKE_OTHER_READY -eq '1') { Write-Output '[{\"id\":\"t_other\"}]' } else { Write-Output '[]' }; return }\n"
        "if ($args -contains 'dispatch') { Write-Output '{\"spawned\":[{\"task_id\":\"t_mock-card\",\"assignee\":\"humanbrainox\",\"workspace\":\"mock\"}]}'; return }\n"
        "throw 'unexpected fake Hermes invocation'\n",
        encoding="utf-8",
    )
    return fake, capture, prompt_capture


def _assert_installed_file_redactor_is_identity(prompt_path: Path) -> None:
    hermes = shutil.which("hermes")
    if hermes is None:
        return
    hermes_executable = Path(hermes).resolve()
    hermes_root = hermes_executable.parents[2]
    hermes_python = hermes_executable.parent / "python.exe"
    check = subprocess.run(
        [
            str(hermes_python),
            "-c",
            (
                "from pathlib import Path; "
                "from agent.redact import redact_sensitive_text; "
                "p=Path(__import__('sys').argv[1]); "
                "s=p.read_text(encoding='utf-8'); "
                "raise SystemExit(0 if redact_sensitive_text(s, file_read=True)==s else 1)"
            ),
            str(prompt_path),
        ],
        cwd=hermes_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert check.returncode == 0, check.stdout + check.stderr


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


def test_single_task_manifest_fails_before_hermes(tmp_path: Path) -> None:
    repo, base_sha = _clean_repository(tmp_path)
    manifest = _write_tasks(tmp_path, [_task_record(base_sha)])
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
    assert "at least two tasks" in (result.stdout + result.stderr)
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
        ({"allowed_paths": ["logs/run_evaluation"]}, "allowed_paths"),
        ({"allowed_paths": ["logs/run_evaluation/results.json"]}, "allowed_paths"),
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


def test_oversized_aggregate_prompt_fails_before_git_or_hermes(tmp_path: Path) -> None:
    repo, base_sha = _clean_repository(tmp_path)
    tasks = [
        _task_record(
            base_sha,
            id=f"large-task-{index}",
            objective="Review the public cell implementation. " + ("x" * 15_900),
        )
        for index in range(1, 129)
    ]
    manifest = _write_tasks(tmp_path, tasks)
    fake_hermes, capture, prompt_capture = _write_fake_hermes(tmp_path)
    refs_before = _git(repo, "for-each-ref", "--format=%(refname) %(objectname)")
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
    assert "UTF-8 context envelope" in (result.stdout + result.stderr)
    assert not capture.exists()
    assert _git(repo, "for-each-ref", "--format=%(refname) %(objectname)") == refs_before
    assert _git(repo, "status", "--porcelain", "--untracked-files=all") == ""


def test_large_delegate_context_is_line_safe_and_reconstructable(tmp_path: Path) -> None:
    repo, base_sha = _clean_repository(tmp_path)
    large_task = _task_record(
        base_sha,
        acceptance=["reproducible public evidence " + ("x" * 1_850)] * 55,
    )
    companion_task = _task_record(base_sha, id="verify-fleet-contract")
    manifest = _write_tasks(tmp_path, [large_task, companion_task])
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    fake_hermes, capture, prompt_capture = _write_fake_hermes(tmp_path)
    env = os.environ | {
        "HERMES_FAKE_CAPTURE": str(capture),
        "HERMES_FAKE_PROMPT_CAPTURE": str(prompt_capture),
        "HERMES_FAKE_REPOSITORY": str(repo),
        "HERMES_FAKE_ATTACHMENT_NAME": f"human-brain-ox-fleet-prompt-{manifest_hash[:12]}.md",
        "HERMES_FAKE_PROMOTED": str(tmp_path / "promoted.txt"),
        "HERMES_FAKE_CREATED": str(tmp_path / "created.json"),
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
    attached_prompt = prompt_capture.read_text(encoding="utf-8")
    assert len(attached_prompt) > 100_000
    assert max(map(len, attached_prompt.splitlines())) <= 1_800
    records_json = attached_prompt.split(
        "--- BEGIN LINE-SAFE DELEGATE RECORDS ---", 1
    )[1].split("--- END LINE-SAFE DELEGATE RECORDS ---", 1)[0]
    records = json.loads(records_json)
    assert list(records[0]) == ["goal_json", "context_json_chunks", "role"]
    assert json.loads(records[0]["goal_json"]).startswith("Complete Human Brain")
    reconstructed_context = json.loads("".join(records[0]["context_json_chunks"]))
    assert len(reconstructed_context) > 100_000
    child_task_json = reconstructed_context.split(
        "--- BEGIN EXACT IMMUTABLE TASK JSON ---", 1
    )[1].split("--- END EXACT IMMUTABLE TASK JSON ---", 1)[0]
    assert json.loads(child_task_json) == large_task


def test_128_minimal_tasks_fit_the_prompt_envelope(tmp_path: Path) -> None:
    repo, base_sha = _clean_repository(tmp_path)
    tasks = [
        _task_record(base_sha, id=f"fleet-task-{index}")
        for index in range(1, 129)
    ]
    manifest = _write_tasks(tmp_path, tasks)
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    fake_hermes, capture, prompt_capture = _write_fake_hermes(tmp_path)
    env = os.environ | {
        "HERMES_FAKE_CAPTURE": str(capture),
        "HERMES_FAKE_PROMPT_CAPTURE": str(prompt_capture),
        "HERMES_FAKE_REPOSITORY": str(repo),
        "HERMES_FAKE_ATTACHMENT_NAME": f"human-brain-ox-fleet-prompt-{manifest_hash[:12]}.md",
        "HERMES_FAKE_PROMOTED": str(tmp_path / "promoted.txt"),
        "HERMES_FAKE_CREATED": str(tmp_path / "created.json"),
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
    attached_prompt = prompt_capture.read_text(encoding="utf-8")
    assert len(attached_prompt.encode("utf-8")) <= 2 * 1024 * 1024
    records_json = attached_prompt.split(
        "--- BEGIN LINE-SAFE DELEGATE RECORDS ---", 1
    )[1].split("--- END LINE-SAFE DELEGATE RECORDS ---", 1)[0]
    records = json.loads(records_json)
    assert len(records) == 128
    assert all(
        json.loads(record["goal_json"])
        and json.loads("".join(record["context_json_chunks"]))
        for record in records
    )
    _assert_installed_file_redactor_is_identity(prompt_capture)


@pytest.mark.parametrize(
    "objective",
    [
        "Inspect ../Fly Brain and compare its implementation.",
        "Read .runtime/private-eval.json before proposing the change.",
        "Inspect logs/run_evaluation before proposing the change.",
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


def test_repository_must_share_launcher_git_common_directory(tmp_path: Path) -> None:
    trusted_repo, _ = _clean_repository(tmp_path, "trusted-repo")
    other_repo, other_sha = _clean_repository(tmp_path, "other-repo")
    manifest = _write_manifest(tmp_path, other_sha)
    fake_hermes, capture, prompt_capture = _write_fake_hermes(tmp_path)
    env = os.environ | {
        "HERMES_FAKE_CAPTURE": str(capture),
        "HERMES_FAKE_PROMPT_CAPTURE": str(prompt_capture),
    }

    result = _run(
        "-ManifestPath",
        str(manifest),
        "-RepositoryPath",
        str(other_repo),
        "-HermesExecutable",
        str(fake_hermes),
        "-Execute",
        "-AuthorizeSourceUpload",
        env=env,
        launcher_repository=trusted_repo,
    )

    assert result.returncode != 0
    assert "same Git repository" in (result.stdout + result.stderr)
    assert not capture.exists()


@pytest.mark.parametrize(
    ("drift", "message"),
    [("isolation", "profile configuration"), ("line-length", "read configuration")],
)
def test_profile_config_drift_fails_before_git_or_board_mutation(
    tmp_path: Path, drift: str, message: str
) -> None:
    repo, base_sha = _clean_repository(tmp_path)
    manifest = _write_manifest(tmp_path, base_sha)
    fake_hermes, capture, prompt_capture = _write_fake_hermes(tmp_path)
    refs_before = _git(repo, "for-each-ref", "--format=%(refname) %(objectname)")
    env = os.environ | {
        "HERMES_FAKE_CAPTURE": str(capture),
        "HERMES_FAKE_PROMPT_CAPTURE": str(prompt_capture),
        "HERMES_FAKE_CONFIG_DRIFT": drift,
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
    assert _git(repo, "for-each-ref", "--format=%(refname) %(objectname)") == refs_before
    assert _git(repo, "status", "--porcelain", "--untracked-files=all") == ""


def test_board_wide_launcher_lock_blocks_without_hermes_or_git_mutation(
    tmp_path: Path,
) -> None:
    repo, base_sha = _clean_repository(tmp_path)
    manifest = _write_manifest(tmp_path, base_sha)
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    fake_hermes, capture, prompt_capture = _write_fake_hermes(tmp_path)
    refs_before = _git(repo, "for-each-ref", "--format=%(refname) %(objectname)")
    env = os.environ | {
        "HERMES_FAKE_CAPTURE": str(capture),
        "HERMES_FAKE_PROMPT_CAPTURE": str(prompt_capture),
        "HERMES_FAKE_REPOSITORY": str(repo),
        "HERMES_FAKE_ATTACHMENT_NAME": f"human-brain-ox-fleet-prompt-{manifest_hash[:12]}.md",
        "HERMES_FAKE_PROMOTED": str(tmp_path / "promoted.txt"),
        "HERMES_FAKE_CREATED": str(tmp_path / "created.json"),
    }
    mutex_name = r"Local\HumanBrainOxFleet_humanbrainox_human_brain_ox"
    holder_script = (
        f"$mutex = [System.Threading.Mutex]::new($false, '{mutex_name}'); "
        "if (-not $mutex.WaitOne(0)) { exit 2 }; "
        "[Console]::Out.WriteLine('READY'); [Console]::Out.Flush(); "
        "[Console]::In.ReadLine() | Out-Null; "
        "$mutex.ReleaseMutex(); $mutex.Dispose()"
    )
    holder = subprocess.Popen(
        [POWERSHELL, "-NoLogo", "-NoProfile", "-Command", holder_script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline().strip() == "READY"
    try:
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
    finally:
        holder.communicate("\n", timeout=10)

    assert result.returncode != 0
    assert "another fleet launcher owns" in (result.stdout + result.stderr)
    assert not capture.exists()
    assert _git(repo, "for-each-ref", "--format=%(refname) %(objectname)") == refs_before


def test_duplicate_post_create_card_fails_before_attachment_or_promotion(
    tmp_path: Path,
) -> None:
    repo, base_sha = _clean_repository(tmp_path)
    manifest = _write_manifest(tmp_path, base_sha)
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    fake_hermes, capture, prompt_capture = _write_fake_hermes(tmp_path)
    promoted = tmp_path / "promoted.txt"
    env = os.environ | {
        "HERMES_FAKE_CAPTURE": str(capture),
        "HERMES_FAKE_PROMPT_CAPTURE": str(prompt_capture),
        "HERMES_FAKE_REPOSITORY": str(repo),
        "HERMES_FAKE_ATTACHMENT_NAME": f"human-brain-ox-fleet-prompt-{manifest_hash[:12]}.md",
        "HERMES_FAKE_PROMOTED": str(promoted),
        "HERMES_FAKE_CREATED": str(tmp_path / "created.json"),
        "HERMES_FAKE_DUPLICATE_AFTER_CREATE": "1",
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
    assert "exactly one deterministic fleet card" in (result.stdout + result.stderr)
    assert not prompt_capture.exists()
    assert not promoted.exists()


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
        ["--profile", "humanbrainox", "config", "get", "model", "--json"],
        ["--profile", "humanbrainox", "config", "get", "delegation", "--json"],
        ["--profile", "humanbrainox", "config", "get", "tool_output", "--json"],
        [
            "--profile",
            "humanbrainox",
            "config",
            "get",
            "file_read_max_chars",
            "--json",
        ],
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
    token_shaped_fixture = "sk-abcdefghijklmno"
    repo, base_sha = _clean_repository(tmp_path, token_shaped_fixture)
    injection_marker = tmp_path / "shell-interpolation-must-not-run.txt"
    objective = (
        'Audit this literal input; Set-Content -LiteralPath "'
        + str(injection_marker)
        + '" -Value pwned; $(Write-Output injected); preserve <multi word>, '
        + "{multi word}, Unicode 🧠 café, and fake fixture "
        + token_shaped_fixture
    )
    task = _task_record(base_sha, objective=objective)
    companion_task = _task_record(
        base_sha,
        id=token_shaped_fixture,
        title="Verify the public worker contract",
        objective="Independently verify the bounded public-source acceptance evidence.",
    )
    manifest = _write_tasks(tmp_path, [task, companion_task])
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
        "HERMES_FAKE_CREATED": str(tmp_path / "created.json"),
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
        "config",
        "get",
        "model",
        "--json",
    ]
    assert calls[2] == [
        "--profile",
        "humanbrainox",
        "config",
        "get",
        "delegation",
        "--json",
    ]
    assert calls[3] == [
        "--profile",
        "humanbrainox",
        "config",
        "get",
        "tool_output",
        "--json",
    ]
    assert calls[4] == [
        "--profile",
        "humanbrainox",
        "config",
        "get",
        "file_read_max_chars",
        "--json",
    ]
    assert calls[5] == [
        "--profile",
        "humanbrainox",
        "kanban",
        "--board",
        "human-brain-ox",
        "list",
        "--archived",
        "--json",
    ]
    create = calls[6]
    body_index = create.index("--body") + 1
    body = create[body_index]
    assert body == (
        f"Read and verify attached {attachment_name} for manifest {manifest_hash}; "
        "stop if it is missing or mismatched. Completion requires exactly one "
        "joined delegate_task batch for all 2 tasks, capture all 2 results, verify "
        "the evidence, commit exactly one hash-named report under "
        "tools/hermes/reports/ and no other parent change, then kanban_complete "
        "with the report path and commit SHA."
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
    assert calls[7] == [
        "--profile",
        "humanbrainox",
        "kanban",
        "--board",
        "human-brain-ox",
        "list",
        "--archived",
        "--json",
    ]
    assert calls[8] == [
        "--profile",
        "humanbrainox",
        "kanban",
        "--board",
        "human-brain-ox",
        "attachments",
        "t_mock-card",
        "--json",
    ]
    attach = calls[9]
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
    assert calls[10] == [
        "--profile",
        "humanbrainox",
        "kanban",
        "--board",
        "human-brain-ox",
        "attachments",
        "t_mock-card",
        "--json",
    ]
    assert calls[11] == [
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
    assert calls[12] == [
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
    assert calls[13] == [
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
    assert calls[14] == [
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
    assert "Set-Content -LiteralPath" not in attached_prompt
    assert injection_marker.name not in attached_prompt
    assert "$(Write-Output injected)" not in attached_prompt
    assert token_shaped_fixture not in attached_prompt
    assert "delegate_task" in attached_prompt
    assert "background=true" in attached_prompt
    assert max(map(len, attached_prompt.splitlines())) <= 1_800
    records_json = attached_prompt.split(
        "--- BEGIN LINE-SAFE DELEGATE RECORDS ---", 1
    )[1].split("--- END LINE-SAFE DELEGATE RECORDS ---", 1)[0]
    records = json.loads(records_json)
    assert all(
        list(record) == ["goal_json", "context_json_chunks", "role"]
        for record in records
    )
    delegates = [
        {
            "goal": json.loads(record["goal_json"]),
            "context": json.loads("".join(record["context_json_chunks"])),
            "role": record["role"],
        }
        for record in records
    ]
    assert len(delegates) == 2
    for delegate, expected_task in zip(delegates, [task, companion_task], strict=True):
        assert list(delegate) == ["goal", "context", "role"]
        assert delegate["role"] == "leaf"
        assert delegate["goal"] == (
            f"Complete Human Brain fleet task {expected_task['id']} "
            "under its immutable contract."
        )
        assert expected_task["objective"] not in delegate["goal"]
        child_task_json = delegate["context"].split(
            "--- BEGIN EXACT IMMUTABLE TASK JSON ---", 1
        )[1].split("--- END EXACT IMMUTABLE TASK JSON ---", 1)[0]
        assert json.loads(child_task_json) == expected_task
        assert CONTRACT.read_text(encoding="utf-8") in delegate["context"]
        assert "[WORKTREE ISOLATION]" in delegate["context"]
    assert "🧠" in delegates[0]["context"]
    assert "<multi word>" not in delegates[0]["goal"]
    assert "{multi word}" not in delegates[0]["goal"]
    child_context = delegates[0]["context"]
    assert manifest_hash in child_context
    assert base_sha in child_context
    assert "humanbrainox" in child_context
    assert "nous:stealth/ox-alpha" in child_context
    assert f"worktree:{repo}" in child_context
    assert f"hermes/ox-fleet-{manifest_hash[:12]}" in child_context
    assert not injection_marker.exists()
    assert _git(repo, "rev-parse", f"refs/heads/hermes/ox-fleet-{manifest_hash[:12]}") == base_sha
    assert manifest_hash in result.stdout
    assert base_sha in result.stdout
    assert "Hermes Agent v0.20.5 (test)" in result.stdout
    assert "humanbrainox" in result.stdout
    assert "nous:stealth/ox-alpha" in result.stdout
    assert "t_mock-card" in result.stdout
    _assert_installed_file_redactor_is_identity(prompt_capture)


def test_manifest_identity_is_derived_from_one_immutable_byte_snapshot() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert "$manifestBytes = [System.IO.File]::ReadAllBytes" in launcher
    assert "Read-TaskManifest -Content $manifestText" in launcher
    assert "Get-Sha256Hex -Bytes $manifestBytes" in launcher
    assert "[System.IO.File]::ReadAllLines" not in launcher
