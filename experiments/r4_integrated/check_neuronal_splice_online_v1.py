"""Independent saved-evidence checks for the selected neuronal splice protocol.

This evaluator may read the parent-only roster. It never belongs in the worker
package and never imports or executes a neuronal model or action executor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import time
from pathlib import Path

from experiments.r4_integrated.neuronal_splice_receipts_v1 import (
    HEAD_LIMITS, LINE_LIMIT, TREE_LIMIT, close, exact_keys, hash_value, hex_bytes,
    logsumexp, number, proposal_digest, require, softmax, tree_bytes,
    validate_head, validate_prediction, validate_proposal, validate_reader,
)

DRAFT = True
ROOT = Path(__file__).resolve().parents[2]
ARMS = ("learned", "action_frozen", "slow_frozen", "memory_disabled")
MODEL_BYTES = 634432
TAPE_BYTES = 656904


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
                      allow_nan=False).encode("ascii")


def documents_adjacency(documents):
    successors = []
    for encoded in documents:
        raw = hex_bytes(encoded)
        start = len(successors)
        successors.extend(range(start + 1, start + len(raw) + 1))
        successors.append(start + len(raw))
    require(1 <= len(successors) <= 128, "document package exceeds tape geometry")
    return successors


def ordered_files(files, *, maximum=128):
    require(type(files) is list and 1 <= len(files) <= 4, "invalid workspace file count")
    result = []
    for item in files:
        exact_keys(item, ("path", "content_hex"))
        path = item["path"]
        require(type(path) is str and 0 < len(path.encode("utf-8")) <= 31,
                "invalid workspace path bound")
        raw = hex_bytes(item["content_hex"], maximum)
        result.append((path, raw))
    require(len({path.casefold() for path, _ in result}) == len(result), "duplicate file path")
    return sorted(result, key=lambda item: item[0].encode("utf-8"))


def workspace_digest(files):
    ordered = ordered_files(files)
    digest = hashlib.sha256(b"humanbrain.raw-splice-workspace.v1\0")
    digest.update(len(ordered).to_bytes(4, "big"))
    for path, raw in ordered:
        for value in (path.encode("utf-8"), raw):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    return digest.hexdigest()


def boundaries(files):
    return [(path, offset) for path, raw in ordered_files(files) for offset in range(len(raw) + 1)]


def apply_saved(files, proposal):
    """Independent literal byte splice; no teacher or semantic target is accepted."""
    validate_proposal(proposal)
    before = ordered_files(files)
    rejection = None
    if proposal["workspace_sha256"] != workspace_digest(files):
        rejection = "stale_workspace"
    elif proposal["operation"] == "SPLICE":
        found = [(path, raw) for path, raw in before if path == proposal["path"]]
        if proposal["stop_reason"] != "eos":
            rejection = "missing_eos"
        elif not found or not 0 <= proposal["start"] <= proposal["end"] <= len(found[0][1]):
            rejection = "invalid_span"
        else:
            start, end = proposal["start"], proposal["end"]
            replacement = hex_bytes(proposal["replacement_hex"], 16)
            before = [(path, raw[:start] + replacement + raw[end:] if path == proposal["path"] else raw)
                      for path, raw in before]
    return [{"path": path, "content_hex": raw.hex()} for path, raw in before], rejection


def task_grade(task, proposal, after_files, rejection, execution):
    expected_after, expected_rejection = apply_saved(task["files"], proposal)
    require(ordered_files(after_files, maximum=144) == ordered_files(expected_after, maximum=144),
            "saved after-files differ from actual selected action")
    require((rejection is None) == (expected_rejection is None), "rejection status mismatch")
    if rejection is not None:
        require(type(rejection) is str and 0 < len(rejection.encode("utf-8")) <= 1024,
                "invalid rejection diagnostic")
    stdout = hex_bytes(execution["stdout_hex"], 16384)
    stderr = hex_bytes(execution["stderr_hex"], 16384)
    require(type(execution["returncode"]) is int or execution["returncode"] is None,
            "invalid process outcome")
    for name, raw in (("stdout", stdout), ("stderr", stderr)):
        require(execution[name + "_sha256"] == hashlib.sha256(raw).hexdigest(), "output digest mismatch")
    require(type(execution["materialized_matches_action"]) is bool, "invalid materialization flag")
    success = (rejection is None and execution["returncode"] == 0 and not stderr
               and execution["materialized_matches_action"] and not execution["decode_errors"]
               and stdout == hex_bytes(task["expected_stdout_hex"])
               and ordered_files(after_files, maximum=144)
               == ordered_files(task["expected_after_files"], maximum=144)
               and (task["required_operation"] != "STOP" or proposal["operation"] == "STOP"))
    canonical_correct = (proposal["operation"] == "STOP" if task["required_operation"] == "STOP"
                         else proposal == task["teacher"])
    return {"task_success": success, "canonical_tuple_correct": canonical_correct}


def reader_work(query, slots, decisions, *, teaching, arm):
    require(arm in ARMS and 1 <= slots <= 128 and 1 <= decisions <= 17, "invalid reader work geometry")
    return dict(
        query_raw_bytes=len(query), query_eos_symbols=1, query_lag_shift_steps=len(query) + 1,
        query_population_normalizations=1, address_feature_slots=slots,
        address_feature_products=slots * 32768, address_feature_channel_reductions=slots * 31744,
        address_score_synapse_evaluations=slots * 1024, address_score_reductions=slots * 1023,
        address_score_evaluations=slots, address_normalizations=1,
        codec_slot_evaluations=slots, codec_output_potentials=slots * 257, codec_normalizations=slots,
        codec_response_gathers=slots * (decisions - 1),
        codec_response_values_gathered=slots * (decisions - 1) * 1028,
        codec_response_bytes_gathered=slots * (decisions - 1) * 7196,
        mixture_output_reductions=257 * decisions, prefix_belief_updates=slots * decisions,
        branch_cursor_initializations=slots, branch_cursor_advances=slots * (decisions - 1),
        teacher_offers=decisions if teaching else 0, codec_update_group_offers=int(teaching),
        address_update_group_offers=int(teaching),
        enabled_codec_update_groups=int(teaching and arm != "slow_frozen"),
        enabled_address_update_groups=int(teaching and arm != "slow_frozen"),
        address_local_synapse_contributions=slots * 1024 if teaching else 0,
        address_local_slot_reductions=(slots - 1) * 1024 if teaching else 0,
    )


def action_work(request, files, source, *, teaching, arm, stop_teacher=False):
    """Shape arithmetic independent of the neuronal feature implementation."""
    ordered = ordered_files(files)
    lengths = dict((path, len(raw)) for path, raw in ordered)
    f, b = len(ordered), sum(length + 1 for length in lengths.values())
    chosen = lengths[source["path"]]
    ends = chosen - source["start"] + 1
    span, payload = source["end"] - source["start"], len(hex_bytes(source["replacement_hex"], 16))
    raw = len(request) + sum(len(path.encode("utf-8")) for path in lengths) + sum(lengths.values())
    raw += chosen + span + payload
    eos = 3 + 2 * f + int(source["stop_reason"] == "eos")
    context_count = 1
    if not stop_teacher:
        raw += sum(lengths.values()) + chosen
        eos += f + 1
        context_count += b + ends
    work = dict(raw_bytes_encoded=raw, eos_codes_encoded=eos, code_rows_loaded=raw + eos,
                boundary_contexts=context_count, context_code_gathers=32 * context_count,
                custody_tensor_bytes_hashed=(MODEL_BYTES + TAPE_BYTES) * (1 if teaching else 2))
    shapes = {"operation": (1, 4244, 4241 * 32)}
    if not stop_teacher:
        shapes = {"start": (b, 6177, (f * 5152 + b * 1024) * 32),
                  "end": (ends, 2049, ends * 2048 * 32), **shapes}
    for head, (rows, width, products) in shapes.items():
        work[head + "_feature_products"] = products
        work[head + "_feature_channel_reductions"] = products // 32 * 31
        work[head + "_feature_normalizations"] = rows
        work[head + "_feature_components"] = rows * width
        outputs = 2 if head == "operation" else rows
        work[head + "_score_products"] = outputs * width
        work[head + "_score_reductions"] = outputs * (width - 1)
        work[head + "_score_evaluations"] = outputs
        work[head + "_softmaxes"] = 1
        if teaching:
            work[head + "_local_products"] = outputs * width
            work[head + "_local_reductions"] = 0 if head == "operation" else (rows - 1) * width
            work[head + "_teacher_offers"] = 1
            work[head + "_log_softmaxes"] = 1
            work[head + "_enabled_update_groups"] = int(arm in ("learned", "memory_disabled"))
    work["operation_bias_additions"] = 2
    return work


def check_feedback(feedback, prediction, teacher, files, successor, request, arm):
    exact_keys(feedback, ("attempt_id", "action_updates", "action_losses", "teacher_choices", "reader", "reader_updates", "work"))
    require(feedback["attempt_id"] == prediction["attempt_id"], "feedback attempt mismatch")
    validate_proposal(teacher)
    stop = teacher["operation"] == "STOP"
    heads = {"operation"} if stop else set(HEAD_LIMITS)
    for name in ("action_updates", "action_losses", "teacher_choices"):
        exact_keys(feedback[name], heads)
    boundary_list = boundaries(files)
    targets = {"operation": int(not stop)}
    if not stop:
        targets["start"] = boundary_list.index((teacher["path"], teacher["start"]))
        targets["end"] = teacher["end"] - teacher["start"]
    source = prediction["proposal"] if stop else teacher
    counts = {"start": len(boundary_list), "end": sum(path == source["path"] and offset >= source["start"]
                                                       for path, offset in boundary_list), "operation": 2}
    for head in heads:
        decision = feedback["teacher_choices"][head]
        validate_head(decision, head, count=counts[head])
        logits = decision["logits"]
        close(feedback["action_losses"][head], logsumexp(logits) - logits[targets[head]])
        require(type(feedback["action_updates"][head]) is bool, "invalid changed flag")
        if arm in ("action_frozen", "slow_frozen"):
            require(not feedback["action_updates"][head], "frozen action group changed")
    exact_keys(feedback["work"], ("action", "reader"))
    require(feedback["work"]["action"] == action_work(request, files, source, teaching=True,
                                                      arm=arm, stop_teacher=stop), "action teacher work mismatch")
    if stop:
        require(feedback["reader"] is None and feedback["reader_updates"] is None
                and feedback["work"]["reader"] == {}, "STOP offered reader teaching")
        return 0, 1
    reader = feedback["reader"]
    exact_keys(feedback["reader_updates"], ("codec", "address"))
    require(all(type(value) is bool for value in feedback["reader_updates"].values()), "invalid reader changed flags")
    require(arm != "slow_frozen" or not any(feedback["reader_updates"].values()), "frozen reader changed")
    require(reader["kind"] == "teacher", "missing reader teacher")
    result = validate_reader(reader, successor, targets=hex_bytes(teacher["replacement_hex"], 16))
    expected = reader_work(request, len(successor), result["decisions"], teaching=True, arm=arm)
    require(reader["work"] == feedback["work"]["reader"] == expected, "reader teacher work mismatch")
    return result["decisions"], 3


class Stream:
    """Keeps scalar/hash state only; the bounded pair function owns current rows."""

    def __init__(self, path, metadata):
        exact_keys(metadata, ("path", "rows", "bytes", "sha256", "max_line_bytes", "max_row_tree_bytes"))
        require(Path(path).name == metadata["path"], "journal path mismatch")
        hash_value(metadata["sha256"])
        self.handle = Path(path).open("rb")
        self.metadata, self.hasher = metadata, hashlib.sha256()
        self.rows = self.bytes = self.max_line = self.max_tree = 0

    def next(self):
        line = self.handle.readline(LINE_LIMIT + 1)
        require(line and len(line) <= LINE_LIMIT and line.endswith(b"\n") and line.isascii(), "invalid bounded line")
        row = json.loads(line)
        require(type(row) is dict, "journal row must be an object")
        size = tree_bytes(row)
        require(size <= TREE_LIMIT, "parsed receipt exceeds tree limit")
        self.hasher.update(line)
        self.rows += 1
        self.bytes += len(line)
        self.max_line, self.max_tree = max(self.max_line, len(line)), max(self.max_tree, size)
        return row, hashlib.sha256(line).hexdigest()

    def finish(self):
        require(self.handle.read(1) == b"", "journal contains trailing records")
        self.handle.close()
        for key, value in (("rows", self.rows), ("bytes", self.bytes), ("max_line_bytes", self.max_line),
                           ("sha256", self.hasher.hexdigest())):
            require(self.metadata[key] == value, "journal completion mismatch")
        require(self.metadata["max_row_tree_bytes"] <= TREE_LIMIT, "writer tree exceeded bound")


def check_monitor(outer, *, pid, command=None):
    require(outer["status"] == "passed" and outer["exit_code"] == 0, "monitor did not complete")
    require(outer["seconds_limit"] == 900 and outer["tree_peak_sum_mib_limit"] == 2048,
            "monitor ceilings changed")
    require(0 < number(outer["elapsed_seconds"]) <= 900
            and 0 < number(outer["observed_tree_peak_sum_mib"]) <= 2048, "observed budget exceeded")
    require(outer["deadline_unix"] - outer["started_unix"] == 900, "deadline mismatch")
    require(command is None or outer["command"] == command, "monitor command mismatch")
    for actual in {pid, outer["child_pid"]}:
        identities = [key for key in outer["observed_process_peaks_mib"] if key.split("/")[0] == str(actual)]
        require(len(identities) == 1, "actual process absent or creation identity ambiguous")
        created = (int(identities[0].split("/")[1]) - 621355968000000000) / 10_000_000
        require(outer["started_unix"] <= created <= outer["started_unix"] + outer["elapsed_seconds"],
                "process creation outside selected launch")


def expected_runtime_bootstrap():
    return ("import sys, runpy; assert sys.flags.utf8_mode == 1; "
            "assert sys.stdout.encoding.lower().replace('-', '') == 'utf8'; "
            "sys.stdout.reconfigure(errors='strict', newline='\\n'); "
            "sys.stderr.reconfigure(errors='strict', newline='\\n'); "
            "sys.path.insert(0, '.'); "
            "import io, os\n"
            "with io.open_code(os.path.abspath('main.py')) as source:\n"
            "    code = compile(source.read(), 'main.py', 'exec')\n"
            "runpy._run_module_code(code, mod_name='__main__', pkg_name='', script_name='main.py')")


def expected_runtime_bootstrap_sha256():
    return hashlib.sha256(expected_runtime_bootstrap().encode()).hexdigest()





def expected_action_bootstrap():
    return (
        'import sys; '
        'assert sys.flags.utf8_mode == 1; '
        "assert sys.stdout.encoding.lower().replace('-', '') == 'utf8'; "
        "sys.stdout.reconfigure(errors='strict', newline='\\n'); "
        "sys.stderr.reconfigure(errors='strict', newline='\\n'); "
        "sys.path.insert(0, '.'); "
        'import io, os\n'
        "with io.open_code(os.path.abspath('main.py')) as source:\n"
        "    code = compile(source.read(), 'main.py', 'exec')\n"
        "def _run_source(code):\n"
        "    temporary_main = type(sys)('__main__')\n"
        "    original_main = sys.modules['__main__']\n"
        "    sys.modules['__main__'] = temporary_main\n"
        "    try:\n"
        "        original_argv0 = sys.argv[0]\n"
        "        sys.argv[0] = 'main.py'\n"
        "        try:\n"
        "            namespace = temporary_main.__dict__\n"
        "            namespace.update(__name__='__main__', __file__='main.py',\n"
        "                             __cached__=None, __doc__=None, __loader__=None,\n"
        "                             __package__='', __spec__=None)\n"
        "            exec(code, namespace)\n"
        "        finally:\n"
        "            sys.argv[0] = original_argv0\n"
        "    finally:\n"
        "        sys.modules['__main__'] = original_main\n"
        "    return namespace.copy()\n"
        "_run_source(code)"
    )


def expected_action_bootstrap_sha256():
    return hashlib.sha256(expected_action_bootstrap().encode()).hexdigest()


def validate_bootstrap_qualification(execution):
    """Authenticate eight reviewed temporary-main pairs, never reexecute them."""
    policy = 'regular_source_public_temporary_module_v1'
    assert execution['qualification_sha256'] == '4c3a6438549ce1ecd4c8d92509f1a60753692b8aab1ab9155df00b12db10941c'
    assert execution['bootstrap_policy'] == policy
    assert execution['bootstrap_sha256'] == expected_action_bootstrap_sha256()
    path = pathlib.Path(execution['bootstrap_qualification_path'])
    assert path.is_absolute() and sha(path) == execution['bootstrap_qualification_sha256']
    manifest = read(path)
    assert set(manifest) == {'schema','status','runtime_qualification_sha256','bridge_sha256',
        'bootstrap_policy','bootstrap_sha256','paired_cases','child_executions','evidence'}
    assert manifest['schema'] == 'temporary-module-bootstrap-qualification-v1' and manifest['status'] == 'passed'
    assert manifest['runtime_qualification_sha256'] == execution['qualification_sha256']
    assert manifest['bridge_sha256'] == sha(ROOT/'experiments/r4_integrated/temporary_module_action_bridge_v1.py')
    assert manifest['bootstrap_policy'] == policy
    assert manifest['bootstrap_sha256'] == expected_action_bootstrap_sha256()
    assert type(manifest['paired_cases']) is int and manifest['paired_cases'] == 8
    assert type(manifest['child_executions']) is int and manifest['child_executions'] == 16
    assert isinstance(manifest['evidence'],dict) and manifest['evidence']
    for evidence, expected in manifest['evidence'].items():
        assert pathlib.Path(evidence).is_absolute() and sha(evidence) == expected
    return {'bootstrap_qualification_sha256':execution['bootstrap_qualification_sha256'],
        'paired_cases':8,'child_executions':16,'authenticated_evidence_files':len(manifest['evidence'])}

def validate_action_launch(execution, launch):
    """Scalar identity only per action; fixed file hashing occurs at boundaries."""
    assert execution['bootstrap_policy'] == 'regular_source_public_temporary_module_v1'
    assert execution['bootstrap_sha256'] == expected_action_bootstrap_sha256()
    assert execution['launch_record_sha256'] == hashlib.sha256(canonical(launch)).hexdigest()
    assert execution['child_command'] == [launch['base_executable'], '-I', '-S', '-X', 'utf8', '-B', '-c', expected_action_bootstrap()]
    assert execution['child_environment_override'] == {}
    assert execution['child_environment_removed'] == ['__PYVENV_LAUNCHER__']
    assert execution['child_python_flags'] == ['-I','-S','-X','utf8','-B']


def validate_launch_record(launch):
    assert set(launch) == {'policy','parent_runtime','base_executable','flags',
        'bootstrap_policy','bootstrap_sha256','environment_removed','artifacts'}
    assert launch['policy'] == 'base_stdlib_python_no_site_v1'
    validate_parent_runtime(launch['parent_runtime'])
    assert launch['base_executable'] == launch['parent_runtime']['base_executable']
    assert launch['flags'] == ['-I','-S','-X','utf8','-B']
    assert launch['environment_removed'] == ['__PYVENV_LAUNCHER__']
    assert launch['bootstrap_policy'] == 'regular_source_runpy_module_code_v1'
    assert launch['bootstrap_sha256'] == expected_runtime_bootstrap_sha256()
    expected = dict(launch['parent_runtime']['artifacts'])
    for name in ('stdlib_python_action_bridge_v1.py','source_python_action_bridge_v1.py'):
        path = (ROOT/'experiments/r4_integrated'/name).resolve()
        expected[str(path)] = sha(path)
    assert launch['artifacts'] == expected
    return hashlib.sha256(canonical(launch)).hexdigest()


def validate_parent_runtime(launch):
    assert set(launch) == {'policy','venv_executable','base_executable','venv_config',
        'python_version','implementation','base_prefix','base_exec_prefix','artifacts'}
    assert launch['policy'] == 'direct_base_python_with_selected_venv_launcher_v1'
    assert launch['implementation'] == 'cpython' and launch['python_version'].startswith('3.11.')
    for key in ('venv_executable','base_executable','venv_config','base_prefix','base_exec_prefix'):
        assert pathlib.Path(launch[key]).is_absolute()
    venv, base, config = (pathlib.Path(launch[key]).resolve() for key in
                         ('venv_executable','base_executable','venv_config'))
    assert venv == (ROOT/'.venv/Scripts/python.exe').resolve() and base != venv
    assert config == (ROOT/'.venv/pyvenv.cfg').resolve()
    entries = dict(line.split(' = ',1) for line in config.read_text().splitlines() if ' = ' in line)
    assert pathlib.Path(entries['home']).resolve() == base.parent
    assert pathlib.Path(launch['base_prefix']).resolve() == base.parent
    assert pathlib.Path(launch['base_exec_prefix']).resolve() == base.parent
    expected = {venv,base,config,(ROOT/'experiments/r4_integrated/direct_python_action_bridge_v1.py').resolve(),base.parent/'python311.dll'}
    expected.update(base.parent/'Lib'/name for name in ('runpy.py','site.py','subprocess.py',
        'encodings/__init__.py','multiprocessing/popen_spawn_win32.py'))
    artifacts = launch['artifacts']
    assert {pathlib.Path(path).resolve() for path in artifacts} == expected
    assert len(artifacts) == len(expected)
    for path, expected_hash in artifacts.items():
        assert pathlib.Path(path).is_absolute() and sha(path) == expected_hash
    return hashlib.sha256(canonical(launch)).hexdigest()


def validate_qualification(path, expected_hash, launch):
    """Authenticate reviewed identity/semantic/isolation evidence, never execute it."""
    path = pathlib.Path(path)
    assert path.is_absolute() and sha(path) == expected_hash
    qualification = read(path)
    assert set(qualification) == {'schema','status','launch_record','executions',
        'identity','semantic_pairs','contamination','evidence'}
    assert qualification['schema'] == 'stdlib-python-action-qualification-v1'
    assert qualification['status'] == 'passed' and qualification['launch_record'] == launch
    assert qualification['executions'] == {'identity':1,'semantic_pairs':14,'contamination':2}
    assert all(type(value) is int for value in qualification['executions'].values())
    evidence = qualification['evidence']
    assert isinstance(evidence,dict) and evidence
    for filename, expected in evidence.items():
        assert pathlib.Path(filename).is_absolute() and sha(filename) == expected
    names = ('namespace_cleanup','utf8_bom_crlf','latin1_cookie','syntax_error',
             'runtime_error','exit_invalid_utf8','pending')
    assert len(qualification['semantic_pairs']) == 7
    for name in ('identity','contamination'):
        item = qualification[name]
        assert set(item) == {'passed','receipt_path','receipt_sha256'}
        assert item['passed'] is True
        assert pathlib.Path(item['receipt_path']).is_absolute()
        assert evidence[item['receipt_path']] == item['receipt_sha256']
    for index, (name,item) in enumerate(zip(names,qualification['semantic_pairs'],strict=True)):
        assert set(item) == {'passed','receipt_path','receipt_sha256','case','case_index'}
        assert item['passed'] is True and item['case'] == name
        assert type(item['case_index']) is int and item['case_index'] == index
        assert pathlib.Path(item['receipt_path']).is_absolute()
        assert evidence[item['receipt_path']] == item['receipt_sha256']
    return {'qualification_sha256':expected_hash,'identity_executions':1,
            'semantic_executions':14,'contamination_executions':2,
            'authenticated_evidence_files':len(evidence)}





def positive(value):
    require(number(value) > 0, "expected positive resource measurement")
    return float(value)


def project_cost(report, outer, *, audit=None, audit_outer=None):
    """Independent separate maxima, including the replicated timing append."""
    require(report["mode"] == "cost" and report["status"] == "completed", "incomplete cost")
    require((audit is None) == (audit_outer is None), "unfinished cost-audit monitor")
    require(len(report["category_probes"]) == 4, "missing action category coverage")
    category = max(positive(item["action_seconds"]) for item in report["category_probes"])
    maxima, regular_total = {}, 0.0
    for arm in ARMS:
        phases = report["timing_summary"][arm]
        require(phases["regular"]["rows"] == 4 and phases["cap"]["rows"] == 1,
                "incomplete cost learning/cap scope")
        maxima[arm] = {}
        for stage in ("model", "action", "journal"):
            samples = []
            for phase, count in (("regular", 4), ("cap", 1)):
                total = positive(phases[phase][stage + "_seconds"])
                maximum = positive(phases[phase][stage + "_max_seconds"])
                require(maximum <= total <= count * maximum + 1e-8, "invalid stage summary")
                samples.append(maximum)
                if phase == "regular":
                    regular_total += total
            if stage == "action":
                samples.append(category)
            maxima[arm][stage] = max(samples)
    residual = positive(outer["elapsed_seconds"]) - regular_total
    require(residual >= 0, "cost stage sums exceed outer duration")
    run_seconds = 512 * sum(sum(row.values()) for row in maxima.values()) + 2 * residual + 30
    parent_worker = positive(report["memory"]["peak_working_set_mib"]) + positive(
        report["worker"]["worker"]["memory"]["peak_working_set_mib"])
    run_peak = max(parent_worker, positive(outer["observed_tree_peak_sum_mib"])) + 12 + 128
    audit_seconds = audit_peak = None
    if audit is not None:
        require(audit["status"] == "passed", "independent cost audit failed")
        check_monitor(audit_outer, pid=audit["pid"])
        sampled, maximum_sum = 0.0, 0.0
        for arm in ARMS:
            phases = audit["learning_pair_timing"][arm]
            require(phases["regular"]["rows"] == 4 and phases["cap"]["rows"] == 1,
                    "audit omitted regular/cap pairs")
            for phase, count in (("regular", 4), ("cap", 1)):
                row = phases[phase]
                require(0 < number(row["max_seconds"]) <= number(row["seconds"])
                        <= count * number(row["max_seconds"]) + 1e-8, "invalid audit pair timing")
            sampled += phases["regular"]["seconds"]
            maximum_sum += max(phases[phase]["max_seconds"] for phase in ("regular", "cap"))
        remainder = positive(audit_outer["elapsed_seconds"]) - sampled
        require(remainder >= 0, "audit pair sums exceed monitor duration")
        audit_seconds = 2 * (256 * maximum_sum + remainder)
        audit_peak = positive(audit_outer["observed_tree_peak_sum_mib"]) + 12 + 32
    total = run_seconds + (audit_seconds or 0)
    peak = max(run_peak, audit_peak or 0)
    return dict(per_arm=maxima, category_action_seconds=category,
                projected_run_seconds=run_seconds, projected_audit_seconds=audit_seconds,
                projected_seconds=total, run_projected_peak_mib=run_peak,
                audit_projected_peak_mib=audit_peak, projected_tree_mib=peak,
                audit_peak_pending=audit is None,
                admissible=audit is not None and total <= 900 and peak <= 2048)


def audit(*args, **kwargs):
    """Full source/launch/report dispatcher remains guarded until schema review."""
    raise RuntimeError("DRAFT: parent report and complete custody integration pending")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cost-dir", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--admission-sha", required=True)
    args = parser.parse_args()
    if DRAFT:
        raise RuntimeError("draft checker cannot audit genuine artifacts")
    result = {"schema": "neuronal-splice-independent-audit-v1", "pid": os.getpid()}
    audit(args, result)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
