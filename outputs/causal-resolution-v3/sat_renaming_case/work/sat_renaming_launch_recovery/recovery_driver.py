"""One clerical launch recovery; immutable science and original absolute deadline."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

WORK = Path(__file__).absolute().parent
ORIGINAL = WORK / "sat_renaming"
ROOT = WORK / "sat_renaming_launch_recovery"
DEADLINE = dt.datetime.fromisoformat("2026-10-16T00:19:26.394306+00:00")
FREEZE_SHA = "3a771ecc50d9481c3fceaca79da5a0965e99b77e681ace6a5f628025b2433ced"
SELECTED_SHA = "041cfd909fd34b75562f794bd1b4bf6aadb1e5e9b67b274ff35d92e4fb2a2d0d"
ADMISSION_SHA = "ab954fdb0efae664fa07d139f165cace09d3b75b27a65b947e61d7dd0be164d9"

def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""): h.update(b)
    return h.hexdigest()

def utc(): return dt.datetime.now(dt.timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def save(name, value):
    with (ROOT / name).open("x") as f: json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n")
def record(p): return {"path": os.path.relpath(p, ROOT), "sha256": sha(p), "bytes": p.stat().st_size}
def remaining(): return (DEADLINE-dt.datetime.now(dt.timezone.utc)).total_seconds()
def event(value):
    with (ROOT / "execution_events.jsonl").open("a") as f:
        f.write(json.dumps(value, sort_keys=True)+"\n"); f.flush(); os.fsync(f.fileno())

def main():
    if remaining() <= 0: raise RuntimeError("Original native deadline expired; no recovery launch")
    assert sha(ORIGINAL / "freeze.json") == FREEZE_SHA
    assert sha(ORIGINAL / "selected_data_freeze.json") == SELECTED_SHA
    admission = WORK / "next_phase/sat_launch_recovery_admission.md"
    assert sha(admission) == ADMISSION_SHA
    old = read(ORIGINAL / "execution.json")
    assert old["status"] == "UNRESOLVED_RUNTIME_FAILURE"
    assert old["batch_attempts"] == old["completed_input_forwards"] == old["conservative_input_forward_charge"] == old["learner_fits"] == 0
    prohibited = ["native_initial_started.json", "native_replay_started.json", "native_initial_events.jsonl", "native_replay_events.jsonl", "native_outputs.npz", "replay_outputs.npz", "native_batches", "replay_batches"]
    assert all(not (ORIGINAL / n).exists() for n in prohibited)
    assert "ModuleNotFoundError: No module named 'torch'" in (ORIGINAL / "native_initial.log").read_text()
    implementation, selected = read(ORIGINAL / "freeze.json"), read(ORIGINAL / "selected_data_freeze.json")
    for row in implementation["files"] + selected["files"]:
        assert sha(ORIGINAL / row["path"]) == row["sha256"]
    ROOT.mkdir(exist_ok=False)
    copy_names = {"freeze.json", "selected_data_freeze.json", "prior_ledger_tokens.npz", "prior_ledger.json"}
    copy_names.update(r["path"] for r in implementation["files"] if not r["path"].startswith("../"))
    copy_names.update(r["path"] for r in selected["files"])
    for name in sorted(copy_names):
        assert "/" not in name
        shutil.copyfile(ORIGINAL / name, ROOT / name)
        assert sha(ROOT / name) == sha(ORIGINAL / name)
    for row in implementation["files"] + selected["files"]:
        assert sha(ROOT / row["path"]) == row["sha256"]
    provenance = ROOT / "original_failure_provenance"; provenance.mkdir()
    for name in ["execution.json", "execution_started.json", "execution_events.jsonl", "native_initial.log", "REPORT.md", "MANIFEST.json"]:
        shutil.copyfile(ORIGINAL / name, provenance / name)
    shutil.copyfile(admission, provenance / admission.name)
    shutil.copyfile(__file__, ROOT / "recovery_driver.py")
    environment = read(ROOT / "environment.json")
    native_python = os.path.abspath(ROOT / environment["native"]["executable_relative"])
    analysis_python = environment["analysis"]["executable"]
    save("recovery_freeze.json", {"utc": utc(), "original_native_deadline": DEADLINE.isoformat(), "original_freeze_sha256": FREEZE_SHA,
        "selected_freeze_sha256": SELECTED_SHA, "admission_sha256": ADMISSION_SHA, "original_failure_sha256": sha(ORIGINAL / "execution.json"),
        "driver_sha256": sha(ROOT / "recovery_driver.py"), "unchanged_copies": [record(ROOT / n) for n in sorted(copy_names)],
        "no_original_native_artifacts": prohibited, "native_executable_lexical": native_python, "native_forward_ceiling": 528, "learner_fit_ceiling": 0,
        "new_proposals": 0, "scientific_code_changes": 0, "analysis_seconds_after_replay": 300, "no_second_recovery": True})
    save("execution_started.json", {"utc": utc(), "freeze_sha256": FREEZE_SHA, "recovery_freeze_sha256": sha(ROOT / "recovery_freeze.json"), "native_deadline": DEADLINE.isoformat()})
    with (ROOT / "execution_events.jsonl").open("x"): pass
    start = time.monotonic(); commands = []; status = "UNRESOLVED_RUNTIME_FAILURE"; detail = None
    def run(executable, script, extra, seconds, threads):
        if seconds <= 0: raise TimeoutError("Original phase deadline exhausted")
        command = [executable, "-B", str(ROOT / script), *extra]
        label = script.removesuffix(".py") + ("_"+extra[0] if extra else "")
        logfile = label + ".log"
        env = os.environ.copy(); env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), NUMEXPR_NUM_THREADS=str(threads))
        event({"event": "STAGE_ATTEMPT", "utc": utc(), "command": command, "timeout_seconds": seconds, "log": logfile})
        begin = time.monotonic()
        with (ROOT / logfile).open("x") as output:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=seconds, check=False)
        receipt = {"command": command, "log": logfile, "returncode": result.returncode, "elapsed_seconds": time.monotonic()-begin, "utc": utc()}
        commands.append(receipt); event(dict(event="STAGE_COMPLETE", **receipt))
        if result.returncode: raise RuntimeError("Subprocess failed without retry: " + logfile)
    try:
        seconds = remaining(); run(native_python, "native.py", ["initial", "--seconds", str(seconds)], seconds, 2)
        seconds = remaining(); run(native_python, "native.py", ["replay", "--seconds", str(seconds)], seconds, 2)
        if read(ROOT / "native_replay.json")["status"] != "PASS": status = "UNRESOLVED_REPLAY_FAILURE"
        else:
            run(analysis_python, "analyze.py", [], 300, 1)
            status = read(ROOT / "analysis.json")["status"]
    except Exception as exc:
        detail = repr(exc)
        event({"event": "STOP_NO_RETRY", "utc": utc(), "detail": detail})
    finally:
        attempts = charge = completed = 0
        for stage in ["initial", "replay"]:
            p = ROOT / ("native_"+stage+"_events.jsonl")
            if p.exists():
                for line in p.read_text().splitlines():
                    row = json.loads(line)
                    if row["event"] == "BATCH_ATTEMPT": attempts += 1; charge += row["conservative_input_forward_charge"]
                    elif row["event"] == "BATCH_COMPLETE": completed += row["input_forwards"]
        assert charge <= 528 and completed <= charge
        save("execution.json", {"status": status, "utc": utc(), "detail": detail, "commands": commands, "elapsed_seconds": time.monotonic()-start,
            "freeze_sha256": FREEZE_SHA, "recovery_freeze_sha256": sha(ROOT / "recovery_freeze.json"), "original_native_deadline": DEADLINE.isoformat(),
            "conservative_input_forward_charge": charge, "completed_input_forwards": completed, "batch_attempts": attempts, "learner_fits": 0,
            "new_proposals": 0, "no_retry": True, "original_attempt_status": old["status"]})
        if not (ROOT / "REPORT.md").exists():
            with (ROOT / "REPORT.md").open("x") as f: f.write("# SAT renaming: clerical recovery\n\nStatus: **"+status+"**.\n\n"+(detail or "")+"\n")
        for row in implementation["files"] + selected["files"]: assert sha(ROOT / row["path"]) == row["sha256"]
        files = sorted(p for p in ROOT.rglob("*") if p.is_file() and p != ROOT / "MANIFEST.json")
        save("MANIFEST.json", {"utc": utc(), "status": status, "files": [record(p) for p in files], "total_bytes": sum(p.stat().st_size for p in files)})
    print(json.dumps(read(ROOT / "execution.json")))

if __name__ == "__main__": main()
