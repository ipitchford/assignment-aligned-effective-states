"""One authorized, bounded execution; immutable receipts and no repeat/resume path."""
import argparse
import subprocess
from protocol import *


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args(); frozen()
    start = time.monotonic()
    save_json("execution_started.json", {"utc": utc(), "freeze_sha256": sha(ROOT / "freeze.json"), "native_forward_ceiling": 528, "fit_ceiling": 0})
    if any((ROOT / x).exists() for x in ["construction_started.json", "native_initial_started.json", "analysis_started.json"]):
        raise RuntimeError("No repeat or resume after any scientific stage has started")
    environment = read_json("environment.json")
    analysis_python = environment["analysis"]["executable"]
    native_python = str((ROOT / environment["native"]["executable_relative"]).resolve())
    commands = []
    with (ROOT / "execution_events.jsonl").open("x"): pass
    status, detail = "ERROR_STOP", None

    def run(executable, script, extra, timeout, threads):
        if timeout <= 0: raise TimeoutError("Phase wall budget exhausted before next subprocess")
        command = [executable, "-B", str(ROOT / script), *extra]
        label = script.removesuffix(".py") + ("_"+extra[0] if extra and not extra[0].startswith("--") else "")
        log = label + ".log"
        env = os.environ.copy(); env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), NUMEXPR_NUM_THREADS=str(threads))
        event = {"event": "STAGE_ATTEMPT", "utc": utc(), "command": command, "timeout_seconds": timeout, "log": log}
        append_event("execution_events.jsonl", event)
        phase = time.monotonic()
        try:
            with (ROOT / log).open("x") as output:
                completed = subprocess.run(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=timeout, check=False)
            result = {"command": command, "log": log, "returncode": completed.returncode, "elapsed_seconds": time.monotonic()-phase}
            commands.append(result); append_event("execution_events.jsonl", dict(event="STAGE_COMPLETE", **result))
            if completed.returncode: raise RuntimeError("Subprocess failed without retry: "+log)
        except subprocess.TimeoutExpired:
            result = {"command": command, "log": log, "returncode": None, "status": "WALL_CAP_STOP", "elapsed_seconds": time.monotonic()-phase}
            commands.append(result); append_event("execution_events.jsonl", dict(event="STAGE_TIMEOUT", **result)); raise

    try:
        run(analysis_python, "construct.py", [], CONSTRUCTION_SECONDS, 1)
        construction = read_json("construction.json")
        if construction["status"] != "PASS":
            status = construction["status"]; detail = construction.get("error")
        else:
            selected_frozen()
            native_start = time.monotonic()
            remaining = NATIVE_SECONDS-(time.monotonic()-native_start)
            run(native_python, "native.py", ["initial", "--seconds", str(remaining)], remaining, 2)
            remaining = NATIVE_SECONDS-(time.monotonic()-native_start)
            run(native_python, "native.py", ["replay", "--seconds", str(remaining)], remaining, 2)
            if read_json("native_replay.json")["status"] != "PASS":
                status = "UNRESOLVED_REPLAY_FAILURE"
            else:
                run(analysis_python, "analyze.py", [], ANALYSIS_SECONDS, 1)
                status = read_json("analysis.json")["status"]
    except Exception as exc:
        status = "UNRESOLVED_RUNTIME_FAILURE"; detail = repr(exc)
    finally:
        # Charge attempted batches even if an inference call failed before completion.
        attempt_charge = 0; completed_charge = 0; batch_attempts = 0
        for label in ["initial", "replay"]:
            path = ROOT / ("native_"+label+"_events.jsonl")
            if path.exists():
                for line in path.read_text().splitlines():
                    event = json.loads(line)
                    if event["event"] == "BATCH_ATTEMPT":
                        attempt_charge += event["conservative_input_forward_charge"]; batch_attempts += 1
                    elif event["event"] == "BATCH_COMPLETE": completed_charge += event["input_forwards"]
        save_json("execution.json", {"status": status, "stopped": True, "utc": utc(), "detail": detail, "commands": commands,
                  "elapsed_seconds": time.monotonic()-start, "freeze_sha256": sha(ROOT/"freeze.json"),
                  "conservative_input_forward_charge": attempt_charge, "completed_input_forwards": completed_charge,
                  "batch_attempts": batch_attempts, "learner_fits": 0, "no_retry": True})
        if not (ROOT/"REPORT.md").exists():
            with (ROOT/"REPORT.md").open("x") as f:
                f.write("# SAT renaming: bounded attempt\n\nStatus: **"+status+"**.\n\nThe fixed attempt stopped. Partial artifacts and receipts are retained; no smaller bank, added proposals, replacement inputs, or repeat inference is authorized.\n\n"+(detail or "")+"\n")
        frozen()
        files = sorted(p for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name != "MANIFEST.json")
        save_json("MANIFEST.json", {"utc": utc(), "status": status, "files": [record(p) for p in files], "total_bytes": sum(p.stat().st_size for p in files)})
    print(json.dumps(read_json("execution.json")))


if __name__ == "__main__": main()
