"""One bounded run; Stage B is mechanically gated; no retry or expansion."""
import datetime
import json
import os
import subprocess
import time
from protocol import *


def main():
    frozen()
    if (ROOT/"execution.json").exists() or (ROOT/"original_native.npz").exists():raise RuntimeError("Study has already started")
    start=time.monotonic();commands=[]
    native_python=str(PILOT/".venv/bin/python")
    analysis_python="/Users/admin/.venvs/gaia-tess-microlensing/bin/python"
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2")
    def run(executable,script,stage):
        log=f"{script.removesuffix('.py')}_{stage}.log"
        command=[executable,str(ROOT/script),stage]
        phase=time.monotonic()
        with (ROOT/log).open("x") as output:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
        commands.append(dict(command=command,log=log,returncode=result.returncode,elapsed_seconds=time.monotonic()-phase))
        write_json("execution.json",dict(status="RUNNING" if result.returncode==0 else "ERROR_STOP",commands=commands,started_utc=started,elapsed_seconds=time.monotonic()-start))
        if result.returncode:raise RuntimeError("Execution failed without retry: "+log)
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    run(native_python,"native.py","original")
    run(analysis_python,"analyze.py","a")
    if read_json("stage_a.json")["gate"]["pass"]:
        run(native_python,"native.py","edited")
        run(analysis_python,"analyze.py","b")
    run(native_python,"native.py","replay")
    run(analysis_python,"analyze.py","finalize")
    verify_pilot()
    write_json("execution.json",dict(status="COMPLETE_STOPPED",commands=commands,started_utc=started,elapsed_seconds=time.monotonic()-start,freeze_sha256=sha(ROOT/"freeze.json")))
    print(json.dumps(read_json("summary.json")))


if __name__=="__main__":main()
