"""Run one replay child with Python-audited original-workspace access denied.

This is an accidental-read guard, not an OS sandbox. Native libraries may perform
I/O outside Python audit events; the archive and path checks remain separate.
"""
from pathlib import Path
import json
import os
import runpy
import sys


if sys.flags.optimize:
    raise RuntimeError('replay checks require Python assertions enabled')
denied_roots = [Path(x).resolve() for x in json.loads(os.environ['REPLAY_DENIED_ROOTS'])]


def guard(event, arguments):
    if event not in {'open', 'os.listdir', 'os.scandir', 'os.chdir'} or not arguments:
        return
    raw = arguments[0]
    if not isinstance(raw, (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(raw)).resolve()
    if any(path == root or path.is_relative_to(root) for root in denied_roots):
        raise PermissionError('replay guard denied original-workspace access: ' + str(path))


sys.addaudithook(guard)
for root in denied_roots:
    try:
        open(root / '__replay_guard_probe__', 'rb')
    except PermissionError:
        pass
    else:
        raise RuntimeError('original-workspace access guard did not reject probe')
print(json.dumps({'replay_guard': 'ACTIVE', 'denied_original_roots': len(denied_roots),
                  'python_optimize': sys.flags.optimize}), flush=True)
script = Path(sys.argv[1]).resolve()
sys.argv = sys.argv[1:]
sys.path.insert(0, str(script.parent))
runpy.run_path(str(script), run_name='__main__')
