"""Replay advertised checks on a separate copy of the delivered bundle.

Usage: python src/verify_bundle.py NEW_EMPTY_PATH
The source is never modified. Failed checks retain logs in the copied bundle.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

SOURCE = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    target = Path(sys.argv[1]).resolve()
    if target.exists():
        raise SystemExit('Destination must not already exist.')
    if target == SOURCE or SOURCE in target.parents:
        raise SystemExit('Destination must be outside the source bundle.')
    shutil.copytree(SOURCE, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    logs = target/'provenance'/'fresh-replay'
    logs.mkdir(parents=True, exist_ok=True)
    files_checked = 0
    release = target/'MANIFEST.json'
    if release.exists():
        for name, digest in json.loads(release.read_text())['files'].items():
            if sha(target/name) != digest:
                raise ValueError('Release manifest mismatch: '+name)
            files_checked += 1
    commands = [
        ['certificates/verify_exact.py'],
        ['nonlinear_certificates/verify_exact.py'],
        ['src/replay_discovery.py'],
        ['application/verify.py'],
        ['nonlinear_certificates/replay_model.py'],
        ['matched_controls/verify.py'],
        ['theory/precision_optimality.py', '--verify',
         'theory/results/precision_optimality_sensitivity.json'],
    ]
    optional = [
        'prospective_theory/verify.py',
        'prospective_application/verify.py',
    ]
    for name in optional:
        if (target/name).exists():
            commands.append([name])
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(logs/'matplotlib'))
    results = []
    for i, args in enumerate(commands):
        proc = subprocess.run([sys.executable, '-B', *args], cwd=target,
                              env=env, capture_output=True, text=True)
        (logs/f'check-{i}.log').write_text(proc.stdout+proc.stderr)
        result = {'command': args, 'returncode': proc.returncode,
                  'log': str(Path('provenance/fresh-replay')/f'check-{i}.log')}
        results.append(result)
        print(json.dumps(result), flush=True)
        if proc.returncode:
            raise SystemExit('Replay failed; retained logs: '+str(logs))
    result = {'status': 'PASS', 'manifest_files_checked': files_checked,
              'source': str(SOURCE), 'copy': str(target), 'checks': results,
              'scope': 'fresh-location computational replay; no retraining or external scientific replication'}
    (logs/'receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status': 'PASS', 'checks': len(results), 'receipt': str(logs/'receipt.json')}))


if __name__ == '__main__':
    main()
