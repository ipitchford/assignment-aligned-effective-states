#!/usr/bin/env python3
"""Hash retained audit files and the exact pre-existing replay dependencies."""
from pathlib import Path
import hashlib
import json

P = Path(__file__).resolve().parent
ROOT = P.parent

def record(path, base):
    payload = path.read_bytes()
    return {"path": str(path.relative_to(base)), "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}

files = [record(p, P) for p in sorted(P.rglob('*'))
         if p.is_file() and p.name != 'MANIFEST.json'
         and '__pycache__' not in p.parts and not any(x.startswith('.') for x in p.relative_to(P).parts)]
dependencies = [ROOT/'application'/name for name in
                ['experiment.py', 'contract.json', 'train.npz', 'discovery_fit.npz',
                 'discovery_validation.npz', 'test.npz']]
for directory, seeds in [('application', [17, 29, 43]),
                         ('nonlinear_application', [117, 129, 143])]:
    for seed in seeds:
        folder = ROOT/directory/f'seed-{seed}'
        names = ['model.npz', 'report-native-8.npz', 'report-native-12.npz', 'report-pca-8.npz']
        names += ['precision-gain-1.npz'] if directory == 'application' else ['report-random-8.npz', 'precision.npz']
        dependencies.extend(folder/name for name in names)
manifest = {'scope': 'exploratory prospective action audit; retained files plus exact external replay dependencies',
            'status': 'completed; retrospective reused test split; no preregistration claim',
            'files': files, 'file_count': len(files), 'bytes': sum(r['bytes'] for r in files),
            'external_dependencies_relative_to_work': [record(p, ROOT) for p in sorted(dependencies)],
            'verification': json.loads((P/'verification.json').read_text())}
(P/'MANIFEST.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
print(json.dumps({'file_count': len(files), 'bytes': manifest['bytes'],
                  'external_dependency_count': len(dependencies),
                  'manifest_sha256': hashlib.sha256((P/'MANIFEST.json').read_bytes()).hexdigest()}))
