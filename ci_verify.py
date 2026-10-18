"""Fast public byte-identity checks, not the full scientific replay."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
if R.name=='code':R=R.parent
manifest=json.loads((R/'MANIFEST.json').read_text())
for row in manifest['files']:
    p=R/row['path']
    if not p.is_file() or p.is_symlink() or p.stat().st_size!=row['size'] or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
        raise RuntimeError('Public byte identity mismatch: '+row['path'])
if list(R.rglob('*.pkl')):raise RuntimeError('Opaque model pickle is not part of public scope')
mapping=R/'outputs/causal-resolution-v3/PUBLIC_PRESENTATION_DERIVATIVES.json'
for row in json.loads(mapping.read_text())['files']:
    p=R/row['path']
    if hashlib.sha256(p.read_bytes()).hexdigest()!=row['public_sha256']:
        raise RuntimeError('Presentation identity mismatch: '+row['path'])
print(json.dumps({'status':'PASS','files':len(manifest['files']),
 'scope':'Public byte manifest and presentation derivative identities only; not 18-stage replay, native execution, scientific replication or external review.'}))
