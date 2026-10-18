"""Explicit optional network retrieval into a NEW external directory; never publish it."""
from pathlib import Path
import argparse,json,hashlib,urllib.request
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
if out.exists() or out.is_relative_to(R):p.error('Use a new directory outside the release')
out.mkdir(parents=True);seen=set()
for row in json.loads((R/'EXTERNAL_DEPENDENCIES.json').read_text())['files']:
    if row['filename'] in seen:continue
    seen.add(row['filename'])
    request=urllib.request.Request(row['url'],headers={'User-Agent':'EvidencePress-Replay/4.8 (+https://evidencepress.org)'})
    with urllib.request.urlopen(request,timeout=60) as response:data=response.read(row['bytes']+1)
    if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:raise RuntimeError('Upstream identity mismatch')
    (out/row['filename']).write_bytes(data)
print('Exact dependencies retrieved. No redistribution permission is inferred.')
