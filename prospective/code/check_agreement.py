"""Recover binary predictions from complete truth-error sets; no model fitting."""
from pathlib import Path
import argparse,json,hashlib,itertools
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--release-root',required=True);ap.add_argument('--diagnostic',required=True);ap.add_argument('--native-arrays',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
ROOT=Path(args.release_root);path=Path(args.diagnostic)
data=json.loads(path.read_text())
with np.load(ROOT/'work/next_phase/sat_all17_validation_v3_20261018/selected_bank.npz',allow_pickle=False) as z:tokens=z['unique_tokens']
with np.load(args.native_arrays,allow_pickle=False) as z:
    logits=z['logits']
    if logits.shape!=(7252,16) or not np.isfinite(logits).all():raise RuntimeError('native schema')
    native=logits.argmax(1)
truth=np.zeros(len(tokens),bool)
for bits in itertools.product((False,True),repeat=5):
    lit=np.array(bits+tuple(not b for b in bits));truth|=(lit[tokens[:,1:40:4]]|lit[tokens[:,2:40:4]]).all(1)
rows={}
for setting,representations in data['cross_validated_refitted_readout']['matched_preprocessing'].items():
    rows[setting]={}
    for name in ('threshold_bits','continuous_activations'):
        summary=representations[name]['nested'];ids=np.asarray(summary['error_row_ids'],int)
        if len(ids)!=summary['errors'] or len(np.unique(ids))!=len(ids) or np.any((ids<0)|(ids>=len(tokens))):raise RuntimeError('error-index schema')
        predicted=truth.copy();predicted[ids]=~predicted[ids];decoded=np.where(predicted,13,14)
        mismatch=np.flatnonzero(decoded!=native)
        rows[setting][name]={'truth_errors':len(ids),'native_argmax_agreement_count':int((decoded==native).sum()),'total':len(tokens),'native_argmax_agreement_rate':float((decoded==native).mean()),'native_disagreement_row_ids':mismatch.tolist()}
record={'status':'PASS','scope':'original protected bank; exact reconstruction of binary classifier predictions from freshly replayed complete truth-error index lists, not recovered fitted coefficients or fresh-bank deployment','output_space':'binary SAT13 / UNSAT14, compared to all16 native vocabulary logits argmax','source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'native_nonanswer_argmax':int((~np.isin(native,[13,14])).sum()),'results':rows}
with Path(args.out).open('x') as f:json.dump(record,f,indent=2)
print(json.dumps({**record,'results':{setting:{name:{k:v for k,v in row.items() if k!='native_disagreement_row_ids'} for name,row in reps.items()} for setting,reps in rows.items()}},indent=2))
