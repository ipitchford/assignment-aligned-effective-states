"""Evaluate already frozen contrast selections; do not select or fit on test."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from affine_audit import minimax_fit

HERE=Path(__file__).resolve().parent
APP=HERE.parent/'application'


def run():
    freeze=json.loads((HERE/'freeze.json').read_text())
    for rel,expected in freeze['files'].items():
        if hashlib.sha256((HERE/rel).read_bytes()).hexdigest()!=expected:
            raise ValueError('frozen file changed: '+rel)
    results=[]
    for seed in [17,29,43]:
        disc=np.load(APP/f'seed-{seed}'/'discovery_bank.npz')
        a=np.c_[np.ones(len(disc['A'])),disc['A']]
        yy=disc['response_numerator']/int(disc['response_denominator'])
        base=np.column_stack([minimax_fit(a,yy[:,k])['coef'] for k in range(10)])
        base=np.rint(base*2**32)/2**32
        spec=np.load(HERE/f'seed-{seed}'/'selected_decoder.npz')
        beta=spec['coef_numerator']/int(spec['coef_denominator'])
        for name in ['seen_strength','unseen_strength','natural_pairs']:
            p=APP/f'seed-{seed}'/f'test_bank_{name}.npz'
            z=np.load(p);n=len(z['A']);y=z['response_numerator']/int(z['response_denominator'])
            designs={'native8_minimax':np.c_[np.ones(n),z['A']],
                     'contrast12_minimax':np.c_[np.ones(n),z['candidate_A'][:,spec['selected']]]}
            for key,design in designs.items():
                coef=base if key=='native8_minimax' else beta
                residual=design@coef-y
                results.append({'seed':seed,'bank':name,'method':key,'n':n,
                  'sum_squared_error_mean':float(np.mean(np.sum(residual**2,axis=1))),
                  'maximum_absolute_error':float(np.max(np.abs(residual))),
                  'fraction_all_outputs_within_1':float(np.mean(np.max(np.abs(residual),axis=1)<=1)),
                  'test_bank_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    receipt={'evaluated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
      'selection_freeze_sha256':hashlib.sha256((HERE/'freeze.json').read_bytes()).hexdigest(),
      'scope':'Frozen supplementary affine-minimax predictor; no test fitting or feature selection',
      'results':results}
    (HERE/'locked_results.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':run()
