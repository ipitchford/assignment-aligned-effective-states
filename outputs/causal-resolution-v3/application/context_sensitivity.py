#!/usr/bin/env python3
"""Post-test descriptive oracle sensitivity; no decoder refitting or selection.

This asks whether the fixed report still loses information if a decoder is
allowed the intervention group and strength.  It does not amend the original
frozen comparison or claim prospectively evaluated fitted performance.
"""
from pathlib import Path
import json,numpy as np
import experiment as ex
P=Path(__file__).resolve().parent
out={'status':'post-test descriptive exact oracle sensitivity; no fitting','context_cost':'four group ids plus two strengths per bank require3 extra bits per response; union of all4 strengths requires4 extra bits','records':[]}
for seed in ex.SEEDS:
    for bn in ['seen_strength','unseen_strength']:
        b=np.load(P/f'seed-{seed}'/f'test_bank_{bn}.npz')
        code=np.concatenate([b['meta'][:,1:],b['report_before'],b['report_after']],axis=1)
        record=ex.collision_summary(code,b['response_numerator']);record.update({'seed':seed,'bank':bn})
        witness=record['max_range_witness'];record['witness_metadata']=[b['meta'][i].tolist() for i in witness['rows']]
        assert record['witness_metadata'][0][1:]==record['witness_metadata'][1][1:]
        out['records'].append(record)
ex.dump(P/'context_sensitivity.json',out)
print(json.dumps([{'seed':r['seed'],'bank':r['bank'],'minimax':r['exact_minimax_report_error'],'oracle_mse':r['oracle_empirical_mse'],'witness':r['witness_metadata']} for r in out['records']]))
