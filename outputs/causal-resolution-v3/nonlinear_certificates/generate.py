"""Exploratory discovery-only affine expressivity certificates.

Reads discovery banks only. No training, feature selection, or test access.
Numerical LP solutions propose witnesses and coefficients; exact fractions
certify their published lower and upper bounds.
"""
from pathlib import Path
from fractions import Fraction as Q
from datetime import datetime, timezone
import hashlib,json,sys
import numpy as np

HERE=Path(__file__).resolve().parent
APP=HERE.parent/'nonlinear_application'
sys.path.insert(0,str(HERE.parent/'certificates'))
from affine_audit import minimax_fit,rational_witness

SEEDS=(117,129,143)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def exact_primal(A,nums,den,coef):
    q=2**32
    beta=[round(float(v)*q) for v in coef]
    error=max(abs(int(y)*q-den*sum(int(x)*b for x,b in zip(row,beta))) for row,y in zip(A,nums))
    return beta,Q(error,den*q)

def main():
    summaries=[]
    for seed in SEEDS:
        path=APP/f'seed-{seed}'/'discovery_bank.npz'
        z=np.load(path,allow_pickle=False)
        A=z['A'];nums=z['response_numerator'];den=int(z['response_denominator'])
        if A.shape!=(1904,8) or nums.shape!=(1904,10) or den!=43980465111040:
            raise ValueError('unexpected nonlinear discovery-bank dimensions or denominator')
        seen=set();ids=[]
        for i,row in enumerate(A):
            key=tuple(int(v) for v in row)
            if key not in seen:
                seen.add(key);ids.append(i)
        design=np.column_stack([np.ones(len(ids),dtype=int),A[ids]])
        target=nums[ids]
        fits=[minimax_fit(design,target[:,k]/den) for k in range(10)]
        candidates=[]
        for k,fit in enumerate(fits):
            cert=rational_witness(design,target[:,k],den,fit['dual'])
            if cert is not None:
                candidates.append((Q(cert['lower_bound']),-k,cert))
        if not candidates:
            raise ValueError('no positive exact affine obstruction was found')
        lower,minus_k,cert=max(candidates,key=lambda row:row[:2]);k=-minus_k
        cert['exact_design_rows']=design[cert['rows']].tolist()
        cert['exact_response_numerators']=target[cert['rows'],k].tolist()
        primals=[exact_primal(design,target[:,j],den,fits[j]['coef']) for j in range(10)]
        upper=max(value for _,value in primals)
        if not 0<=upper-lower<Q(1,10**7):
            raise ValueError('exact lower and joint primal upper do not form a tight bracket')
        record={
            'seed':seed,
            'analysis':'exploratory post-test supplementary analysis of frozen discovery cases only; no training, feature selection, or test outcomes read',
            'source_bank':f'seed-{seed}/discovery_bank.npz',
            'source_bank_sha256':sha(path),
            'original_bank_rows':len(A),
            'subbank_selection':'first discovery row for each distinct native8 signed-difference vector; affine intercept included',
            'subbank_original_row_ids':ids,
            'subbank_rows':len(ids),
            'lookup_error_exact':0,
            'output':k,
            'certificate':cert,
            'witness_original_bank_row_ids':[ids[i] for i in cert['rows']],
            'witness_metadata':z['meta'][[ids[i] for i in cert['rows']]].tolist(),
            'primal':{'design':design.tolist(),'response_numerators':target.tolist(),
                      'response_denominator':den,'coefficient_numerators':[b for b,_ in primals],
                      'coefficient_denominator':2**32,'upper_bounds':[str(v) for _,v in primals]},
            'joint_lower_bound':str(lower),'joint_upper_bound':str(upper),
            'primary_tolerance':'1','rejects_primary_tolerance':lower>1,
            'scope':'finite discovery-subbank affine impossibility; injective reports make lookup error zero; no prevalence, semantics, feature-selection optimality, or transfer claim'}
        (HERE/f'seed-{seed}.json').write_text(json.dumps(record,indent=2)+'\n')
        summary={'seed':seed,'subbank_rows':len(ids),'witness_cases':cert['support_size'],'output':k,
                 'lower':str(lower),'lower_float':float(lower),'upper':str(upper),
                 'rejects_epsilon1':lower>1}
        summaries.append(summary);print(json.dumps(summary),flush=True)
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'analysis':'exploratory discovery-only supplement',
            'test_outcomes_accessed':False,'models_trained':False,'features_selected':False,
            'generator_sha256':sha(Path(__file__)),
            'affine_audit_sha256':sha(HERE.parent/'certificates/affine_audit.py'),
            'results':summaries}
    (HERE/'summary.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
