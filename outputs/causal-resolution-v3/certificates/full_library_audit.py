"""Retrospective discovery-only full-menu audit with complete primal receipts."""
from pathlib import Path
import gzip,json
import numpy as np
from affine_audit import minimax_fit,rational_witness
from discovery_audit import dump_cert,exact_upper

HERE=Path(__file__).resolve().parent


def run():
    out=[];primal=[]
    for seed in [17,29,43]:
        z=np.load(HERE.parent/'application'/f'seed-{seed}'/'discovery_bank.npz')
        a=np.c_[np.ones(len(z['A']),dtype=int),z['candidate_A']]
        den=int(z['response_denominator']);nums=z['response_numerator'];y=nums/den
        fits=[minimax_fit(a,y[:,k]) for k in range(10)]
        k=max(range(10),key=lambda j:fits[j]['error'])
        c=rational_witness(a,nums[:,k],den,fits[k]['dual'])
        all_upper=[exact_upper(a,nums[:,j],den,fits[j]['coef']) for j in range(10)]
        out.append({'seed':seed,'feature_count':a.shape[1]-1,'output':k,
          'certificate':dump_cert(a,nums[:,k],den,c),
          'exact_upper':str(max(u[2] for u in all_upper)),
          'scope':'Retrospective deterministic discovery-bank certificate; no test outcomes used'})
        primal.append({'seed':seed,'design':a.tolist(),'response_numerators':nums.tolist(),
          'response_denominator':den,'coefficient_numerators':[u[0] for u in all_upper],
          'coefficient_denominator':2**32,'upper_bounds':[str(u[2]) for u in all_upper]})
    (HERE/'full_library_certificates.json').write_text(json.dumps(out,indent=2)+'\n')
    with gzip.open(HERE/'full_library_primal.json.gz','wt') as f:json.dump(primal,f)
    print(json.dumps([{'seed':r['seed'],'lower':r['certificate']['lower_bound_float'],'upper':r['exact_upper']} for r in out],indent=2))


if __name__=='__main__':run()
