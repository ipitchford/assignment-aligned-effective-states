"""Solver-free arithmetic/provenance replay of matched-control results."""
from pathlib import Path
from fractions import Fraction as Q
import hashlib,json
import numpy as np

P=Path(__file__).resolve().parent
ROOT=P.parent/'review-round1'
checks=[]
def check(condition,name):
    if not condition:raise AssertionError(name)
    checks.append(name)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

contract=read(P/'contract.json')
for f,h in contract['inputs'].items():check(sha(ROOT/f)==h,'input '+f)
for f,h in read(P/'freeze.json')['files'].items():check(sha(P/f)==h,'freeze '+f)
for base,receipt in [(ROOT/'application','freeze.json'),(ROOT/'application','precision_freeze.json'),(ROOT/'certificates','freeze.json')]:
    for f,h in read(base/receipt)['files'].items():check(sha(base/f)==h,'original '+f)

discovery=read(P/'discovery_results.json')
test=read(P/'test_results.json')
check(len(discovery['results'])==12,'12 final methods')
check(len(test['results'])==36,'36 test records')
largest_gap=Q();exact_discovery=0;exact_test=0
for r in discovery['results']:
    seed=r['seed'];method=r['method']
    bank=np.load(ROOT/'application'/f'seed-{seed}'/'discovery_bank.npz')
    saved=np.load(P/f'seed-{seed}'/f'{method}.npz')
    a=np.c_[np.ones(len(bank['candidate_A']),dtype=int),bank['candidate_A'][:,saved['selected']]]
    nums=bank['response_numerator'];den=int(bank['response_denominator']);coef=saved['coefficient_numerator'];q=int(saved['coefficient_denominator'])
    check(np.array_equal(a,saved['design']),method+' design provenance')
    check(np.array_equal(nums,saved['response_numerator']) and den==int(saved['response_denominator']),method+' targets provenance')
    check(q==2**32 and coef.shape==(a.shape[1],10),method+' coefficient dimensions')
    check(len(r['outputs'])==10,method+' output count')
    lower=[];upper=[]
    for out in r['outputs']:
        k=out['output'];c=out['certificate'];ix=c['rows'];u=[Q(t) for t in c['weights']]
        check(np.array_equal(a[ix],c['exact_design_rows']),method+' witness design')
        check(np.array_equal(nums[ix,k],c['exact_response_numerators']),method+' witness targets')
        check(sum(abs(w) for w in u)==1,method+' witness norm')
        check(all(sum(w*int(a[i,j]) for w,i in zip(u,ix))==0 for j in range(a.shape[1])),method+' witness nullspace')
        lo=sum(w*Q(int(nums[i,k]),den) for w,i in zip(u,ix))
        check(lo==Q(c['lower_bound']),method+' exact lower')
        # Recompute predictions as Python-integer row sums, independently of the run's matrix expression.
        hi=max(Q(abs(int(y)*q-den*sum(int(x)*int(v) for x,v in zip(row,coef[:,k]))),den*q) for row,y in zip(a,nums[:,k]))
        check(hi==Q(out['exact_upper']),method+' exact upper')
        check(0<=hi-lo<Q(1,10**6),method+' discovery bracket')
        largest_gap=max(largest_gap,hi-lo);lower.append(lo);upper.append(hi);exact_discovery+=1
    check(max(lower)==Q(r['joint_lower']) and max(upper)==Q(r['joint_upper']),method+' joint bracket')
    for name in ['seen_strength','unseen_strength','natural_pairs']:
        path=ROOT/'application'/f'seed-{seed}'/f'test_bank_{name}.npz';b=np.load(path)
        aa=np.c_[np.ones(len(b['A']),int),b['candidate_A'][:,saved['selected']]]
        yn=b['response_numerator'];dd=int(b['response_denominator'])
        e=aa.astype(object)@coef.astype(object)*dd-yn.astype(object)*q
        mx=Q(max(abs(int(v)) for v in e.flat),dd*q)
        rr=[z for z in test['results'] if z['seed']==seed and z['method']==method and z['bank']==name]
        check(len(rr)==1,method+' unique evaluation');rr=rr[0]
        check(sha(path)==rr['test_bank_sha256'],method+' test provenance')
        check(mx==Q(rr['exact_max_error']),method+' exact test maximum')
        mse=float(np.mean(np.sum((np.asarray(e,dtype=float)/(dd*q))**2,axis=1)))
        check(abs(mse-rr['mse_sum_outputs'])<1e-10,method+' test mse')
        exact_test+=1

for seed in [17,29,43]:
    bank=np.load(ROOT/'application'/f'seed-{seed}'/'discovery_bank.npz');selected=list(range(8));p=bank['candidate_A'].shape[1]
    for step in range(1,5):
        t=read(P/f'greedy-seed-{seed}-step-{step}.json');candidates=t['all_candidates']
        check(len(candidates)==p-len(selected),f'{seed}/{step} full candidate count')
        check({r['candidate'] for r in candidates}==set(range(p))-set(selected),f'{seed}/{step} exhaustive candidate identities')
        check(all(len(r['coordinate_errors'])==10 and r['objective']==max(r['coordinate_errors']) for r in candidates),f'{seed}/{step} every coordinate objective')
        best=min(candidates,key=lambda r:(round(r['objective'],9),r['candidate']))
        check(best['candidate']==t['chosen_candidate'],f'{seed}/{step} declared tie rule')
        selected.append(best['candidate']);check(selected==t['selected'],f'{seed}/{step} updated selection')
    check(np.array_equal(np.array(selected),np.load(P/f'seed-{seed}'/'greedy_minimax12.npz')['selected']),f'{seed} final greedy provenance')

receipt={'status':'PASS','checks':len(checks),'solver_called':False,'exact_discovery_output_brackets':exact_discovery,'exact_test_maxima':exact_test,'largest_discovery_bracket_gap':str(largest_gap),'scope':'Frozen inputs/selection traces plus exact arithmetic replay; no assertion of exact optimality of numerical greedy choices, external validity or semantic meaning.','script_sha256':sha(__file__),'discovery_results_sha256':sha(P/'discovery_results.json'),'test_results_sha256':sha(P/'test_results.json')}
(P/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
