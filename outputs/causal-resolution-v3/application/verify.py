#!/usr/bin/env python3
"""Independent integer replay of retained frozen data; no training or fitting.

Uses Python arbitrary-precision integer arithmetic, and computes endpoint
logits separately rather than the experiment's difference shortcut.
"""
from pathlib import Path
from fractions import Fraction
import hashlib,json,sys
import numpy as np

P=Path(__file__).resolve().parent
checked=[]
def check(condition,name):
    if not condition:raise AssertionError(name)
    checked.append(name)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

for receipt in ['freeze.json','precision_freeze.json']:
    lock=json.loads((P/receipt).read_text())
    for f,d in lock['files'].items():check(sha(P/f)==d,f'{receipt}: {f}')

splits={s:np.load(P/f'{s}.npz') for s in ['train','discovery_fit','discovery_validation','test']}
all_ids=np.concatenate([d['ids'] for d in splits.values()])
check(len(all_ids)==1797 and len(np.unique(all_ids))==1797,'split partition and disjointness')
contract=json.loads((P/'contract.json').read_text());groups=contract['interventions']['groups']
test_results=json.loads((P/'test_results.json').read_text());precision_results=json.loads((P/'precision_test_results.json').read_text())
details=[]
for seed in [17,29,43]:
    sd=P/f'seed-{seed}';m=np.load(sd/'model.npz');w1=m['w1'].astype(object);b1=m['b1'].astype(object);w2=m['w2'].astype(object);b2=m['b2'].astype(object)
    hs={}
    for split,d in splits.items():
        h=np.maximum(d['x'].astype(object)@w1+16*b1,0)
        hs[split]=h
    for bn in ['seen_strength','unseen_strength','natural_pairs']:
        b=np.load(sd/f'test_bank_{bn}.npz');meta=b['meta'];lookup={int(i):j for j,i in enumerate(splits['test']['ids'])}
        h4=[];p4=[]
        for row in meta:
            if bn=='natural_pairs':
                i,j=row;h4.append(4*hs['test'][lookup[int(i)]]);p4.append(4*hs['test'][lookup[int(j)]])
            else:
                i,g,strength=row;h=hs['test'][lookup[int(i)]];p=4*h.copy();p[groups[int(g)]]=(4-int(strength))*h[groups[int(g)]];h4.append(4*h);p4.append(p)
        h4=np.asarray(h4,dtype=object);p4=np.asarray(p4,dtype=object)
        z=h4@w2+4*65536*b2;zp=p4@w2+4*65536*b2
        centered=10*z-z.sum(axis=1,keepdims=True);centeredp=10*zp-zp.sum(axis=1,keepdims=True)
        exact=centeredp-centered
        check(np.array_equal(exact,b['response_numerator']),f'{seed}/{bn}: every target via arbitrary-precision endpoint logits')
        check(int(b['response_denominator'])==4*65536*4096*10,f'{seed}/{bn}: denominator')
        o=np.load(sd/'report-native-8.npz');qw=o['w'].astype(object);t=o['threshold'].astype(object)
        q=(h4@qw.T>4*t).astype(np.int8);qp=(p4@qw.T>4*t).astype(np.int8)
        check(np.array_equal(q,b['report_before']) and np.array_equal(qp,b['report_after']),f'{seed}/{bn}: every report predicate')
        check(np.array_equal(qp-q,b['A']),f'{seed}/{bn}: signed design')
        cand=np.load(sd/'discovery_bank.npz');cw=cand['candidate_w'].astype(object);ct=cand['candidate_threshold'].astype(object)
        ca=(p4@cw.T>4*ct).astype(np.int8)-(h4@cw.T>4*ct).astype(np.int8)
        check(np.array_equal(ca,b['candidate_A']),f'{seed}/{bn}: all supplementary candidate columns')
        if bn!='natural_pairs':
            expected={2,4} if bn=='seen_strength' else {1,3}
            check(set(meta[:,2])==expected,f'{seed}/{bn}: declared strengths')
        result=next(r for r in test_results['results'] if r['seed']==seed)['banks'][bn]['results']['native-8']['report_pair']
        witness=result['max_range_witness'];lo,hi=witness['rows'];coord=witness['coordinate']
        check(np.array_equal(np.concatenate([q[lo],qp[lo]]),np.concatenate([q[hi],qp[hi]])),f'{seed}/{bn}: witness report-pair equality')
        gap=abs(int(exact[lo,coord])-int(exact[hi,coord]));check(gap==result['exact_worst_range_numerator'],f'{seed}/{bn}: witness exact gap')
        check(gap>2*int(b['response_denominator']),f'{seed}/{bn}: rejects primary tolerance')
        details.append({'seed':seed,'bank':bn,'rows':len(meta),'primary_native8_witness_gap':str(Fraction(gap,int(b['response_denominator']))),'max_integer_numerator':max(abs(int(v)) for v in exact.flat)})
    for path in sorted(sd.glob('precision-*.npz')):
        o=np.load(path);M=o['M'];C=o['C'];bits=o['bits'];eps=Fraction(str(float(o['epsilon'])))
        box=np.maximum(0,16*(np.maximum(w1,0).sum(axis=0)+b1))
        check(np.array_equal(M,box),f'{seed}/{path.stem}: independent whole-input-box bound')
        check(np.array_equal(C,10*w2-w2.sum(axis=1,keepdims=True)),f'{seed}/{path.stem}: centered readout')
        for k in range(10):
            bound=sum((Fraction(int(M[j])*abs(int(C[j,k])),65536*4096*10*(1<<int(bits[j]))) for j in range(32)),Fraction())
            check(bound<=eps,f'{seed}/{path.stem}: exact output-{k} paired bound')
        stored=next(r for r in precision_results['results'] if r['seed']==seed)
        for bn in ['seen_strength','unseen_strength','natural_pairs']:
            rr=stored['banks'][bn][path.stem.removeprefix('precision-')]
            check(Fraction(rr['exact_max_error_numerator'],rr['exact_error_denominator'])<=eps,f'{seed}/{path.stem}/{bn}: saved exact finite error inequality')

if (P/'context_sensitivity.json').exists():
    contexts=json.loads((P/'context_sensitivity.json').read_text())
    for r in contexts['records']:
        b=np.load(P/f"seed-{r['seed']}"/f"test_bank_{r['bank']}.npz")
        lo,hi=r['max_range_witness']['rows'];k=r['max_range_witness']['coordinate']
        check(np.array_equal(b['meta'][lo,1:],b['meta'][hi,1:]),f"context {r['seed']}/{r['bank']}: same intervention")
        check(np.array_equal(b['report_before'][lo],b['report_before'][hi]) and np.array_equal(b['report_after'][lo],b['report_after'][hi]),f"context {r['seed']}/{r['bank']}: same report pair")
        gap=abs(int(b['response_numerator'][lo,k])-int(b['response_numerator'][hi,k]))
        check(gap==r['exact_worst_range_numerator'],f"context {r['seed']}/{r['bank']}: exact witness gap")
        check(gap>2*int(b['response_denominator']),f"context {r['seed']}/{r['bank']}: rejects epsilon1")

result={'status':'passed','checks':len(checked),'boundary':'independent arithmetic implementation and receipt verification by the same research team; not an independent scientific replication','details':details,'input_hashes':{f:sha(P/f) for f in ['experiment.py','precision_repair.py','contract.json','freeze.json','precision_freeze.json','test_results.json','precision_test_results.json']},'verification_script_sha256':sha(__file__)}
(P/'verification.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':'passed','checks':len(checked),'integer_rows_replayed':sum(d['rows'] for d in details)}))
