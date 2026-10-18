#!/usr/bin/env python3
"""Independent exact replay plus reporting of the exploratory nonlinear study."""
from pathlib import Path
from fractions import Fraction
import csv,hashlib,json
import numpy as np
P=Path(__file__).resolve().parent
S=4096;H=65536;D=4*H*S*S*10
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def check(v,s):
    if not v:raise AssertionError(s)
    checks.append(s)
fr=json.loads((P/'freeze.json').read_text())
for f,d in fr['files'].items():check(sha(P/f)==d,'frozen '+f)
check(sha(P.parent/'application'/'experiment.py')==fr['sibling_experiment_sha256'],'frozen dependency')
c=json.loads((P/'contract.json').read_text());rr=json.loads((P/'test_results.json').read_text());dr=json.loads((P/'discovery_summary.json').read_text());te=np.load(P/'test.npz');idlookup={int(v):j for j,v in enumerate(te['ids'])};details=[]
for r in rr['results']:
    seed=r['seed'];sd=P/f'seed-{seed}';m=np.load(sd/'model.npz');w1=m['w1'].astype(object);b1=m['b1'].astype(object);w2=m['w2'].astype(object);b2=m['b2'].astype(object);w3=m['w3'].astype(object);b3=m['b3'].astype(object)
    h=np.maximum(te['x'].astype(object)@w1+16*b1,0)
    for bn in ['seen_strength','unseen_strength']:
        b=np.load(sd/f'test_bank_{bn}.npz');hs=[];ps=[]
        for i,g,st in b['meta']:
            x=h[idlookup[int(i)]];xp=4*x.copy();group=c['groups'][int(g)];xp[group]=(4-int(st))*x[group];hs.append(4*x);ps.append(xp)
        hs=np.asarray(hs,dtype=object);ps=np.asarray(ps,dtype=object)
        pre=hs@w2+4*H*b2;prep=ps@w2+4*H*b2
        logits=np.maximum(pre,0)@w3+4*H*S*b3;logitsp=np.maximum(prep,0)@w3+4*H*S*b3
        y=(10*logitsp-logitsp.sum(axis=1,keepdims=True))-(10*logits-logits.sum(axis=1,keepdims=True))
        check(np.array_equal(y,b['response_numerator']),f'{seed}/{bn} all exact target numerators')
        check(int(b['response_denominator'])==D,f'{seed}/{bn} denominator')
        o=np.load(sd/'report-native-8.npz');q=(hs@o['w'].T>4*o['threshold']).astype(np.int8);qp=(ps@o['w'].T>4*o['threshold']).astype(np.int8)
        check(np.array_equal(q,b['report_before']) and np.array_equal(qp,b['report_after']),f'{seed}/{bn} report predicates')
        check(np.array_equal(qp-q,b['A']),f'{seed}/{bn} additive design')
        co=np.load(sd/'discovery_bank.npz');ca=(ps@co['candidate_w'].T>4*co['candidate_threshold']).astype(np.int8)-(hs@co['candidate_w'].T>4*co['candidate_threshold']).astype(np.int8)
        check(np.array_equal(ca,b['candidate_A']),f'{seed}/{bn} supplementary candidate design')
        crossings=((pre>0)!=(prep>0)).sum(axis=1)
        detail={'seed':seed,'bank':bn,'rows':len(y),'fraction_rows_crossing_downstream_relu_boundary':float(np.mean(crossings>0)),'mean_changed_downstream_gate_count':float(np.mean(crossings)),'max_changed_downstream_gate_count':int(crossings.max())}
        details.append(detail)
        pp=np.load(sd/'precision.npz');M=pp['M'].astype(object);bits=pp['bits'];levels=np.asarray([1<<int(j) for j in bits],dtype=object);Q=max(int(v) for v in levels);safe=np.where(M>0,4*M,1)
        pq=np.minimum((hs*levels)//safe,levels-1);pqp=np.minimum((ps*levels)//safe,levels-1);pq[:,M==0]=0;pqp[:,M==0]=0
        mh=M*(2*pq+1)*(Q//levels);mhp=M*(2*pqp+1)*(Q//levels)
        pp2=np.maximum(mh@w2+2*Q*H*b2,0);pp2p=np.maximum(mhp@w2+2*Q*H*b2,0)
        zz=pp2@w3+2*Q*H*S*b3;zzp=pp2p@w3+2*Q*H*S*b3
        dy=(10*zzp-zzp.sum(axis=1,keepdims=True))-(10*zz-zz.sum(axis=1,keepdims=True));err=dy-y*(Q//2);maximum=max(abs(int(v)) for v in err.flat);pden=2*Q*H*S*S*10
        check(maximum<=pden,f'{seed}/{bn} exact precision epsilon1 pass')
        check(maximum==r['banks'][bn]['results']['precision']['exact_max_error_numerator'],f'{seed}/{bn} exact precision maximum replay')
    M=np.maximum(0,16*(np.maximum(w1,0).sum(axis=0)+b1));C=np.abs(w2)@np.abs(10*w3-w3.sum(axis=1,keepdims=True));pp=np.load(sd/'precision.npz')
    check(np.array_equal(M,pp['M']) and np.array_equal(C,pp['C']),f'{seed} absolute-weight whole-box constants')
    for k in range(10):
        bound=sum((Fraction(int(M[j])*int(C[j,k]),H*S*S*10*(1<<int(pp['bits'][j]))) for j in range(32)),Fraction())
        check(bound<=1,f'{seed} exact global precision bound output{k}')
result={'status':'passed','checks':len(checks),'rows_replayed':sum(d['rows'] for d in details),'arithmetic':'Python arbitrary-precision integer endpoint forward passes; exact rational global bounds','scope':'verification by same research team, not independent scientific replication','details':details,'script_sha256':sha(__file__)}
(P/'verification.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')

rows=[]
for r in rr['results']:
    for bn,b in r['banks'].items():
        for method,v in b['results'].items():
            rows.append({'seed':r['seed'],'bank':bn,'method':method,'mse_sum_outputs':v['prediction']['mse_sum_outputs'],'max_absolute_error':v['prediction']['max_abs'],'fraction_within_epsilon1':v['prediction']['within_epsilon']['1.0'],'unrestricted_report_pair_empirical_floor':v.get('report_pair',{}).get('oracle_empirical_mse'),'unrestricted_report_pair_exact_minimax':v.get('report_pair',{}).get('exact_minimax_report_error')})
with (P/'all_metrics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

lines=['# Exploratory nonlinear-downstream architecture extension','',
'This supplement tests the first hidden-state cut of a 64→32 ReLU→16 ReLU→10 classifier. It was designed after the shallow-model study and reuses the same already-inspected image split. Its choices were fixed before evaluating these new models on that split, but it is not a pristine external replication or a population-level architecture comparison. The principal application and its original freezes remain unchanged.','',
'## Design','',
'Three new predetermined initialisation seeds, 117, 129 and 143, use the same 200-epoch NumPy Adam recipe and 1,074/238/121/364 train/discovery-fit/discovery-validation/test split. The first hidden layer has 32 units and is the nominated cut. Four fixed eight-unit groups are attenuated at strengths 1/2 and 1 during discovery; test uses those strengths on new images and the unseen strengths 1/4 and 3/4. No label, input identity, intervention identity or full hidden state is supplied to the report decoder.','',
'Candidate reports are native median predicates (8 or 12 state bits), PCA median predicates (8 bits), and a capacity-matched random-direction baseline (8 bits). Each family has 32 candidate linear projections. Random directions are fixed Gaussian unit vectors seeded by model seed plus 5671; PCA is learned from training states. All projection coefficients are quantised at scale 4096 and all thresholds are training order-statistic medians. The same greedy discovery-fit response-MSE procedure selects features, and the same separate discovery-validation ridge choice fits a shared additive response map B[T(h′)−T(h)]. No extension choices are tuned on its test outcomes.','',
'The random direction control has the same dense encoder coefficient budget as PCA. Native8 uses 8 nonzero encoder weights and 80 response coefficients; native12 uses 12 and 120. Dense8 uses at most 256 nonzero encoder weights and 80 response coefficients. Transmission costs are twice the state-bit counts because both endpoint reports are supplied. This does not equalise semantic legibility, which is not measured.','',
'The model weights are integer multiples of 1/4096. Exact endpoint evaluation passes the intervened hidden numerators through the second ReLU using Python arbitrary-precision integers. Centered-logit response differences have denominator 43980465111040. Only after checking the range are target numerators stored as int64. The freeze SHA-256 is '+rr['freeze_sha256']+'. New-model test evaluation began at '+rr['test_started_utc']+'.','',
'## Held-out results','',
'| Seed | Test accuracy | Export/float class agreement | Largest centered-logit export error |','|---|---:|---:|---:|']
for r in rr['results']:lines.append(f"|{r['seed']}|{100*r['test_accuracy']:.2f}%|{100*r['export_float_class_agreement']:.2f}%|{r['export_float_max_logit_error']:.6f}|")
lines+=['','All models reached 100% training accuracy. MSE below sums squared error across all ten centered logits and then averages intervention rows. Each bank has 2,912 rows per model; rows sharing an input are dependent.','',
'| Seed | Native8 unseen MSE | Native12 unseen MSE | PCA8 unseen MSE | Random8 unseen MSE |','|---|---:|---:|---:|---:|']
for r in rr['results']:
    b=r['banks']['unseen_strength']['results'];lines.append(f"|{r['seed']}|"+'|'.join(f"{b[n]['prediction']['mse_sum_outputs']:.3f}" for n in ['native-8','native-12','pca-8','random-8'])+'|')
lines+=['','The tested additive decoder is more accurate with native predicates than either dense eight-bit baseline in these three models, and PCA is worse than the matched random directions here. This cannot establish general native-feature superiority: it is conditional on the selected one-bit reports, discovery objective, restricted shared decoder and small digit models. All low-bit methods fail the primary maximum-error tolerance of one logit unit.','',
'The exact report-pair conditional-variance floor remains below actual decoder MSE, while report-pair collisions independently falsify information sufficiency at the stated tolerance. Exact affine-circuit certificates on injective report banks, where available, are reported in the separate dual-certificate supplement. The fitted-risk gap alone is not taken as a proof of affine approximation impossibility.','',
'## The nonlinear layer is active in these interventions','',
'A descriptive post-test diagnostic counts second-layer ReLU gates whose sign changes under an intervention. It was not used for selection.','',
'| Seed | Seen strengths: any gate changes | Unseen strengths: any gate changes | Mean changed gates, unseen |','|---|---:|---:|---:|']
for seed in [117,129,143]:
    a=next(d for d in details if d['seed']==seed and d['bank']=='seen_strength');b=next(d for d in details if d['seed']==seed and d['bank']=='unseen_strength')
    lines.append(f"|{seed}|{100*a['fraction_rows_crossing_downstream_relu_boundary']:.2f}%|{100*b['fraction_rows_crossing_downstream_relu_boundary']:.2f}%|{b['mean_changed_downstream_gate_count']:.3f}|")
lines+=['','Thus this is not merely a second layer that stays in one global linear regime. The extension remains a small feedforward model; it does not test attention, recurrence, natural-language semantics or foundation-model prevalence.','',
'## Conservative nonlinear precision certificate','',
'For each first-hidden coordinate, the same exact input-box bound M_j applies. Let C_mk = 10W3_int[m,k]−Σ_lW3_int[m,l]. Since ReLU is coordinatewise 1-Lipschitz, midpoint quantisation with b_j bits gives a paired response bound', '',
'Σ_j M_j Σ_m |W2_int[j,m]| |C_mk| /(65536·4096²·10·2^{b_j})','',
'in output k. This is a conservative absolute-weight product bound, not a sharp attainable error or an optimal bit allocation. The greedy allocator and its tie rules are identical in spirit to the primary precision supplement but operate on these fixed larger coefficients. Selection uses model weights only. The decoder evaluates the original nonlinear downstream network on midpoint states, charging its 698 stored weights and biases. This is a numerical compression baseline rather than a simpler semantic algorithm.','',
'| Seed | Uniform state bits | Gain-aware state bits | Certified maximum error | Observed unseen maximum error | Lossless integer box bits |','|---|---:|---:|---:|---:|---:|']
for r,d in zip(rr['results'],dr):
    p=d['precision'];obs=r['banks']['unseen_strength']['results']['precision']['prediction']['max_abs'];lines.append(f"|{r['seed']}|{p['uniform_state_bits']}|{p['state_bits']}|{p['max_bound']:.6f}|{obs:.6f}|{p['lossless_h_integer_box_bits']}|")
lines+=['','Every finite tested precision prediction satisfies epsilon1 by exact integer comparison. The analytic bound covers the whole declared input box and arbitrary coordinate attenuation, including downstream ReLU boundary crossings. Its conservatism is visible in the much smaller observed errors. The lossless code comparison is a coordinatewise fixed-length integer-box benchmark; it is not an optimum joint code or a finite lossless code for a continuous domain.','',
'## Reproduction boundary','',
f'Independent arithmetic replay by the same team passed {len(checks)} checks and rebuilt all {sum(d["rows"] for d in details):,} target rows and precision predictions using Python integers. The retained code is nonlinear.py; verification/reporting is verify_and_report.py. The sibling application/experiment.py supplies frozen common numerical utilities, with its hash in this supplement’s freeze. No external download or scipy is required for training/evaluation. Keep the two application directories together when reproducing. All model/report matrices, exact targets, original and additional precision parameters, contracts and results are retained.','',
'Dataset attribution and licence are the same as the primary application: Alpaydin and Kaynak (1998), UCI Optical Recognition of Handwritten Digits, DOI https://doi.org/10.24432/C50P49, CC BY 4.0; this uses the preserved 1,797-image scikit-learn subset.']
(P/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'status':'passed','checks':len(checks),'rows_replayed':result['rows_replayed'],'regime_crossings':details}))
