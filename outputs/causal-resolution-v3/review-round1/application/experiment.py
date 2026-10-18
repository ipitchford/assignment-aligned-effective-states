#!/usr/bin/env python3
"""Frozen finite-report audit of three trained digit classifiers.

NumPy and the Python standard library suffice.  No data are downloaded.
Phases deliberately separate preparation, discovery, sealing, and evaluation.
"""
from __future__ import annotations
import argparse, csv, gzip, hashlib, json, os, platform, shutil, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SCALE = 4096
H_DEN = 16 * SCALE
RESPONSE_DEN = 4 * H_DEN * SCALE * 10
SEEDS = [17, 29, 43]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")

def write_once(path, obj):
    if Path(path).exists():
        raise FileExistsError(f"Refusing to overwrite sealed-stage output: {path}")
    dump(path, obj)

def prepare():
    if (HERE / "contract.json").exists():
        raise FileExistsError("Existing experiment contract; do not re-split.")
    source = Path('/Users/admin/.venvs/gaia-tess-microlensing/lib/python3.11/site-packages/sklearn/datasets/data/digits.csv.gz')
    shutil.copyfile(source, HERE / 'digits.csv.gz')
    with gzip.open(HERE / 'digits.csv.gz', 'rt') as f:
        data = np.loadtxt(f, delimiter=',')
    x, y = data[:, :64].astype(np.int64), data[:, 64].astype(np.int64)
    rng = np.random.default_rng(20261015)
    ids = {k: [] for k in ('train', 'discovery_fit', 'discovery_validation', 'test')}
    for c in range(10):
        perm = rng.permutation(np.flatnonzero(y == c))
        a, b = int(.60 * len(perm)), int(.80 * len(perm))
        disc = perm[a:b]
        cut = int(2 * len(disc) / 3)
        ids['train'].extend(perm[:a])
        ids['discovery_fit'].extend(disc[:cut])
        ids['discovery_validation'].extend(disc[cut:])
        ids['test'].extend(perm[b:])
    for split, ix in ids.items():
        ix = rng.permutation(np.asarray(ix, dtype=np.int64))
        np.savez_compressed(HERE / f'{split}.npz', x=x[ix], y=y[ix], ids=ix)
    groups = np.random.default_rng(874).permutation(32).reshape(4, 8)
    contract = {
      'version': 1, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
      'status': 'frozen before training, report selection, or test evaluation; local prospective split, not external preregistration',
      'dataset': {'name': 'sklearn bundled UCI optical handwritten digits', 'n': len(y), 'source_file': str(source), 'sha256': sha(HERE / 'digits.csv.gz'), 'citation_url': 'https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits', 'normalisation': 'integer pixel / 16', 'writer_identity': 'not supplied; random stratified image split does not establish writer independence'},
      'split': {'seed': 20261015, 'fractions': [.60,.20,.20], 'discovery_fit_fraction': 2/3, 'counts': {s:len(v) for s,v in ids.items()}, 'independent_units': 'three model initialisations and held-out images; interventions of an image are dependent', 'files': {s: sha(HERE / f'{s}.npz') for s in ids}},
      'model': {'seeds': SEEDS, 'architecture': [64,32,10], 'activation': 'ReLU', 'epochs': 200, 'batch_size': 128, 'adam_lr_first_150': .003, 'adam_lr_last_50': .001, 'weight_decay': .0001, 'early_stopping': False, 'export': 'weights and biases rounded to integer / 4096; exact integer ReLU and centered-logit target'},
      'interventions': {'groups': groups.tolist(), 'discovery_strengths': [.5,1.0], 'test_seen_strengths': [.5,1.0], 'test_unseen_strengths': [.25,.75], 'definition': 'hprime_j=(1-strength)*h_j in one fixed 8-unit group; others unchanged', 'natural_bank': 'cyclic pairs of distinct held-out inputs, replacing the whole nominated hidden state; separately reported', 'off_manifold': 'group-damped hidden states are controlled mathematical interventions, not asserted to be natural images'},
      'report': {'state_bit_budgets': [4,8,12,16], 'primary_bits': 8, 'native_candidates': '32 individual units thresholded at train median', 'native_groups': '16 disjoint pairs, greedily strongest positive Pearson coactivation on training states; sum thresholded at train median', 'pca_candidates': '32 training covariance eigenvectors, integer coefficients rounded at scale4096, training-median predicates', 'candidate_ranking': 'greedy discovery-fit response mean-square error with selection ridge=0.01; same procedure for all families', 'quantiles': 'sorted integer linear-projection element at floor(q*(n-1)); predicate uses strict greater-than', 'context': 'none; no input identity, label, intervention id or hidden state supplied to decoder'},
      'decoder': {'formula': 'B times (T(hprime)-T(h)); one shared B across groups, strengths and natural pairs', 'class': 'linear in signed predicate differences; no intercept or interactions for response', 'parameter_count': '10 times number of state predicates', 'ridge_grid': [1e-6,.0001,.01,.1,1.0], 'ridge_selection': 'lowest discovery-validation response MSE; first grid value breaks ties', 'final_fit': 'discovery-fit only (validation never refitted)', 'baseline_class_readout': 'same B applied to T(h), with discovery-fit natural-state mean residual as intercept; diagnostic only'},
      'repair': {'base': 'native 8-bit report', 'additional_bits': 4, 'candidate_pool': 'native predicates at training quartiles 0.25,0.5,0.75, with duplicate integer predicates removed', 'selection': 'four greedy additions minimizing discovery-fit response MSE, selection ridge=0.01; no validation or test targets in feature choice', 'naive_cost_matched': '12-bit native median report from original greedy ranking', 'claim': 'measured discovery-selected refinement; success not assumed'},
      'metrics': {'primary_target': '10 centered logit changes (sum over outputs for squared error)', 'primary_tolerance': 1.0, 'sensitivity_tolerances': [.25,.5,1.0,2.0,4.0], 'sensitivity_uniform_response_allowance': [0,.125,.25,.5,1], 'precision': f'exact signed integer response numerator / {RESPONSE_DEN}', 'information_floor': 'empirical within-identical-report-pair conditional variance; in-sample finite-bank oracle only', 'decoder_summary_floor': 'same variance after signed-difference aggregation, separately named', 'collision_certificate': 'within-report-pair coordinate range exceeds twice tolerance, exact integer comparisons; complete bank', 'coverage': 'finite-bank statements, no population or semantic-completeness certification'},
      'costs': {'transmission': 'B bits per state; 2B bits for the pair', 'encoder': 'number of nonzero fixed integer weights plus B thresholds, coefficient magnitude/precision retained', 'decoder': '10B binary64 coefficients; one fixed 10-vector baseline-only intercept', 'learned_feature_warning': 'PCA is a learned linear direction baseline, not an SAE and not a semantic explanation'},
      'controls': ['training/export prediction agreement', 'full-state downstream identity gives exact response', 'same report-pair collision and signed-change decoder collision separated', 'holdout model/report hashes sealed before evaluating test', 'test cases and intervention magnitudes never used for feature or ridge selection', 'three predetermined model seeds; all seed results retained'],
      'environment': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform()},
    }
    write_once(HERE / 'contract.json', contract)
    write_once(HERE / 'test_lock.json', {'created_utc': contract['created_utc'], 'test_sha256': sha(HERE / 'test.npz'), 'contract_sha256': sha(HERE / 'contract.json'), 'split_ids_sha256': hashlib.sha256(np.asarray(sorted(ids['test']),dtype='<i8').tobytes()).hexdigest(), 'opened_for_outcomes': False})
    print(json.dumps({'phase':'prepared','counts':contract['split']['counts'],'contract_sha256':sha(HERE/'contract.json')}), flush=True)

def softmax(z):
    e = np.exp(z-z.max(axis=1,keepdims=True)); return e/e.sum(axis=1,keepdims=True)

def train_model(xi, y, seed):
    rng=np.random.default_rng(seed); x=xi.astype(float)/16
    pars=[rng.normal(0,np.sqrt(2/64),(64,32)),np.zeros(32),rng.normal(0,np.sqrt(2/32),(32,10)),np.zeros(10)]
    m=[np.zeros_like(v) for v in pars]; v=[np.zeros_like(v) for v in pars]; t=0
    for epoch in range(200):
        lr=.003 if epoch<150 else .001
        order=rng.permutation(len(x))
        for start in range(0,len(x),128):
            ix=order[start:start+128]; xb=x[ix]; yb=y[ix]
            h=np.maximum(xb@pars[0]+pars[1],0); logits=h@pars[2]+pars[3]
            d=softmax(logits); d[np.arange(len(ix)),yb]-=1; d/=len(ix)
            dh=(d@pars[2].T)*(h>0)
            grads=[xb.T@dh+.0001*pars[0],dh.sum(axis=0),h.T@d+.0001*pars[2],d.sum(axis=0)]
            t+=1
            for j,g in enumerate(grads):
                m[j]=.9*m[j]+.1*g; v[j]=.999*v[j]+.001*g*g
                pars[j]-=lr*(m[j]/(1-.9**t))/(np.sqrt(v[j]/(1-.999**t))+1e-8)
    return pars

def hidden(x, model):
    return np.maximum(x@model['w1']+16*model['b1'],0).astype(np.int64)

def logits_num(h, model):
    z=h@model['w2']+H_DEN*model['b2']; return 10*z-z.sum(axis=1,keepdims=True)

def target_num(h4,h4p,model):
    d=(h4p-h4)@model['w2']; return 10*d-d.sum(axis=1,keepdims=True)

def bank(h, ids, strengths, groups):
    hs=[]; ps=[]; meta=[]
    for strength in strengths:
        remaining=int(round(4*(1-strength)))
        for g,units in enumerate(groups):
            hp=4*h.copy(); hp[:,units]=remaining*h[:,units]
            hs.append(4*h); ps.append(hp)
            meta.extend([[int(i),g,int(round(4*strength))] for i in ids])
    return np.concatenate(hs),np.concatenate(ps),np.asarray(meta,dtype=np.int64)

def natural_bank(h,ids):
    return 4*h,4*np.roll(h,-1,axis=0),np.stack([ids,np.roll(ids,-1)],axis=1)

def quantile_int(v,q):
    return np.sort(v)[int(np.floor(q*(len(v)-1)))]

def encoder_candidates(h):
    native=np.eye(32,dtype=np.int64)
    hc=h.astype(float)-h.mean(axis=0)
    norms=np.sqrt((hc*hc).sum(axis=0)); cov=hc.T@hc
    denom=np.outer(norms,norms); corr=np.divide(cov,denom,out=np.zeros_like(cov),where=denom>0)
    pairs=[]; remaining=set(range(32))
    while remaining:
        opts=[(float(corr[a,b]),-a,-b,a,b) for a in sorted(remaining) for b in sorted(remaining) if a<b]
        _,_,_,a,b=max(opts); pairs.append([a,b]); remaining.remove(a);remaining.remove(b)
    group=np.zeros((16,32),dtype=np.int64)
    for k,(a,b) in enumerate(pairs): group[k,a]=group[k,b]=1
    vals,vec=np.linalg.eigh(cov/len(h)); vec=vec[:,np.argsort(vals)[::-1]].T
    for k in range(len(vec)):
        if vec[k,np.argmax(np.abs(vec[k]))]<0: vec[k]*=-1
    pca=np.rint(vec*SCALE).astype(np.int64)
    result={}
    for name,w in [('native',native),('group',group),('pca',pca)]:
        proj=h@w.T
        th=np.array([quantile_int(proj[:,j],.5) for j in range(len(w))],dtype=np.int64)
        result[name]={'w':w,'threshold':th,'source':np.arange(len(w),dtype=np.int64)}
    rw=[];rt=[];ri=[];rq=[]
    for j in range(32):
        seen=set()
        for q in [.25,.5,.75]:
            t=int(quantile_int(h[:,j],q))
            if t in seen: continue
            seen.add(t);rw.append(native[j]);rt.append(t);ri.append(j);rq.append(q)
    result['repair_pool']={'w':np.asarray(rw),'threshold':np.asarray(rt,dtype=np.int64),'source':np.asarray(ri),'quantile':np.asarray(rq)}
    return result,pairs

def encode(h4,w,threshold):
    return (h4@w.T>4*threshold).astype(np.int8)

def design(h4,h4p,w,threshold):
    q=encode(h4,w,threshold); qp=encode(h4p,w,threshold)
    return qp-q,q,qp

def ridge(a,y,lam):
    return np.linalg.solve(a.T@a/len(a)+lam*np.eye(a.shape[1]),a.T@y/len(a))

def objective_from_gram(gram,cross,y2,indices,lam=.01):
    gg=gram[np.ix_(indices,indices)];xy=cross[indices]
    beta=np.linalg.solve(gg+lam*np.eye(len(indices)),xy)
    return float(y2-2*np.sum(beta*xy)+np.sum(beta*(gg@beta)))

def greedy_order(a,y,n,initial=None):
    af=a.astype(float); gram=af.T@af/len(af);cross=af.T@y/len(af);y2=np.mean(np.sum(y*y,axis=1))
    selected=[] if initial is None else list(initial); trace=[]
    while len(selected)<n:
        candidates=[(objective_from_gram(gram,cross,y2,selected+[j]),j) for j in range(a.shape[1]) if j not in selected]
        loss,j=min(candidates); selected.append(j);trace.append({'column':j,'fit_mse':loss})
    return selected,trace

def choose_decoder(af,yf,av,yv,grid):
    records=[]
    for lam in grid:
        beta=ridge(af.astype(float),yf,lam)
        loss=float(np.mean(np.sum((av@beta-yv)**2,axis=1)))
        records.append((loss,lam,beta))
    best=min(records,key=lambda r:(r[0],r[1]))
    return best[2],best[1],[{'lambda':r[1],'validation_mse':r[0]} for r in records]

def score(pred,ynum,den=RESPONSE_DEN):
    y=ynum.astype(float)/den; err=pred-y; max_each=np.abs(err).max(axis=1)
    return {'n':len(y),'mse_sum_outputs':float(np.mean(np.sum(err*err,axis=1))), 'rmse_per_output':float(np.sqrt(np.mean(err*err))), 'max_abs':float(np.max(max_each)), 'median_max_abs':float(np.median(max_each)), 'p95_max_abs':float(np.quantile(max_each,.95)), 'within_epsilon':{str(e):float(np.mean(max_each<=e)) for e in [.25,.5,1.,2.,4.]}, 'mean_target_squared_norm':float(np.mean(np.sum(y*y,axis=1)))}

def collision_summary(codes,ynum):
    _,inverse,counts=np.unique(codes,axis=0,return_inverse=True,return_counts=True)
    ng=len(counts);mins=np.full((ng,ynum.shape[1]),np.iinfo(np.int64).max,dtype=np.int64);maxs=np.full_like(mins,np.iinfo(np.int64).min)
    np.minimum.at(mins,inverse,ynum);np.maximum.at(maxs,inverse,ynum)
    ranges=(maxs-mins).max(axis=1);sums=np.zeros((ng,ynum.shape[1]),float);np.add.at(sums,inverse,ynum/RESPONSE_DEN)
    mean=sums/counts[:,None];res=ynum/RESPONSE_DEN-mean[inverse]
    witnesses=[]
    for e in [.25,.5,1.,2.,4.]:
        bad=ranges>int(2*e*RESPONSE_DEN)
        witnesses.append({'epsilon':e,'incompatible_fibres':int(bad.sum()),'images_of_rows_in_incompatible_fibres':int(counts[bad].sum()),'fraction_rows_in_incompatible_fibres':float(counts[bad].sum()/len(ynum))})
    gi=int(np.argmax(ranges));rows=np.flatnonzero(inverse==gi)
    coord=int(np.argmax(maxs[gi]-mins[gi]));lo=int(rows[np.argmin(ynum[rows,coord])]);hi=int(rows[np.argmax(ynum[rows,coord])])
    return {'unique_codes':ng,'collision_fibres':int((counts>1).sum()),'fraction_rows_sharing_a_code':float(counts[counts>1].sum()/len(ynum)), 'oracle_empirical_mse':float(np.mean(np.sum(res*res,axis=1))), 'exact_minimax_report_error':float(ranges.max()/(2*RESPONSE_DEN)), 'exact_worst_range_numerator':int(ranges.max()),'denominator':RESPONSE_DEN, 'tolerances':witnesses,'max_range_witness':{'rows':[lo,hi],'coordinate':coord,'response_numerators':[int(ynum[lo,coord]),int(ynum[hi,coord])],'code':codes[lo].tolist()}, 'allowance_brackets_epsilon1':[{'delta':d,'definite_bad_fibres':int((ranges>int(2*(1+d)*RESPONSE_DEN)).sum()),'possible_bad_fibres':int((ranges>int(2*(1-d)*RESPONSE_DEN)).sum())} for d in [0,.125,.25,.5,1.]]}

def fit():
    if (HERE/'freeze.json').exists(): raise FileExistsError('Frozen reports already exist.')
    c=json.loads((HERE/'contract.json').read_text());tr=np.load(HERE/'train.npz');df=np.load(HERE/'discovery_fit.npz');dv=np.load(HERE/'discovery_validation.npz')
    groups=np.asarray(c['interventions']['groups']);records=[]
    for seed in SEEDS:
        sd=HERE/f'seed-{seed}';sd.mkdir(exist_ok=False)
        pars=train_model(tr['x'],tr['y'],seed)
        names=['w1','b1','w2','b2'];model={n:np.rint(p*SCALE).astype(np.int64) for n,p in zip(names,pars)}
        np.savez_compressed(sd/'model.npz',**model,**{f'float_{n}':p for n,p in zip(names,pars)})
        ht=hidden(tr['x'],model);hf=hidden(df['x'],model);hv=hidden(dv['x'],model)
        h4,h4p,mf=bank(hf,df['ids'],[.5,1.],groups);v4,v4p,mv=bank(hv,dv['ids'],[.5,1.],groups)
        yn=target_num(h4,h4p,model);yvn=target_num(v4,v4p,model);yf=yn/RESPONSE_DEN;yv=yvn/RESPONSE_DEN
        pool,pairs=encoder_candidates(ht);np.savez_compressed(sd/'candidates.npz',**{f'{fam}_{key}':v for fam,o in pool.items() for key,v in o.items()})
        configurations={};orders={};selection={};all_design={}
        for fam in ['native','group','pca']:
            o=pool[fam];a,_,_=design(h4,h4p,o['w'],o['threshold']);av,_,_=design(v4,v4p,o['w'],o['threshold']);all_design[fam]=a
            order,trace=greedy_order(a,yf,16);orders[fam]=order;selection[fam]=trace
            for bits in [4,8,12,16]:
                ix=order[:bits];beta,lam,grid=choose_decoder(a[:,ix],yf,av[:,ix],yv,c['decoder']['ridge_grid'])
                configurations[f'{fam}-{bits}']={'w':o['w'][ix],'threshold':o['threshold'][ix],'beta':beta,'source':o['source'][ix],'lambda':lam,'grid':grid,'family':fam,'selection_columns':ix}
        # Append a pool of alternative native thresholds to the fixed 8-bit report.
        base=configurations['native-8'];rp=pool['repair_pool'];combined_w=list(base['w']);combined_t=list(base['threshold']);combined_sources=list(base['source']);combined_q=[.5]*8
        for w,t,s,q in zip(rp['w'],rp['threshold'],rp['source'],rp['quantile']):
            if any(np.array_equal(w,w0) and int(t)==int(t0) for w0,t0 in zip(combined_w,combined_t)): continue
            combined_w.append(w);combined_t.append(t);combined_sources.append(s);combined_q.append(q)
        rw=np.asarray(combined_w);rt=np.asarray(combined_t);ra,_,_=design(h4,h4p,rw,rt);rav,_,_=design(v4,v4p,rw,rt)
        ri,rtrace=greedy_order(ra,yf,12,initial=range(8));beta,lam,grid=choose_decoder(ra[:,ri],yf,rav[:,ri],yv,c['decoder']['ridge_grid'])
        configurations['repair-12']={'w':rw[ri],'threshold':rt[ri],'beta':beta,'source':np.asarray(combined_sources)[ri],'lambda':lam,'grid':grid,'family':'native quartile refinement','selection_columns':ri,'quantiles':np.asarray(combined_q)[ri]}
        np.savez_compressed(sd/'discovery_bank.npz',A=all_design['native'][:,orders['native'][:8]],response_numerator=yn,response_denominator=np.asarray(RESPONSE_DEN),meta=mf,candidate_A=ra,candidate_w=rw,candidate_threshold=rt,h4=h4,h4prime=h4p)
        discovery_results={}
        for name,o in configurations.items():
            af,q,qp=design(h4,h4p,o['w'],o['threshold']);av,qv,qpv=design(v4,v4p,o['w'],o['threshold'])
            intercept=(logits_num(hf,model)/(H_DEN*SCALE*10)-encode(4*hf,o['w'],o['threshold'])@o['beta']).mean(axis=0)
            np.savez_compressed(sd/f'report-{name}.npz',w=o['w'],threshold=o['threshold'],beta=o['beta'],intercept=intercept,source=o['source'])
            discovery_results[name]={'fit':score(af@o['beta'],yn),'validation':score(av@o['beta'],yvn),'state_bits':len(o['w']),'pair_bits':2*len(o['w']),'encoder_nonzeros':int(np.count_nonzero(o['w'])),'thresholds':len(o['w']),'decoder_coefficients':10*len(o['w']),'max_abs_encoder_integer':int(np.abs(o['w']).max()),'lambda':o['lambda'],'ridge_validation':o['grid'],'source_columns':o['source'].tolist(),'threshold_in_hidden_integer_units':o['threshold'].tolist(),'selection_columns':o['selection_columns']}
            if 'quantiles' in o: discovery_results[name]['quantiles']=o['quantiles'].tolist()
        floatpred=np.argmax(np.maximum(tr['x']/16@pars[0]+pars[1],0)@pars[2]+pars[3],axis=1);expred=np.argmax(logits_num(ht,model),axis=1)
        record={'seed':seed,'train_float_accuracy':float(np.mean(floatpred==tr['y'])),'train_export_accuracy':float(np.mean(expred==tr['y'])),'train_export_float_agreement':float(np.mean(expred==floatpred)),'discovery_validation_export_accuracy':float(np.mean(np.argmax(logits_num(hv,model),axis=1)==dv['y'])),'groups':pairs,'candidate_orders':orders,'selection_traces':selection,'repair_trace':rtrace,'results':discovery_results}
        dump(sd/'discovery_results.json',record);records.append(record)
        print(json.dumps({'phase':'discovery_complete','seed':seed,'export_train_accuracy':record['train_export_accuracy'],'native8_val_mse':discovery_results['native-8']['validation']['mse_sum_outputs'],'repair12_val_mse':discovery_results['repair-12']['validation']['mse_sum_outputs']}),flush=True)
    dump(HERE/'discovery_summary.json',records)
    frozen={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.glob('seed-*/*')) if p.is_file()}
    frozen.update({p:sha(HERE/p) for p in ['experiment.py','contract.json','test_lock.json','test.npz','discovery_summary.json']})
    write_once(HERE/'freeze.json',{'sealed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'test_evaluation_started':False,'files':frozen,'scope':'all model weights, report maps, selected thresholds, decoder coefficients, discovery outputs and analysis code fixed before opening locked outcomes'})
    print(json.dumps({'phase':'sealed','freeze_sha256':sha(HERE/'freeze.json')}),flush=True)

def evaluate():
    if (HERE/'test_results.json').exists(): raise FileExistsError('Locked test already evaluated.')
    lock=json.loads((HERE/'freeze.json').read_text())
    for f,d in lock['files'].items():
        if sha(HERE/f)!=d: raise ValueError('Freeze mismatch: '+f)
    started=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    c=json.loads((HERE/'contract.json').read_text());te=np.load(HERE/'test.npz');groups=np.asarray(c['interventions']['groups']);results=[]
    for seed in SEEDS:
        sd=HERE/f'seed-{seed}';model=np.load(sd/'model.npz');h=hidden(te['x'],model)
        logits=logits_num(h,model);pred=np.argmax(logits,axis=1)
        fp=np.maximum(te['x']/16@model['float_w1']+model['float_b1'],0)@model['float_w2']+model['float_b2'];fpc=fp-fp.mean(axis=1,keepdims=True)
        banks={'seen_strength':bank(h,te['ids'],[.5,1.],groups),'unseen_strength':bank(h,te['ids'],[.25,.75],groups),'natural_pairs':natural_bank(h,te['ids'])}
        rec={'seed':seed,'test_n':len(h),'export_accuracy':float(np.mean(pred==te['y'])),'float_accuracy':float(np.mean(np.argmax(fp,axis=1)==te['y'])),'export_float_prediction_agreement':float(np.mean(pred==np.argmax(fp,axis=1))),'export_float_max_logit_error':float(np.max(np.abs(logits/(H_DEN*SCALE*10)-fpc))),'banks':{}}
        for bank_name,(h4,h4p,meta) in banks.items():
            yn=target_num(h4,h4p,model);res={}
            for path in sorted(sd.glob('report-*.npz')):
                name=path.stem.removeprefix('report-');o=np.load(path);a,q,qp=design(h4,h4p,o['w'],o['threshold']);estimate=a@o['beta']
                res[name]={'prediction':score(estimate,yn),'report_pair':collision_summary(np.concatenate([q,qp],axis=1),yn),'signed_difference':collision_summary(a,yn)}
                if name=='native-8':
                    disc=np.load(sd/'discovery_bank.npz');candidate_a,_,_=design(h4,h4p,disc['candidate_w'],disc['candidate_threshold'])
                    np.savez_compressed(sd/f'test_bank_{bank_name}.npz',A=a,response_numerator=yn,response_denominator=np.asarray(RESPONSE_DEN),meta=meta,candidate_A=candidate_a,report_before=q,report_after=qp)
                if bank_name=='natural_pairs':
                    approx=encode(4*h,o['w'],o['threshold'])@o['beta']+o['intercept']
                    res[name]['baseline_prediction_agreement']=float(np.mean(np.argmax(approx,axis=1)==pred))
                    res[name]['baseline_task_accuracy']=float(np.mean(np.argmax(approx,axis=1)==te['y']))
            original_z=4*(h@model['w2']+H_DEN*model['b2']);intervened_z=h4p@model['w2']+4*H_DEN*model['b2']
            base_z=h4@model['w2']+4*H_DEN*model['b2']
            assert np.allclose((intervened_z-base_z)-(h4p-h4)@model['w2'],0)
            rec['banks'][bank_name]={'n_rows':len(yn),'distinct_images':len(h),'response_denominator':RESPONSE_DEN,'class_flip_fraction':float(np.mean(np.argmax(base_z,axis=1)!=np.argmax(intervened_z,axis=1))),'results':res}
        results.append(rec)
        print(json.dumps({'phase':'test_complete','seed':seed,'export_accuracy':rec['export_accuracy'],'unseen_native8_mse':rec['banks']['unseen_strength']['results']['native-8']['prediction']['mse_sum_outputs'],'unseen_repair12_mse':rec['banks']['unseen_strength']['results']['repair-12']['prediction']['mse_sum_outputs']}),flush=True)
    write_once(HERE/'test_results.json',{'test_started_utc':started,'test_completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'freeze_sha256':sha(HERE/'freeze.json'),'contract_sha256':sha(HERE/'contract.json'),'test_sha256':sha(HERE/'test.npz'),'results':results})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','fit','evaluate']);args=parser.parse_args()
    {'prepare':prepare,'fit':fit,'evaluate':evaluate}[args.phase]()
