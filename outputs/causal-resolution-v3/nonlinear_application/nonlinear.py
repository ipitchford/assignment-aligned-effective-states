#!/usr/bin/env python3
"""Exploratory architecture extension, frozen before new-model test outcomes.

Requires NumPy and the preserved sibling application/experiment.py only.
Exact endpoint evaluation uses Python arbitrary-precision integers.
"""
from pathlib import Path
from fractions import Fraction
import argparse,hashlib,json,shutil,sys,time
import numpy as np
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'application'))
import experiment as base

S=4096;H=16*S;DEN=4*H*S*S*10;SEEDS=[117,129,143]
base.RESPONSE_DEN=DEN

def train(xi,y,seed):
    rng=np.random.default_rng(seed);x=xi/16
    sizes=[64,32,16,10];pars=[]
    for a,b in zip(sizes[:-1],sizes[1:]):pars += [rng.normal(0,np.sqrt(2/a),(a,b)),np.zeros(b)]
    m=[np.zeros_like(p) for p in pars];v=[np.zeros_like(p) for p in pars];t=0
    for ep in range(200):
        lr=.003 if ep<150 else .001
        order=rng.permutation(len(x))
        for start in range(0,len(x),128):
            ix=order[start:start+128];xx=x[ix]
            a=np.maximum(xx@pars[0]+pars[1],0);b=np.maximum(a@pars[2]+pars[3],0);z=b@pars[4]+pars[5]
            dz=base.softmax(z);dz[np.arange(len(ix)),y[ix]]-=1;dz/=len(ix)
            db=(dz@pars[4].T)*(b>0);da=(db@pars[2].T)*(a>0)
            gs=[xx.T@da+.0001*pars[0],da.sum(axis=0),a.T@db+.0001*pars[2],db.sum(axis=0),b.T@dz+.0001*pars[4],dz.sum(axis=0)]
            t+=1
            for j,g in enumerate(gs):
                m[j]=.9*m[j]+.1*g;v[j]=.999*v[j]+.001*g*g
                pars[j]-=lr*(m[j]/(1-.9**t))/(np.sqrt(v[j]/(1-.999**t))+1e-8)
    return pars

def h1(x,m):return np.maximum(x@m['w1']+16*m['b1'],0)

def endpoint(hnum,m,scale=4):
    """Returns centered logits numerator with denominator scale*H*S*S*10."""
    a=np.maximum(hnum.astype(object)@m['w2'].astype(object)+scale*H*m['b2'].astype(object),0)
    z=a@m['w3'].astype(object)+scale*H*S*m['b3'].astype(object)
    return 10*z-z.sum(axis=1,keepdims=True)

def targets(h,hp,m):
    z=endpoint(hp,m)-endpoint(h,m)
    if max(abs(int(v)) for v in z.flat)>=2**63:raise OverflowError('Saved target needs wider integer representation')
    return z.astype(np.int64)

def float_logits(x,m):
    a=np.maximum(x/16@m['float_w1']+m['float_b1'],0)
    b=np.maximum(a@m['float_w2']+m['float_b2'],0)
    z=b@m['float_w3']+m['float_b3'];return z-z.mean(axis=1,keepdims=True)

def precision(m):
    M=np.maximum(0,16*(np.maximum(m['w1'],0).sum(axis=0)+m['b1']))
    C3=10*m['w3']-m['w3'].sum(axis=1,keepdims=True)
    C=np.abs(m['w2']).astype(object)@np.abs(C3).astype(object)
    d=H*S*S*10;terms=[[Fraction(int(M[j])*int(C[j,k]),d) for k in range(10)] for j in range(32)]
    totals=[sum(terms[j][k] for j in range(32)) for k in range(10)];bits=np.zeros(32,dtype=np.int64);trace=[]
    while max(totals)>1:
        opts=[]
        for j in range(32):
            if M[j]==0:continue
            after=[totals[k]-terms[j][k]/2 for k in range(10)]
            opts.append((max(after),sum(after),j))
        _,_,j=min(opts);bits[j]+=1;trace.append(j);terms[j]=[t/2 for t in terms[j]];totals=[totals[k]-terms[j][k] for k in range(10)]
    initial=max(sum(Fraction(int(M[j])*int(C[j,k]),d) for j in range(32)) for k in range(10))
    uniform=0
    while initial/(1<<uniform)>1:uniform+=1
    return M,C.astype(np.int64),bits,{'state_bits':int(bits.sum()),'uniform_state_bits':int((M>0).sum())*uniform,'uniform_bits_per_active_coordinate':uniform,'max_bound':float(max(totals)),'bounds_exact':[str(t) for t in totals],'gain_order':trace,'denominator':d,'lossless_h_integer_box_bits':sum(int(v).bit_length() for v in M),'lossless_quarter_box_bits':sum(int(4*v).bit_length() for v in M),'decoder_weights_and_biases':698,'reason':'conservative absolute-weight Lipschitz product through intervening ReLU, not a tight bit optimum'}

def prepare():
    if (P/'contract.json').exists():raise FileExistsError('Already prepared')
    P.mkdir(exist_ok=True)
    for name in ['train.npz','discovery_fit.npz','discovery_validation.npz','test.npz']:
        shutil.copyfile(P.parent/'application'/name,P/name)
    old=json.loads((P.parent/'application'/'contract.json').read_text())
    contract={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'status':'exploratory post-test architecture extension; same image split already inspected in shallow-model study; all extension choices frozen before evaluating these new models on test','purpose':'test whether report and decoder diagnoses persist at a cut with nonlinear downstream computation','architecture':[64,32,16,10],'activation':'ReLU at both hidden layers','cut':'first 32-unit hidden layer','seeds':SEEDS,'training':old['model']|{'seeds':SEEDS,'architecture':[64,32,16,10]},'split':old['split'],'groups':old['interventions']['groups'],'discovery_strengths':[.5,1.],'test_banks':{'seen_strength':[.5,1.],'unseen_strength':[.25,.75]},'reports':{'native':[8,12],'pca':[8],'random':[8]},'all_candidate_counts':32,'random_projection':'32 Gaussian rows normalized to unit norm, seed model_seed+5671, rounded at scale4096; training-median thresholds','selection':'same discovery-fit greedy response MSE with selection ridge0.01; candidate median thresholds and PCA from training only; decoder ridge selected on separate discovery validation','decoder':'shared B[T(hprime)−T(h)], no intercept or context,10B coefficients','ridge_grid':[1e-6,.0001,.01,.1,1.],'primary_epsilon':1.,'sensitivity_epsilons':[.25,.5,1.,2.,4.],'response':f'exact centered-logit difference numerator / {DEN}; Python arbitrary-precision endpoint arithmetic before checked int64 storage','precision_supplement':'native coordinate bins, weight-only greedy bits, absolute-weight product bound through ReLU, epsilon1; original nonlinear downstream computation of midpoint states,698 weights/biases charged','prior_outcomes':'shallow-model test outcomes motivated architecture extension and are already known; this is not a new pristine dataset holdout','files':{name:base.sha(P/name) for name in ['train.npz','discovery_fit.npz','discovery_validation.npz','test.npz']},'dependency_sha256':base.sha(P.parent/'application'/'experiment.py'),'source_code_sha256':base.sha(__file__)}
    base.write_once(P/'contract.json',contract)
    print(json.dumps({'phase':'prepared','contract_sha256':base.sha(P/'contract.json')}),flush=True)

def fit():
    if (P/'freeze.json').exists():raise FileExistsError('Already frozen')
    c=json.loads((P/'contract.json').read_text());tr=np.load(P/'train.npz');df=np.load(P/'discovery_fit.npz');dv=np.load(P/'discovery_validation.npz');groups=np.asarray(c['groups']);records=[]
    for seed in SEEDS:
        sd=P/f'seed-{seed}';sd.mkdir(exist_ok=False);pars=train(tr['x'],tr['y'],seed);names=['w1','b1','w2','b2','w3','b3']
        m={n:np.rint(p*S).astype(np.int64) for n,p in zip(names,pars)};np.savez_compressed(sd/'model.npz',**m,**{f'float_{n}':p for n,p in zip(names,pars)})
        ht=h1(tr['x'],m);hf=h1(df['x'],m);hv=h1(dv['x'],m)
        f,fp,meta=base.bank(hf,df['ids'],[.5,1.],groups);v,vp,vmeta=base.bank(hv,dv['ids'],[.5,1.],groups)
        yn=targets(f,fp,m);ynv=targets(v,vp,m);y=yn/DEN;yv=ynv/DEN
        pool,_=base.encoder_candidates(ht);rng=np.random.default_rng(seed+5671);rw=rng.normal(size=(32,32));rw/=np.linalg.norm(rw,axis=1,keepdims=True);rw=np.rint(rw*S).astype(np.int64)
        projection=ht@rw.T;rt=np.asarray([base.quantile_int(projection[:,j],.5) for j in range(32)],dtype=np.int64);pool['random']={'w':rw,'threshold':rt,'source':np.arange(32)}
        rec={'seed':seed,'reports':{}}
        for family,budgets in c['reports'].items():
            o=pool[family];a,_,_=base.design(f,fp,o['w'],o['threshold']);av,_,_=base.design(v,vp,o['w'],o['threshold']);order,trace=base.greedy_order(a,y,max(budgets))
            for bits in budgets:
                ix=order[:bits];beta,lam,grid=base.choose_decoder(a[:,ix],y,av[:,ix],yv,c['ridge_grid']);name=f'{family}-{bits}'
                np.savez_compressed(sd/f'report-{name}.npz',w=o['w'][ix],threshold=o['threshold'][ix],beta=beta,source=o['source'][ix])
                rec['reports'][name]={'fit':base.score(a[:,ix]@beta,yn,DEN),'validation':base.score(av[:,ix]@beta,ynv,DEN),'selected_columns':ix,'selection_trace':trace,'lambda':lam,'ridge_validation':grid,'state_bits':bits,'pair_bits':2*bits,'encoder_nonzeros':int(np.count_nonzero(o['w'][ix])),'decoder_coefficients':bits*10}
                if name=='native-8':
                    cand_order=ix+[j for j in range(32) if j not in ix]
                    np.savez_compressed(sd/'discovery_bank.npz',A=a[:,ix],response_numerator=yn,response_denominator=np.asarray(DEN),meta=meta,candidate_A=a[:,cand_order],candidate_w=o['w'][cand_order],candidate_threshold=o['threshold'][cand_order])
        M,C,bits,cert=precision(m);np.savez_compressed(sd/'precision.npz',M=M,C=C,bits=bits);rec['precision']=cert
        z=endpoint(4*ht,m);rec['training_accuracy']=float(np.mean(np.argmax(z,axis=1)==tr['y']));rec['discovery_validation_accuracy']=float(np.mean(np.argmax(endpoint(4*hv,m),axis=1)==dv['y']))
        base.dump(sd/'discovery_results.json',rec);records.append(rec)
        print(json.dumps({'phase':'discovery_complete','seed':seed,'train_accuracy':rec['training_accuracy'],'validation_accuracy':rec['discovery_validation_accuracy'],'precision_bits':cert['state_bits']}),flush=True)
    base.dump(P/'discovery_summary.json',records)
    files={str(f.relative_to(P)):base.sha(f) for f in sorted(P.glob('seed-*/*')) if f.is_file()};files.update({f:base.sha(P/f) for f in ['nonlinear.py','contract.json','discovery_summary.json','test.npz']})
    base.write_once(P/'freeze.json',{'sealed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'new_model_test_outcomes_not_opened':True,'files':files,'sibling_experiment_sha256':base.sha(P.parent/'application'/'experiment.py')})
    print(json.dumps({'phase':'sealed','sha256':base.sha(P/'freeze.json')}),flush=True)

def evaluate():
    if (P/'test_results.json').exists():raise FileExistsError('Already evaluated')
    fr=json.loads((P/'freeze.json').read_text())
    for f,d in fr['files'].items():
        if base.sha(P/f)!=d:raise ValueError('Freeze mismatch '+f)
    if base.sha(P.parent/'application'/'experiment.py')!=fr['sibling_experiment_sha256']:raise ValueError('Dependency changed')
    started=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());te=np.load(P/'test.npz');c=json.loads((P/'contract.json').read_text());groups=np.asarray(c['groups']);results=[]
    for seed in SEEDS:
        sd=P/f'seed-{seed}';m=np.load(sd/'model.npz');h=h1(te['x'],m);z=endpoint(4*h,m);fz=float_logits(te['x'],m)
        rec={'seed':seed,'test_accuracy':float(np.mean(np.argmax(z,axis=1)==te['y'])),'float_accuracy':float(np.mean(np.argmax(fz,axis=1)==te['y'])),'export_float_class_agreement':float(np.mean(np.argmax(z,axis=1)==np.argmax(fz,axis=1))),'export_float_max_logit_error':float(np.max(np.abs(z.astype(float)/DEN-fz))),'banks':{}}
        for bn,strengths in c['test_banks'].items():
            a,ap,meta=base.bank(h,te['ids'],strengths,groups);yn=targets(a,ap,m);r={}
            for path in sorted(sd.glob('report-*.npz')):
                o=np.load(path);design,q,qp=base.design(a,ap,o['w'],o['threshold']);name=path.stem.removeprefix('report-')
                r[name]={'prediction':base.score(design@o['beta'],yn,DEN),'report_pair':base.collision_summary(np.concatenate([q,qp],axis=1),yn),'signed_difference':base.collision_summary(design,yn)}
                if name=='native-8':
                    candidates=np.load(sd/'discovery_bank.npz');ca,_,_=base.design(a,ap,candidates['candidate_w'],candidates['candidate_threshold'])
                    np.savez_compressed(sd/f'test_bank_{bn}.npz',A=design,response_numerator=yn,response_denominator=np.asarray(DEN),meta=meta,candidate_A=ca,report_before=q,report_after=qp)
            pp=np.load(sd/'precision.npz');M=pp['M'];bits=pp['bits'];levels=np.left_shift(1,bits);Q=int(levels.max());safe=np.where(M>0,4*M,1)
            q=np.minimum((a*levels)//safe,levels-1);qp=np.minimum((ap*levels)//safe,levels-1);q[:,M==0]=qp[:,M==0]=0
            hhat=M*(2*q+1)*(Q//levels);hhatp=M*(2*qp+1)*(Q//levels)
            pn=endpoint(hhatp,m,scale=2*Q)-endpoint(hhat,m,scale=2*Q);pd=2*Q*H*S*S*10
            error=pn-yn.astype(object)*(Q//2);maximum=max(abs(int(v)) for v in error.flat);assert maximum<=pd
            r['precision']={'prediction':base.score(pn.astype(float)/pd,yn,DEN),'state_bits':int(bits.sum()),'exact_max_error_numerator':maximum,'exact_error_denominator':pd,'exact_primary_tolerance_pass':True}
            before=endpoint(a,m);after=endpoint(ap,m);rec['banks'][bn]={'n':len(yn),'class_flip_fraction':float(np.mean(np.argmax(before,axis=1)!=np.argmax(after,axis=1))),'results':r}
        results.append(rec)
        print(json.dumps({'phase':'test_complete','seed':seed,'accuracy':rec['test_accuracy'],'unseen_native8_mse':rec['banks']['unseen_strength']['results']['native-8']['prediction']['mse_sum_outputs'],'unseen_precision_max':rec['banks']['unseen_strength']['results']['precision']['prediction']['max_abs']}),flush=True)
    base.write_once(P/'test_results.json',{'test_started_utc':started,'freeze_sha256':base.sha(P/'freeze.json'),'scope':'exploratory post-test architecture extension using previously inspected split; all new model/report choices frozen before their test outcomes','results':results})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','fit','evaluate']);args=p.parse_args();{'prepare':prepare,'fit':fit,'evaluate':evaluate}[args.phase]()
