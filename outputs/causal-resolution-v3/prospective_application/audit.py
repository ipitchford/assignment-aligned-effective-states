#!/usr/bin/env python3
"""Exploratory pre-intervention-report action prediction; no model retraining.

Prospective describes query timing, not preregistration. All image splits and
many previous test outcomes were inspected before this study was designed.
"""
from pathlib import Path
from fractions import Fraction
import hashlib,json,sys,time
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parent
sys.path.insert(0,str(ROOT/'application'))
import experiment as util
S=4096;H=65536

def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def endpoint(hnum,m,scale,deep):
    hnum=hnum.astype(object)
    if deep:
        h2=np.maximum(hnum@m['w2'].astype(object)+scale*H*m['b2'].astype(object),0)
        z=h2@m['w3'].astype(object)+scale*H*S*m['b3'].astype(object)
    else:z=hnum@m['w2'].astype(object)+scale*H*m['b2'].astype(object)
    return 10*z-z.sum(axis=1,keepdims=True)

def projection_encoder(m,M,deep):
    if deep:
        w=m['w2'].astype(object);lo2=np.maximum(0,M.astype(object)@np.minimum(w,0)+H*m['b2']);hi2=np.maximum(0,M.astype(object)@np.maximum(w,0)+H*m['b2'])
        last=m['w3'];bias=m['b3'];c=(10*last-last.sum(axis=1,keepdims=True)).astype(object);bc=10*bias-bias.sum();lo=lo2@np.maximum(c,0)+hi2@np.minimum(c,0)+H*S*bc;hi=hi2@np.maximum(c,0)+lo2@np.minimum(c,0)+H*S*bc;den=H*S*S*10
    else:
        last=m['w2'];bias=m['b2'];c=(10*last-last.sum(axis=1,keepdims=True)).astype(object);bc=10*bias-bias.sum();lo=M.astype(object)@np.minimum(c,0)+H*bc;hi=M.astype(object)@np.maximum(c,0)+H*bc;den=H*S*10
    width=hi-lo;bits=[]
    for w in width:
        b=0
        while int(w)>den*(1<<b):b+=1
        bits.append(b)
    return {'lower':np.asarray(lo,dtype=np.int64),'width':np.asarray(width,dtype=np.int64),'bits':np.asarray(bits,dtype=np.int64),'denominator':den}

def project_code(h,m,enc,deep):
    v=endpoint(h,m,1,deep);levels=np.asarray([1<<int(b) for b in enc['bits']],dtype=object);width=enc['width'].astype(object);safe=np.where(width>0,width,1)
    q=np.minimum(((v-enc['lower'])*levels)//safe,levels-1);q[:,width==0]=0
    assert np.all(q>=0) and np.all(q<levels)
    return q.astype(np.int64)

def precision_code(h,M,bits):
    levels=np.left_shift(1,bits);safe=np.where(M>0,M,1);q=np.minimum((h*levels)//safe,levels-1);q[:,M==0]=0;return q

def action_design(code,groups,strength,mean,std):
    """This interface has no hprime, input image, label, or target response."""
    z=np.column_stack([np.ones(len(code)),(code-mean)/std]);d=z.shape[1];a=np.zeros((len(code),8*d),float)
    for g in range(4):
        ix=np.flatnonzero(groups==g);a[ix,2*g*d:(2*g+1)*d]=strength[ix,None]*z[ix];a[ix,(2*g+1)*d:(2*g+2)*d]=(strength[ix]*(1-strength[ix]))[:,None]*z[ix]
    return a

def predict(code,groups,strength,fit):
    return action_design(code,groups,strength,fit['mean'],fit['std'])@fit['beta']

def train_decoder(qf,qv,fmeta,vmeta,y,yv):
    mean=qf.mean(axis=0);std=qf.std(axis=0);std[std==0]=1
    af=action_design(qf,fmeta[:,1],fmeta[:,2]/4,mean,std);av=action_design(qv,vmeta[:,1],vmeta[:,2]/4,mean,std)
    beta,lam,grid=util.choose_decoder(af,y,av,yv,[1e-6,.0001,.01,.1,1.])
    return {'mean':mean,'std':std,'beta':beta},lam,grid

def response_metrics(pred,ynum,den):return util.score(pred,ynum,den)

def state_oracle(code,ynum,den):
    """ynum: states x (four groups times ten response coordinates), same strength."""
    _,inv,counts=np.unique(code,axis=0,return_inverse=True,return_counts=True);ng=len(counts)
    mins=np.full((ng,ynum.shape[1]),np.iinfo(np.int64).max,dtype=np.int64);maxs=np.full_like(mins,np.iinfo(np.int64).min);np.minimum.at(mins,inv,ynum);np.maximum.at(maxs,inv,ynum)
    ranges=(maxs-mins).max(axis=1);sums=np.zeros((ng,ynum.shape[1]));np.add.at(sums,inv,ynum/den);m=sums/counts[:,None];risk=np.mean(np.sum((ynum/den-m[inv])**2,axis=1))/4
    group=int(np.argmax(ranges));coord=int(np.argmax(maxs[group]-mins[group]));ix=np.flatnonzero(inv==group);a=int(ix[np.argmin(ynum[ix,coord])]);b=int(ix[np.argmax(ynum[ix,coord])])
    return {'unique_state_reports':ng,'states':len(code),'empirical_oracle_mse_per_action':float(risk),'exact_minimax_error':float(Fraction(int(ranges.max()),2*den)),'worst_range_numerator':int(ranges.max()),'response_denominator':den,'primary_incompatible_fibres':int((ranges>2*den).sum()),'witness':{'rows':[a,b],'group':coord//10,'coordinate':coord%10,'code':code[a].tolist(),'numerators':[int(ynum[a,coord]),int(ynum[b,coord])]}}

def action_policy(code,true_signature,pred_signature,current_num,den,ids,labels):
    n=len(code);order=np.argsort(-current_num.astype(float),axis=1,kind='stable');top=order[:,0];runner=order[:,1]
    # Extra policy context is only the ordered class pair (90 possibilities).
    # No baseline margin/logit value is given to the response predictor.
    truth=true_signature.reshape(n,4,10)/den;prediction=pred_signature.reshape(n,4,10)
    td=np.empty((n,4));pd=np.empty((n,4))
    for i in range(n):td[i]=truth[i,:,runner[i]]-truth[i,:,top[i]];pd[i]=prediction[i,:,runner[i]]-prediction[i,:,top[i]]
    actual=np.argmax(td,axis=1);chosen=np.argmax(pd,axis=1);regret=td[np.arange(n),actual]-td[np.arange(n),chosen]
    signatures=np.column_stack([code,top,runner]);_,inv=np.unique(signatures,axis=0,return_inverse=True);best=None
    for f in np.unique(inv):
        ix=np.flatnonzero(inv==f)
        for ai,i in enumerate(ix):
            for j in ix[ai+1:]:
                gi,gj=actual[i],actual[j]
                if gi==gj:continue
                gap=min(td[i,gi]-td[i,gj],td[j,gj]-td[j,gi])
                item=(float(gap),int(i),int(j))
                if best is None or item>best:best=item
    witness=None
    if best:
        gap,i,j=best;witness={'rows':[i,j],'input_ids':[int(ids[i]),int(ids[j])],'digit_labels':[int(labels[i]),int(labels[j])],'same_top_class':int(top[i]),'same_runner_class':int(runner[i]),'same_state_code':code[i].tolist(),'best_groups':[int(actual[i]),int(actual[j])],'all_true_margin_drops':[td[i].tolist(),td[j].tolist()],'cross_preferred_action_regret_min':gap}
    return {'query':'choose group that most decreases baseline top-class versus runner-up margin at strength3/4','extra_policy_context_bits':7,'top_runner_context_supplied_as_labels_only':True,'best_group_accuracy':float(np.mean(chosen==actual)),'mean_action_regret':float(np.mean(regret)),'p95_action_regret':float(np.quantile(regret,.95)),'max_action_regret':float(regret.max()),'fraction_response_margin_change_sign_errors':float(np.mean((td>0)!=(pd>0))),'same_code_same_class_pair_different_action_witness':witness}

def precision_predict(q,M,bits,m,groups,strength_numerators,deep,group_units):
    levels=np.left_shift(1,bits);Q=int(levels.max());mid=M*(2*q+1)*(Q//levels);before=4*mid;after=before.copy()
    for g in range(4):
        ix=np.flatnonzero(groups==g)
        for j in group_units[g]:after[ix,j]=(4-strength_numerators[ix])*mid[ix,j]
    pn=endpoint(after,m,8*Q,deep)-endpoint(before,m,8*Q,deep);pd=8*Q*H*S*(S if deep else 1)*10
    return pn,pd

def main():
    if (P/'results.json').exists():raise FileExistsError('Refuse to overwrite completed exploratory analysis')
    P.mkdir(exist_ok=True);original=json.loads((ROOT/'application'/'contract.json').read_text());units=np.asarray(original['interventions']['groups']);tr=np.load(ROOT/'application'/'train.npz');df=np.load(ROOT/'application'/'discovery_fit.npz');dv=np.load(ROOT/'application'/'discovery_validation.npz');te=np.load(ROOT/'application'/'test.npz')
    contract={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'status':'post-test exploratory reuse; prospective means prediction before the action, not preregistration; existing test images and previous outcomes already inspected','models':'reuse shallow17/29/43 and nonlinear117/129/143; no neural retraining','report_timing':'T(h) computed from original state once; T(hprime) never provided','actions':{'groups':units.tolist(),'discovery_strengths':[.5,1.],'evaluation_strengths':[.25,.5,.75,1.],'context_bits':4,'definition':'one of four group indices and one of four declared strengths'},'decoder':'one shared polynomial-strength action-conditioned affine model: s B_g[1,standardized_code]+s(1−s) G_g[1,standardized_code]','decoder_coefficients':'80*(number of code coordinates+1); 4groups*2strength terms*10outputs','fitting':'discovery-fit only; standardization on discovery-fit codes; ridge grid1e-6,1e-4,.01,.1,1 chosen on original separate discovery-validation; no selection from current test outcomes','encoders':'reuse original native8/native12/PCA8 and nonlinear random8; shallow random8 uses32 fixed Gaussian candidates with original retrospective discovery selection, explicitly a new exploratory baseline; current-output box quantizer uses width/2^b<=1','projection_warning':'current-output quantizer already computes nominated current readout; its cost and inability to see future state are explicit','precision_control':'reuse previously selected native-coordinate bit allocation; simulate action only on midpoint reconstruction, never on unavailable true state; fixed original downstream computation charged','policy':'at strength3/4 choose group maximizing drop in original top-vs-runner-up margin; ordered top/runner labels provided as7extra context bits; no exact baseline margin supplied; descriptive same-code and same-class-pair witness search','inference':'no population/pristine holdout claim, no semantic or generality claim','source_code_sha256':sha(__file__)}
    dump(P/'analysis_contract.json',contract);records=[]
    for deep,seeds,folder in [(False,[17,29,43],'application'),(True,[117,129,143],'nonlinear_application')]:
      for seed in seeds:
        src=ROOT/folder/f'seed-{seed}';dest=P/f'seed-{seed}';dest.mkdir();m=np.load(src/'model.npz');M=np.maximum(0,16*(np.maximum(m['w1'],0).sum(axis=0)+m['b1']));hs={name:np.maximum(d['x']@m['w1']+16*m['b1'],0) for name,d in [('train',tr),('fit',df),('validation',dv),('test',te)]}
        den=4*H*S*(S if deep else 1)*10
        f,fp,fmeta=util.bank(hs['fit'],df['ids'],[.5,1.],units);v,vp,vmeta=util.bank(hs['validation'],dv['ids'],[.5,1.],units)
        yf=(endpoint(fp,m,4,deep)-endpoint(f,m,4,deep)).astype(np.int64);yv=(endpoint(vp,m,4,deep)-endpoint(v,m,4,deep)).astype(np.int64)
        projections=projection_encoder(m,M,deep);np.savez_compressed(dest/'current_output_encoder.npz',**projections)
        encoders={}
        for name in ['native-8','native-12','pca-8']+(['random-8'] if deep else []):
            o=np.load(src/f'report-{name}.npz');encoders[name]={'w':o['w'],'threshold':o['threshold'],'state_bits':len(o['w']),'source':str((src/f'report-{name}.npz').relative_to(ROOT)),'new':False}
        if not deep:
            rng=np.random.default_rng(seed+5671);w=rng.normal(size=(32,32));w=np.rint(S*w/np.linalg.norm(w,axis=1,keepdims=True)).astype(np.int64);proj=hs['train']@w.T;t=np.array([util.quantile_int(proj[:,j],.5) for j in range(32)],dtype=np.int64);a,_,_=util.design(f,fp,w,t);order,trace=util.greedy_order(a,yf/den,8);encoders['random-8']={'w':w[order],'threshold':t[order],'state_bits':8,'new':True,'selection_trace':trace};np.savez_compressed(dest/'random_encoder.npz',w=w[order],threshold=t[order])
        codes={}
        for name,o in encoders.items():codes[name]={split:util.encode(4*h,o['w'],o['threshold']).astype(np.int64) for split,h in hs.items()}
        codes['current-output']={split:project_code(h,m,projections,deep) for split,h in hs.items()};codes['action-only']={split:np.empty((len(h),0)) for split,h in hs.items()}
        current=endpoint(4*hs['test'],m,4,deep).astype(np.int64);rec={'seed':seed,'architecture':'nonlinear' if deep else 'linear','target_denominator':den,'model_sha256':sha(src/'model.npz'),'current_output_state_bits':int(projections['bits'].sum()),'current_output_encoder_cost':698 if deep else 320,'methods':{},'banks':{}}
        fits={}
        for name,co in codes.items():
            qf=np.tile(co['fit'],(8,1));qv=np.tile(co['validation'],(8,1));fit,lam,grid=train_decoder(qf,qv,fmeta,vmeta,yf/den,yv/den);fits[name]=fit
            np.savez_compressed(dest/f'decoder-{name}.npz',**fit)
            bits=int(projections['bits'].sum()) if name=='current-output' else (0 if name=='action-only' else encoders[name]['state_bits'])
            rec['methods'][name]={'state_bits':bits,'action_query_bits':4,'total_query_bits':bits+4,'code_coordinates':co['fit'].shape[1],'decoder_coefficients':int(fit['beta'].size),'lambda':lam,'validation_grid':grid,'fit_mse':response_metrics(predict(qf,fmeta[:,1],fmeta[:,2]/4,fit),yf,den)['mse_sum_outputs']}
        for bn,strengths in [('seen_strength',[.5,1.]),('unseen_strength',[.25,.75])]:
            before,after,meta=util.bank(hs['test'],te['ids'],strengths,units);yn=(endpoint(after,m,4,deep)-endpoint(before,m,4,deep)).astype(np.int64);res={};signatures={}
            for ni,sn in enumerate([int(4*s) for s in strengths]):signatures[sn]=yn[ni*4*len(te['ids']):(ni+1)*4*len(te['ids'])].reshape(4,len(te['ids']),10).transpose(1,0,2).reshape(len(te['ids']),40)
            for name,co in codes.items():
                q=np.tile(co['test'],(8,1));pred=predict(q,meta[:,1],meta[:,2]/4,fits[name]);res[name]={'prediction':response_metrics(pred,yn,den),'state_response_oracles':{str(sn):state_oracle(co['test'],sig,den) for sn,sig in signatures.items()}}
                if bn=='unseen_strength':
                    sigpred=pred[4*len(te['ids']):].reshape(4,len(te['ids']),10).transpose(1,0,2).reshape(len(te['ids']),40);res[name]['policy']=action_policy(co['test'],signatures[3],sigpred,current,den,te['ids'],te['y'])
                np.savez_compressed(dest/f'predictions-{bn}-{name}.npz',state_codes=co['test'],predictions=pred)
            # Existing native precision report, with action computed from its reconstruction only.
            pp=np.load(src/('precision.npz' if deep else 'precision-gain-1.npz'));q=precision_code(hs['test'],pp['M'],pp['bits']);qrows=np.tile(q,(8,1));pn,pd=precision_predict(qrows,pp['M'],pp['bits'],m,meta[:,1],meta[:,2],deep,units);scale=pd//den;err=pn-yn.astype(object)*scale;maximum=max(abs(int(z)) for z in err.flat);assert maximum<=pd
            pred=pn.astype(float)/pd;res['native-precision']={'prediction':response_metrics(pred,yn,den),'state_bits':int(pp['bits'].sum()),'decoder_coefficients':698 if deep else 320,'exact_max_error_numerator':maximum,'exact_prediction_denominator':pd,'exact_epsilon1_pass':True,'no_true_postaction_state_used_by_predictor':True}
            if bn=='unseen_strength':
                sigpred=pred[4*len(te['ids']):].reshape(4,len(te['ids']),10).transpose(1,0,2).reshape(len(te['ids']),40);res['native-precision']['policy']=action_policy(q,signatures[3],sigpred,current,den,te['ids'],te['y'])
            np.savez_compressed(dest/f'bank-{bn}.npz',response_numerator=yn,response_denominator=np.asarray(den),meta=meta,ids=te['ids'],labels=te['y'],current_output_numerator=current,signature_second_strength=signatures[int(4*strengths[1])])
            rec['banks'][bn]={'rows':len(yn),'results':res}
        records.append(rec);dump(dest/'results.json',rec)
        print(json.dumps({'seed':seed,'architecture':rec['architecture'],'output_report_bits':rec['current_output_state_bits'],'unseen':{n:r['prediction']['mse_sum_outputs'] for n,r in rec['banks']['unseen_strength']['results'].items()}}),flush=True)
    dump(P/'results.json',{'status':'exploratory complete','timing':'all choices made after prior test inspection; no preregistration or untouched-holdout claim','contract_sha256':sha(P/'analysis_contract.json'),'script_sha256':sha(__file__),'results':records})

if __name__=='__main__':main()
