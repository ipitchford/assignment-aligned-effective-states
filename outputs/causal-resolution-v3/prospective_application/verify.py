#!/usr/bin/env python3
"""Independent integer checks, policy-certificate replay and fixed illustrations."""
from pathlib import Path
from fractions import Fraction
import hashlib,json
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parent;S=4096;H=65536

# Public presentation adapter: scientific bytes remain exact; old prose identity
# is resolved only through a pinned map, not claimed as a current byte hash.
def presentation_identity(data):
    actual = hashlib.sha256(data).hexdigest()
    for parent in Path(__file__).resolve().parents:
        mapping = parent / "PUBLIC_PRESENTATION_DERIVATIVES.json"
        if mapping.is_file():
            raw = mapping.read_bytes()
            if hashlib.sha256(raw).hexdigest() != "324de49b83ab2bc5d63606852d56f837ab6c91ccca031defc1e83a2798eef618":
                raise RuntimeError("Presentation map identity mismatch")
            for row in json.loads(raw)["files"]:
                if actual == row["public_sha256"] and len(data) == row["public_bytes"]:
                    return row["original_sha256"]
            return actual
    raise RuntimeError("Missing pinned presentation map")

def sha(p):return presentation_identity(Path(p).read_bytes())
checks=[]
def check(ok,label):
    if not ok:raise AssertionError(label)
    checks.append(label)
def exact(v,d):return str(Fraction(int(v),int(d)))
def forward(h,m,scale,deep):
    if deep:h=np.maximum(h@m['w2'].astype(object)+scale*H*m['b2'].astype(object),0);z=h@m['w3'].astype(object)+scale*H*S*m['b3'].astype(object)
    else:z=h@m['w2'].astype(object)+scale*H*m['b2'].astype(object)
    return 10*z-z.sum(axis=1,keepdims=True)

def main():
    results=json.loads((P/'results.json').read_text());dc=json.loads((P/'decision_certificates.json').read_text());contract=json.loads((P/'analysis_contract.json').read_text());te=np.load(ROOT/'application'/'test.npz');lookup={int(x):j for j,x in enumerate(te['ids'])};groups=contract['actions']['groups'];cases=[];rows=0
    check(sha(P/'audit.py')==results['script_sha256'],'audit source hash')
    check(sha(P/'analysis_contract.json')==results['contract_sha256'],'contract hash')
    check(sha(P/'decision_certificates.py')==dc['script_sha256'],'decision-certificate source hash')
    for r in results['results']:
        seed=r['seed'];deep=r['architecture']=='nonlinear';src=ROOT/('nonlinear_application' if deep else 'application')/f'seed-{seed}';dest=P/f'seed-{seed}';m=np.load(src/'model.npz');check(sha(src/'model.npz')==r['model_sha256'],f'{seed} original checkpoint unchanged');h=np.maximum(te['x'].astype(object)@m['w1'].astype(object)+16*m['b1'].astype(object),0);den=4*H*S*(S if deep else 1)*10
        codes={}
        for name in ['native-8','native-12','pca-8','random-8','current-output','action-only']:
            saved=np.load(dest/f'predictions-unseen_strength-{name}.npz')['state_codes']
            if name=='action-only':q=np.empty((len(h),0))
            elif name=='current-output':
                e=np.load(dest/'current_output_encoder.npz');current=forward(h,m,1,deep);levels=np.asarray([1<<int(b) for b in e['bits']],dtype=object);q=np.minimum(((current-e['lower'])*levels)//np.where(e['width']>0,e['width'],1),levels-1);q[:,e['width']==0]=0
            else:
                ep=(dest/'random_encoder.npz') if name=='random-8' and not deep else src/f'report-{name}.npz';e=np.load(ep);q=(h@e['w'].T>e['threshold']).astype(np.int64)
            check(np.array_equal(q,saved),f'{seed}/{name} code computed from pre-action state only');codes[name]=saved
        for bn in ['seen_strength','unseen_strength']:
            b=np.load(dest/f'bank-{bn}.npz');meta=b['meta'];hh=[];hp=[]
            for im,g,s in meta:
                a=h[lookup[int(im)]];ap=4*a.copy();ap[groups[int(g)]]=(4-int(s))*a[groups[int(g)]];hh.append(4*a);hp.append(ap)
            hh=np.asarray(hh,dtype=object);hp=np.asarray(hp,dtype=object);true=forward(hp,m,4,deep)-forward(hh,m,4,deep);rows+=len(true)
            check(np.array_equal(true,b['response_numerator']),f'{seed}/{bn} all target response numerators replay')
            check(int(b['response_denominator'])==den,f'{seed}/{bn} target denominator')
            for name,q in codes.items():
                fit=np.load(dest/f'decoder-{name}.npz');pred=np.empty((len(meta),10));d=q.shape[1]+1
                for i,(im,g,s) in enumerate(meta):
                    state=q[lookup[int(im)]];x=np.concatenate([[1.],(state-fit['mean'])/fit['std']]);a=float(s)/4;coef=fit['beta'][2*int(g)*d:(2*int(g)+2)*d];pred[i]=a*(x@coef[:d])+a*(1-a)*(x@coef[d:])
                saved=np.load(dest/f'predictions-{bn}-{name}.npz')['predictions'];check(np.allclose(pred,saved,rtol=0,atol=1e-11),f'{seed}/{bn}/{name} prediction uses original code and action only')
            pp=np.load(src/('precision.npz' if deep else 'precision-gain-1.npz'));M=pp['M'].astype(object);lev=np.asarray([1<<int(v) for v in pp['bits']],dtype=object);Q=max(map(int,lev));q=np.minimum((h*lev)//np.where(M>0,M,1),lev-1);q[:,M==0]=0;mid=M*(2*q+1)*(Q//lev);hh=[];hp=[]
            for im,g,s in meta:
                x=mid[lookup[int(im)]];xp=4*x.copy();xp[groups[int(g)]]=(4-int(s))*x[groups[int(g)]];hh.append(4*x);hp.append(xp)
            pn=forward(np.asarray(hp,dtype=object),m,8*Q,deep)-forward(np.asarray(hh,dtype=object),m,8*Q,deep);pd=8*Q*H*S*(S if deep else 1)*10;err=pn-true*(2*Q);mx=max(abs(int(v)) for v in err.flat);stored=r['banks'][bn]['results']['native-precision'];check(mx==stored['exact_max_error_numerator'] and pd==stored['exact_prediction_denominator'],f'{seed}/{bn} precision response from reconstructed state only');check(mx<=pd,f'{seed}/{bn} exact epsilon1 pass')
        b=np.load(dest/'bank-unseen_strength.npz');sig=b['signature_second_strength'].reshape(-1,4,10);z=b['current_output_numerator'];order=np.argsort(-z,axis=1,kind='stable');top=order[:,0];runner=order[:,1]
        drop=np.asarray([[int(sig[i,g,runner[i]])-int(sig[i,g,top[i]]) for g in range(4)] for i in range(len(h))],dtype=object);regret=drop.max(axis=1,keepdims=True)-drop
        for name,q in codes.items():
            cert=next(v for v in dc['records'] if v['seed']==seed and v['method']==name);sign=np.column_stack([q,top,runner]);_,inv=np.unique(sign,axis=0,return_inverse=True);meanN=0;worstN=0
            for f in np.unique(inv):
                rr=regret[inv==f];meanN+=int(rr.sum(axis=0).min());worstN=max(worstN,int(rr.max(axis=0).min()))
            check(exact(meanN,len(h)*den)==cert['empirical_minimum_mean_regret_exact'],f'{seed}/{name} exact mean-regret oracle')
            check(exact(worstN,den)==cert['minimum_worst_case_regret_exact'],f'{seed}/{name} exact worst-regret oracle')
            w=cert['two_state_witness']
            if w:
                i,j=w['rows'];check(np.array_equal(q[i],q[j]) and top[i]==top[j] and runner[i]==runner[j],f'{seed}/{name} equal report and policy context');rr=np.maximum(regret[i],regret[j]);check(exact(rr.min(),den)==w['any_common_action_worst_regret_exact'],f'{seed}/{name} every common action lower bound')
            if name=='native-8' and w:
                detail={'seed':seed,'architecture':r['architecture'],'certificate':w,'strength':.75,'cases':[]}
                for i in w['rows']:
                    before=z[i];after=before+sig[i];c=int(top[i]);v=int(runner[i]);cl=np.argmax(after,axis=1);marginN=[int(a[c])-int(a[v]) for a in after]
                    detail['cases'].append({'row':i,'image_id':int(b['ids'][i]),'digit_label':int(b['labels'][i]),'baseline_top':c,'baseline_runner':v,'baseline_top_runner_margin_exact':exact(int(before[c])-int(before[v]),den),'baseline_top_runner_margin':float(Fraction(int(before[c])-int(before[v]),den)),'after_top_runner_margins_exact':[exact(x,den) for x in marginN],'after_top_runner_margins':[float(Fraction(x,den)) for x in marginN],'after_predicted_classes':cl.tolist(),'class_flip_actions':np.flatnonzero(cl!=c).tolist(),'after_all_centered_logits_exact':[[exact(a,den) for a in row] for row in after]})
                flips=[set(v['class_flip_actions']) for v in detail['cases']];detail['both_have_a_flipping_action']=all(flips);detail['no_common_class_flipping_action']=not bool(flips[0]&flips[1]);cases.append(detail)
    dump={'status':'passed','checks':len(checks),'response_rows_replayed':rows,'arithmetic':'independent Python-integer forward evaluation and exact rational decision enumeration','predictor_input_check':'all fitted predictions independently rebuilt from pre-action code and action only; no target or post-action state supplied','scope':'same-team arithmetic verification; retrospective exploratory study, not independent scientific replication','script_sha256':sha(__file__)}
    (P/'verification.json').write_text(json.dumps(dump,indent=2,sort_keys=True)+'\n');(P/'classification_witnesses.json').write_text(json.dumps({'status':'descriptive fixed-strength illustrations, exact checked','cases':cases},indent=2,sort_keys=True)+'\n')
    print(json.dumps(dump))

if __name__=='__main__':main()
