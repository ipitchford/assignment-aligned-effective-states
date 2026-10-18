"""Post-test exploratory matched affine-minimax controls for the shallow study.

The frozen review-round1 inputs are read-only. Selection never reads test arrays.
Every candidate is evaluated in the greedy sweep; duplicate design rows are
collapsed to per-output response extrema by an exactly equivalent LP reduction.
Final models use the original, uncompressed minimax_fit implementation.
"""
from pathlib import Path
from fractions import Fraction as Q
import argparse, hashlib, importlib.util, json, os, platform, sys, time
import numpy as np
import scipy
from scipy.optimize import linprog

P=Path(__file__).resolve().parent
ROOT=P.parent/'review-round1'
APP=ROOT/'application'
CERT=ROOT/'certificates'
sys.path.insert(0,str(CERT))
from affine_audit import minimax_fit, rational_witness
COEF_DEN=2**32
SEEDS=[17,29,43]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def once(p,d):
    if Path(p).exists():raise FileExistsError(p)
    dump(p,d)
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())

def contract():
    files=[CERT/'affine_audit.py']
    for seed in SEEDS:
        sd=APP/f'seed-{seed}'
        files += [sd/'discovery_bank.npz',sd/'report-native-12.npz',CERT/f'seed-{seed}'/'selected_decoder.npz']
    once(P/'contract.json',{
        'created_utc':utc(),'status':'POST-TEST EXPLORATORY control; prior test outcomes already known to the research team and reviewer. This new procedural freeze is not pristine validation or preregistration.',
        'purpose':'Separate added-bit effects and the selection heuristic from matched affine minimax response decoding.',
        'inputs':{str(f.relative_to(ROOT)):sha(f) for f in files},'seeds':SEEDS,
        'methods':['native8_affine_minimax','native12_affine_minimax','contrast12_affine_minimax','greedy_minimax12'],
        'feature_sets':'Native12 uses the original median-predicate selection; contrast12 uses the originally selected contrast-guided quartile set; greedy starts at original native8 and adds four columns from the same native quartile candidate library.',
        'greedy':'Every remaining candidate is evaluated at each of four rounds by all ten independent coordinate minimax LPs; select lowest maximum objective rounded to 9 decimal places, then lowest original candidate index. No test data used. This is numerical greedy selection, not exact minimum-cost repair or an exact proof of greedy-step optimality.',
        'search_speed':'Duplicate design rows replaced by scalar response extrema; mathematically equivalent to the full LP. Pilot and final checks compare compressed/full objectives.',
        'final_decoder':'Original affine_audit.minimax_fit with intercept, one independent coefficient vector per output; all final sets refitted by the same uncompressed solver implementation; nearest integer coefficient numerator over 2^32.',
        'certificates':'For every final method/output, rational lower witness and exact residual upper for dyadic coefficient vector; shared retained design and integer response arrays.',
        'test':'Existing seen-strength, unseen-strength and natural-pair banks; no selection from these outcomes. All final decoder arrays frozen before this script evaluates them.',
        'resources':{'processes':1,'blas_threads':1,'highs_threads':'single global solver thread requested via OMP_NUM_THREADS=1 and HiGHS threads=1 in compressed search'},
        'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform()},
        'script_sha256':sha(__file__)
    })

def compressed_fits(a,nums,den):
    unique,inv=np.unique(a,axis=0,return_inverse=True)
    ng=len(unique);d=a.shape[1]
    lo=np.full((ng,10),np.iinfo(np.int64).max,dtype=np.int64)
    hi=np.full((ng,10),np.iinfo(np.int64).min,dtype=np.int64)
    np.minimum.at(lo,inv,nums);np.maximum.at(hi,inv,nums)
    con=np.r_[np.c_[unique,-np.ones(ng)],np.c_[-unique,-np.ones(ng)]]
    c=np.r_[np.zeros(d),1.]
    fits=[]
    for k in range(10):
        # scipy passes the explicit thread option through to HiGHS.
        fit=linprog(c,A_ub=con,b_ub=np.r_[lo[:,k]/den,-hi[:,k]/den],bounds=[(None,None)]*d+[(0,None)],method='highs',options={'threads':1})
        if not fit.success:raise RuntimeError(fit.message)
        fits.append({'error':float(fit.fun),'coef':fit.x[:d]})
    return fits,ng

def native12_indices(bank,seed):
    r=np.load(APP/f'seed-{seed}'/'report-native-12.npz')
    indices=[]
    for w,t in zip(r['w'],r['threshold']):
        found=[j for j,(cw,ct) in enumerate(zip(bank['candidate_w'],bank['candidate_threshold'])) if np.array_equal(w,cw) and int(t)==int(ct)]
        if len(found)!=1:raise AssertionError('native12 candidate identity')
        indices.append(found[0])
    return indices

def final_certify(seed,method,selected,bank):
    a=np.c_[np.ones(len(bank['candidate_A']),dtype=int),bank['candidate_A'][:,selected]]
    nums=bank['response_numerator'];den=int(bank['response_denominator']);y=nums/den
    fits=[minimax_fit(a,y[:,k]) for k in range(10)]
    coefs=np.column_stack([np.rint(f['coef']*COEF_DEN).astype(np.int64) for f in fits])
    rows=[]
    for k,fit in enumerate(fits):
        cert=rational_witness(a,nums[:,k],den,fit['dual'])
        ix=cert['rows'];cert.update({'exact_design_rows':a[ix].tolist(),'exact_response_numerators':nums[ix,k].tolist()})
        residual=nums[:,k].astype(object)*COEF_DEN-(a.astype(object)@coefs[:,k].astype(object))*den
        upper=Q(max(abs(int(v)) for v in residual),den*COEF_DEN)
        lower=Q(cert['lower_bound'])
        if not 0<=upper-lower<Q(1,10**6):raise AssertionError(f'{seed}/{method}/{k}: loose bracket {upper-lower}')
        rows.append({'output':k,'numerical_minimax':fit['error'],'certificate':cert,'exact_upper':str(upper),'upper_minus_lower':str(upper-lower)})
    sd=P/f'seed-{seed}';sd.mkdir(exist_ok=True)
    np.savez_compressed(sd/f'{method}.npz',selected=np.array(selected),coefficient_numerator=coefs,coefficient_denominator=np.array(COEF_DEN),candidate_w=bank['candidate_w'][selected],candidate_threshold=bank['candidate_threshold'][selected],design=a,response_numerator=nums,response_denominator=np.array(den))
    summary={'seed':seed,'method':method,'state_bits':len(selected),'pair_bits':2*len(selected),'encoder_nonzero_weights':len(selected),'thresholds':len(selected),'decoder_coefficients':10*(len(selected)+1),'coefficient_denominator':COEF_DEN,'selected':selected,'outputs':rows,'joint_lower':str(max(Q(r['certificate']['lower_bound']) for r in rows)),'joint_upper':str(max(Q(r['exact_upper']) for r in rows))}
    if method=='contrast12_affine_minimax':
        orig=np.load(CERT/f'seed-{seed}'/'selected_decoder.npz')
        summary['original_frozen_coefficient_numerators_identical']=bool(np.array_equal(orig['coef_numerator'],coefs))
        summary['maximum_numerator_difference_from_original']=int(np.max(np.abs(orig['coef_numerator'].astype(object)-coefs.astype(object))))
    dump(sd/f'{method}.json',summary)
    return summary

def run():
    started=time.perf_counter()
    if (P/'freeze.json').exists():raise FileExistsError('Existing exploratory freeze')
    contract()
    all_results=[];traces=[]
    for seed in SEEDS:
        seedstart=time.perf_counter();bank=np.load(APP/f'seed-{seed}'/'discovery_bank.npz')
        cand=bank['candidate_A'];nums=bank['response_numerator'];den=int(bank['response_denominator']);base=list(range(8))
        a=np.c_[np.ones(len(cand),dtype=int),cand[:,base]]
        pilotstart=time.perf_counter();pilot,ng=compressed_fits(a,nums,den);pilot_seconds=time.perf_counter()-pilotstart
        # Independent full-row objective check once per seed.
        for k,pfit in enumerate(pilot):
            f=minimax_fit(a,nums[:,k]/den)
            if abs(f['error']-pfit['error'])>1e-7:raise AssertionError('compressed LP changed objective')
        print(json.dumps({'event':'pilot','seed':seed,'rows':len(a),'unique_rows':ng,'ten_coordinate_seconds':pilot_seconds,'predicted_full_greedy_seconds':pilot_seconds*sum(cand.shape[1]-8-i for i in range(4))}),flush=True)
        selected=base.copy();trace=[]
        for step in range(4):
            stepstart=time.perf_counter();records=[]
            for j in range(cand.shape[1]):
                if j in selected:continue
                a=np.c_[np.ones(len(cand),dtype=int),cand[:,selected+[j]]]
                t=time.perf_counter();fits,unique=compressed_fits(a,nums,den)
                errors=[f['error'] for f in fits];records.append({'candidate':j,'objective':max(errors),'coordinate_errors':errors,'unique_design_rows':unique,'seconds':time.perf_counter()-t})
                if len(records)%20==0:print(json.dumps({'event':'candidate_progress','seed':seed,'step':step+1,'evaluated':len(records),'remaining':cand.shape[1]-len(selected)-len(records),'elapsed_seconds':time.perf_counter()-stepstart}),flush=True)
            chosen=min(records,key=lambda r:(round(r['objective'],9),r['candidate']))
            selected.append(chosen['candidate'])
            rec={'seed':seed,'step':step+1,'selected':selected.copy(),'chosen_candidate':chosen['candidate'],'chosen_objective':chosen['objective'],'candidate_count':len(records),'seconds':time.perf_counter()-stepstart,'all_candidates':records}
            trace.append(rec);dump(P/f'greedy-seed-{seed}-step-{step+1}.json',rec)
            print(json.dumps({'event':'greedy_step','seed':seed,'step':step+1,'chosen':chosen['candidate'],'minimax':chosen['objective'],'candidates':len(records),'seconds':rec['seconds']}),flush=True)
        methods={'native8_affine_minimax':base,'native12_affine_minimax':native12_indices(bank,seed),'contrast12_affine_minimax':np.load(CERT/f'seed-{seed}'/'selected_decoder.npz')['selected'].tolist(),'greedy_minimax12':selected}
        for name,indices in methods.items():
            t=time.perf_counter();r=final_certify(seed,name,indices,bank);r['certification_seconds']=time.perf_counter()-t;all_results.append(r)
            print(json.dumps({'event':'final_certified','seed':seed,'method':name,'lower':r['joint_lower'],'upper':r['joint_upper']}),flush=True)
        traces.append({'seed':seed,'selected':selected,'candidate_lp_count':sum(r['candidate_count'] for r in trace)*10,'selection_seconds':sum(r['seconds'] for r in trace),'total_seed_seconds':time.perf_counter()-seedstart})
    discovery={'created_utc':utc(),'status':'post-test exploratory; discovery-only selection','results':all_results,'timing':traces,'total_seconds':time.perf_counter()-started}
    dump(P/'discovery_results.json',discovery)
    files=[P/'contract.json',P/'run.py',P/'discovery_results.json']+sorted(P.glob('seed-*/*'))+sorted(P.glob('greedy-*.json'))
    once(P/'freeze.json',{'created_utc':utc(),'status':'Exploratory new-analysis freeze only; original test outcomes were already known','files':{str(f.relative_to(P)):sha(f) for f in files}})
    print(json.dumps({'event':'frozen','total_seconds':time.perf_counter()-started}),flush=True)

def evaluate():
    if (P/'test_results.json').exists():raise FileExistsError('Exploratory test already evaluated')
    freeze=read(P/'freeze.json')
    for rel,h in freeze['files'].items():
        if sha(P/rel)!=h:raise AssertionError('Changed frozen file: '+rel)
    results=[]
    for seed in SEEDS:
        for name in ['seen_strength','unseen_strength','natural_pairs']:
            b=np.load(APP/f'seed-{seed}'/f'test_bank_{name}.npz');yn=b['response_numerator'];den=int(b['response_denominator']);y=yn/den
            for path in sorted((P/f'seed-{seed}').glob('*.npz')):
                r=np.load(path);a=np.c_[np.ones(len(yn),dtype=int),b['candidate_A'][:,r['selected']]]
                # Exact numerator of dyadic prediction minus rational target.
                en=(a.astype(object)@r['coefficient_numerator'].astype(object))*den-yn.astype(object)*COEF_DEN
                ed=den*COEF_DEN;err=np.asarray(en,dtype=float)/ed
                mx=max(abs(int(v)) for v in en.flat)
                results.append({'seed':seed,'bank':name,'method':path.stem,'n':len(y),'mse_sum_outputs':float(np.mean(np.sum(err*err,axis=1))),'maximum_absolute_error':float(Q(mx,ed)),'exact_max_error':str(Q(mx,ed)),'fraction_all_outputs_within_1':float(np.mean(np.max(np.abs(en),axis=1)<=ed)),'mse_relative_zero_response':float(np.mean(np.sum(err*err,axis=1))/np.mean(np.sum(y*y,axis=1))),'test_bank_sha256':sha(APP/f'seed-{seed}'/f'test_bank_{name}.npz')})
    once(P/'test_results.json',{'evaluated_utc':utc(),'status':'POST-TEST EXPLORATORY; same previously opened banks. No claim of a new protected test.','freeze_sha256':sha(P/'freeze.json'),'results':results})
    print(json.dumps({'event':'evaluated','comparisons':len(results)}),flush=True)

if __name__=='__main__':
    import warnings
    warnings.filterwarnings('ignore',message='Unrecognized options detected:.*threads')
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['run','evaluate']);args=parser.parse_args()
    {'run':run,'evaluate':evaluate}[args.phase]()
