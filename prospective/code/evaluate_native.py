"""Frozen new-bank evaluator: fixed17-unit labels, no fitting or rematching."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, platform, sys
import numpy as np
import torch
HERE=Path(__file__).resolve().parent.parent
BANK=HERE/'bank/bank.npz'
UNITS=[10,41,55,77,81,185,224,245,261,291,317,326,380,411,416,492,495]
LABELS=[16,0,28,7,14,3,1,9,27,23,17,30,19,25,28,21,2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise RuntimeError(message)
def truth(tokens):
    result=np.ones((len(tokens),32),dtype=bool)
    for a in range(32):
        bits=np.array([bool((a>>(4-i))&1) for i in range(5)])
        literals=np.stack((tokens[:,1:40:4],tokens[:,2:40:4]),axis=2)
        values=bits[literals%5] ^ (literals>=5)
        result[:,a]=values.any(axis=2).all(axis=1)
    return result
def infer(model,tokens):
    final=model.blocks[-1];points={'pre':final.mlp.hook_pre,'post':final.mlp.hook_post,'resid_mid':final.hook_resid_mid,'resid_post':final.hook_resid_post}
    result={};counts={}
    def capture(key):
        def hook(tensor,**kwargs):
            counts[key]=counts.get(key,0)+1;result[key]=tensor[:,-1,:].detach().cpu().numpy().copy()
        return hook
    for key,point in points.items():point.add_hook(capture(key))
    try:
        with torch.inference_mode():result['logits']=model(torch.from_numpy(tokens.copy()))[:,-1,:].detach().cpu().numpy().copy()
    finally:model.remove_all_hooks()
    require(counts==dict.fromkeys(points,1),'hook counts')
    for key,d in [('pre',512),('post',512),('resid_mid',128),('resid_post',128),('logits',16)]:
        require(result[key].shape==(len(tokens),d) and np.isfinite(result[key]).all(),'array shape/finite '+key)
    require(np.array_equal(result['post'],np.maximum(result['pre'],0)),'ReLU identity')
    return result
def auc(x,y):
    require(y.any() and (~y).any(),'AUROC requires both classes')
    pos=x[y];neg=x[~y]; total=0.0
    for start in range(0,len(pos),128):
        delta=pos[start:start+128,None]-neg
        total+=(delta>0).sum()+0.5*(delta==0).sum()
    return float(total/(len(pos)*len(neg)))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--release-root',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();ROOT=Path(args.release_root).resolve();output=Path(args.out).resolve()
    require(sha(BANK)=='f2d30d0f09dab11c6e06859f4e8b1630a652eb77a5e1d7bfc02e9e3706bd0e13','bank hash')
    require(not output.exists(),'refuse overwrite')
    freeze_path=HERE/'observed/receipt.json'
    with np.load(BANK,allow_pickle=False) as z:tokens=z['tokens'].copy();stored_truth=z['truth'].copy()
    require(tokens.shape==(4096,42) and np.issubdtype(tokens.dtype,np.integer),'token schema')
    require(np.all(tokens[:,0:40:4]==10) and np.all(tokens[:,3:40:4]==11) and np.all(tokens[:,40]==12),'token grammar')
    literals=np.stack((tokens[:,1:40:4],tokens[:,2:40:4]),axis=2)
    require(np.all((literals>=0)&(literals<10)) and np.all(literals[:,:,0]%5!=literals[:,:,1]%5),'literal grammar')
    keys=[tuple(sorted(tuple(sorted(map(int,c))) for c in row)) for row in literals]
    require(len(set(keys))==4096 and all(len(set(k))==10 for k in keys),'distinct clauses/formulas')
    targets=truth(tokens[:,:41]);reverse=[int(format(a,'05b')[::-1],2) for a in range(32)]
    require(np.array_equal(targets,stored_truth[:,reverse]),'MSB/LSB semantic bridge')
    task=np.where(targets.any(1),13,14);require(np.array_equal(task,tokens[:,41]),'task token labels')
    require(int((task==13).sum())==2048,'balanced classes')
    spec=importlib.util.spec_from_file_location('frozen_published_runtime',ROOT/'work/sat_case/runtime.py')
    runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
    model,config,model_receipt=runtime.load_model(threads=2)
    output.mkdir()
    (output/'forward-started.json').write_text(json.dumps({'source_receipt_sha256':sha(freeze_path),'bank_sha256':sha(BANK),'model':model_receipt},indent=2))
    blocks=[]
    for start in range(0,len(tokens),256):blocks.append(infer(model,tokens[start:start+256,:41]))
    arrays={key:np.concatenate([block[key] for block in blocks]) for key in blocks[0]}
    replay=infer(model,tokens[:256,:41])
    require(all(np.array_equal(arrays[key][:256],replay[key]) for key in arrays),'batch replay mismatch')
    with (output/'native-outputs.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    rows=[];group={};solution_count=targets.sum(1)
    for unit,label in zip(UNITS,LABELS):
        y=targets[:,label];score=arrays['pre'][:,unit];detected=arrays['post'][:,unit]>=0.5
        value=auc(score,y);group.setdefault(label,[]).append(value)
        cells={}
        for name,mask in [('sole_positive',(solution_count==1)&y),('multi_positive',(solution_count>=2)&y),('unsat_negative',solution_count==0),('sole_negative',(solution_count==1)&~y),('multi_negative',(solution_count>=2)&~y)]:
            cells[name]={'detected':int((detected&mask).sum()),'denominator':int(mask.sum())}
        rows.append({'unit':unit,'assignment_id_msb_x0':label,'auroc':value,'positive_detected':int((detected&y).sum()),'positive_total':int(y.sum()),'negative_detected':int((detected&~y).sum()),'negative_total':int((~y).sum()),'threshold_cells':cells})
    pred=arrays['logits'].argmax(1);correct=pred==task
    record={'status':'PASS','scope':'balanced class-conditional distinct-clause source-grammar bank, same checkpoint; historical exclusions, not natural prevalence or independent checkpoint replication','bank_sha256':sha(BANK),'source_receipt_sha256':sha(freeze_path),'outputs_sha256':sha(output/'native-outputs.npz'),'rows':4096,'batches':16,'replayed_rows':256,'batch_replay_bitwise_equal':True,'model':model_receipt,'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__},'assignment_macro_auroc':float(np.mean([np.mean(v) for v in group.values()])),'unit_min_auroc':min(r['auroc'] for r in rows),'per_unit':rows,'task':{'correct':int(correct.sum()),'total':4096,'non_answer_argmax':int((~np.isin(pred,[13,14])).sum()),'sat':{'correct':int((correct&(task==13)).sum()),'total':2048},'unsat':{'correct':int((correct&(task==14)).sum()),'total':2048}},'fitted_truth_classifier_endpoint':'not run: no persisted fitted models, and no new fit authorized by this evaluator','history_scope':'25 packaged token arrays and additional whole-formula clause scans; unavailable training/test corpus not checked; no signed-permutation exclusion'}
    with (output/'receipt.json').open('x') as f:json.dump(record,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in record.items() if k not in ('per_unit','model')}))
if __name__=='__main__':main()
