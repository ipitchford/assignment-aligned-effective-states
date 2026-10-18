"""Independent discovery-only binding of nonlinear certificates to exported models.

Does not import nonlinear.py or its target computation, read test files,
train a model, or select any report features.
"""
from pathlib import Path
import hashlib,json
import numpy as np

HERE=Path(__file__).resolve().parent
APP=HERE.parent/'nonlinear_application'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    fitpath=APP/'discovery_fit.npz';contractpath=APP/'contract.json'
    fit=np.load(fitpath,allow_pickle=False)
    contract=json.loads(contractpath.read_text());groups=contract['groups']
    freeze=json.loads((APP/'freeze.json').read_text())
    if sha(fitpath)!=contract['files']['discovery_fit.npz'] or sha(contractpath)!=freeze['files']['contract.json']:
        raise ValueError('discovery split or contract freeze mismatch')
    lookup={int(i):j for j,i in enumerate(fit['ids'])}
    inputs={str(p.relative_to(APP)):sha(p) for p in [fitpath,contractpath]};results=[]
    for seed in [117,129,143]:
        sd=APP/f'seed-{seed}'
        for name in ['model.npz','report-native-8.npz','discovery_bank.npz']:
            p=sd/name;relative=str(p.relative_to(APP));inputs[relative]=sha(p)
            if inputs[relative]!=freeze['files'][relative]:
                raise ValueError('discovery input freeze mismatch: '+relative)
        model=np.load(sd/'model.npz',allow_pickle=False)
        w1,b1,w2,b2,w3,b3=[model[k].astype(object) for k in ['w1','b1','w2','b2','w3','b3']]
        h=np.maximum(fit['x'].astype(object)@w1+16*b1,0)
        bank=np.load(sd/'discovery_bank.npz',allow_pickle=False)
        before=[];after=[]
        for image,g,s in bank['meta']:
            if int(s) not in [2,4]:raise ValueError('unexpected discovery intervention strength')
            v=h[lookup[int(image)]];changed=4*v.copy();changed[groups[int(g)]]=(4-int(s))*v[groups[int(g)]]
            before.append(4*v);after.append(changed)
        before=np.array(before,dtype=object);after=np.array(after,dtype=object)
        second0=np.maximum(before@w2+4*65536*b2,0)
        second1=np.maximum(after@w2+4*65536*b2,0)
        logits0=second0@w3+4*65536*4096*b3
        logits1=second1@w3+4*65536*4096*b3
        nums=(10*logits1-logits1.sum(axis=1,keepdims=True))-(10*logits0-logits0.sum(axis=1,keepdims=True))
        if not np.array_equal(nums,bank['response_numerator']) or int(bank['response_denominator'])!=43980465111040:
            raise ValueError('model response mismatch')
        enc=np.load(sd/'report-native-8.npz',allow_pickle=False)
        w,t=enc['w'].astype(object),enc['threshold'].astype(object)
        A=(after@w.T>4*t).astype(int)-(before@w.T>4*t).astype(int)
        if not np.array_equal(A,bank['A']):raise ValueError('model native8 report mismatch')
        seen=set();ids=[]
        for i,row in enumerate(A):
            key=tuple(row)
            if key not in seen:seen.add(key);ids.append(i)
        record=json.loads((HERE/f'seed-{seed}.json').read_text());primal=record['primal']
        design=np.column_stack([np.ones(len(ids),dtype=int),A[ids]])
        if ids!=record['subbank_original_row_ids'] or design.tolist()!=primal['design'] or nums[ids].tolist()!=primal['response_numerators']:
            raise ValueError('self-contained certificate subbank mismatch')
        if record['source_bank_sha256']!=sha(sd/'discovery_bank.npz'):
            raise ValueError('source-bank hash mismatch')
        witness_ids=record['witness_original_bank_row_ids']
        if bank['meta'][witness_ids].tolist()!=record['witness_metadata']:
            raise ValueError('witness metadata mismatch')
        results.append({'seed':seed,'rows_replayed':len(nums),'subbank_rows':len(ids),'targets_and_native_reports':'all exact matches'})
    result={'status':'PASS','arithmetic':'Python arbitrary-precision endpoint logits and native predicate evaluation',
            'test_outcomes_accessed':False,'models_trained':False,'features_selected':False,
            'freeze_sha256':sha(APP/'freeze.json'),
            'input_sha256':inputs,'verifier_sha256':sha(Path(__file__)),
            'results':results,'scope':'exported model plus discovery-image linkage; not independent retraining or scientific replication'}
    (HERE/'model_replay.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'PASS','rows_replayed':sum(r['rows_replayed'] for r in results),'seeds':[r['seed'] for r in results]}))

if __name__=='__main__':main()
