"""Pre-test supplementary contrast audit. Selection uses discovery bank only."""
from pathlib import Path
from fractions import Fraction
import hashlib, json, time
import numpy as np
from affine_audit import minimax_fit, rational_witness, verify_witness

ROOT=Path(__file__).resolve().parent
APP=ROOT.parent/'application'


def exact_upper(a, nums, response_denom, coef):
    """Dyadic coefficient export and exact finite-bank residual upper bound."""
    q=2**32
    beta=[int(round(float(v)*q)) for v in coef]
    maxnum=max(abs(int(y)*q-sum(int(x)*b for x,b in zip(row,beta))*int(response_denom))
               for row,y in zip(a,nums))
    return beta,q,Fraction(maxnum,int(response_denom)*q)


def dump_cert(a,nums,den,cert):
    ids=cert['rows']
    return dict(cert, exact_design_rows=np.asarray(a)[ids].tolist(),
                exact_response_numerators=np.asarray(nums)[ids].tolist())


def run():
    # This timestamp records a local prospective freeze; it is not preregistration.
    contract={
      'version':1,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
      'scope':'Supplementary after discovery; before root accessed any test outcomes',
      'seeds':[17,29,43], 'base':'native8; allow affine intercept',
      'selection':'4 greedy additions. Each round choose worst minimax output, then maximum absolute exact dual contrast per new predicate. Lowest-index exact tie.',
      'output_decoder':'coordinatewise affine minimax; dyadic coefficient export denominator 2^32',
      'pure_expressivity_audit':'one first row per distinct base design; intercept included; worst-output sparse exact contrast',
      'optimality':'No global optimality claim for greedy feature selection. LP lower/upper bounds separately certified.',
      'test_rule':'Read frozen test bank only after selection files and hashes have been written.'}
    (ROOT/'supplementary_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    results=[]
    for seed in contract['seeds']:
        bank=APP/f'seed-{seed}'/'discovery_bank.npz'
        z=np.load(bank)
        base=z['A'].astype(int); cand=z['candidate_A'].astype(int)
        nums=z['response_numerator']; den=int(z['response_denominator'])
        y=nums/den; selected=list(range(base.shape[1])); rounds=[]
        for step in range(5):
            a=np.c_[np.ones(len(base),dtype=int), cand[:,selected]]
            fits=[minimax_fit(a,y[:,k]) for k in range(y.shape[1])]
            k=max(range(len(fits)),key=lambda j:(fits[j]['error'],-j))
            fit=fits[k]
            cert=rational_witness(a,nums[:,k],den,fit['dual'])
            bn,bden,upper=exact_upper(a,nums[:,k],den,fit['coef'])
            row={'step':step,'selected':selected.copy(),'worst_output':k,
                 'minimax_errors':[v['error'] for v in fits],
                 'certificate':dump_cert(a,nums[:,k],den,cert),
                 'upper_bound':str(upper),'upper_bound_float':float(upper),
                 'upper_coefficients_numerator':bn,'upper_coefficients_denominator':bden}
            rounds.append(row)
            if step==4:break
            ids=cert['rows']; u=[Fraction(t) for t in cert['weights']]
            scores={j:abs(sum(w*int(cand[i,j]) for i,w in zip(ids,u)))
                    for j in range(cand.shape[1]) if j not in selected}
            chosen=max(scores,key=lambda j:(scores[j],-j))
            row['added_candidate']=chosen; row['contrast_score']=str(scores[chosen])
            if not scores[chosen]:
                row['stopped']='No remaining candidate detects this witness'
                break
            selected.append(chosen)
        # Injective design subbank: lookup error is exactly zero on these rows.
        seen=set(); ids=[]
        for i,row in enumerate(base):
            key=tuple(row.tolist())
            if key not in seen:seen.add(key); ids.append(i)
        ua=np.c_[np.ones(len(ids),dtype=int),base[ids]]
        un=nums[ids]
        uf=[minimax_fit(ua,un[:,k]/den) for k in range(un.shape[1])]
        uk=max(range(len(uf)),key=lambda j:(uf[j]['error'],-j))
        uc=rational_witness(ua,un[:,uk],den,uf[uk]['dual'])
        pure={'subbank_rows':len(ids),'output':uk,'lookup_error_exact':0,
              'original_bank_row_ids':[ids[i] for i in uc['rows']],
              'certificate':dump_cert(ua,un[:,uk],den,uc)}
        dec=np.column_stack([f['coef'] for f in fits])
        dec_nums=np.rint(dec*(2**32)).astype(np.int64)
        out=ROOT/f'seed-{seed}'
        out.mkdir(exist_ok=True)
        np.savez_compressed(out/'selected_decoder.npz',selected=np.array(selected),
                            coef_numerator=dec_nums,coef_denominator=np.array(2**32),
                            candidate_w=z['candidate_w'][selected],
                            candidate_threshold=z['candidate_threshold'][selected])
        result={'seed':seed,'discovery_bank_sha256':hashlib.sha256(bank.read_bytes()).hexdigest(),
                'rounds':rounds,'pure_expressivity':pure,'selected':selected,
                'selected_decoder_sha256':hashlib.sha256((out/'selected_decoder.npz').read_bytes()).hexdigest()}
        (out/'discovery_audit.json').write_text(json.dumps(result,indent=2)+'\n')
        results.append(result)
        print(json.dumps({'seed':seed,'initial_minimax':rounds[0]['upper_bound_float'],
          'repaired_minimax':rounds[-1]['upper_bound_float'],'pure_contrast':uc['lower_bound'],
          'pure_support':uc['support_size'],'selected':selected}),flush=True)
    (ROOT/'discovery_summary.json').write_text(json.dumps(results,indent=2)+'\n')
    files=[ROOT/'supplementary_contract.json',ROOT/'discovery_audit.py',ROOT/'affine_audit.py']
    files += sorted(ROOT.glob('seed-*/*'))
    freeze={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (ROOT/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')


if __name__=='__main__':run()
