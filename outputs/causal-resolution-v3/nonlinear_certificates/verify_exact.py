"""Independent standard-library replay of the self-contained nonlinear records."""
from pathlib import Path
from fractions import Fraction as Q
import hashlib,json

ROOT=Path(__file__).resolve().parent
SEEDS=(117,129,143)

def check(record):
    seed=record['seed'];p=record['primal'];c=record['certificate']
    a=p['design'];ys=p['response_numerators'];bs=p['coefficient_numerators']
    den=p['response_denominator'];q=p['coefficient_denominator'];n=record['subbank_rows']
    ids=record['subbank_original_row_ids'];k=record['output']
    if seed not in SEEDS or n!=len(a) or len(ys)!=n or len(ids)!=n or len(set(ids))!=n:
        raise ValueError('incorrect subbank dimensions or identifiers')
    if (any(type(i) is not int or not 0<=i<1904 for i in ids) or
        type(den) is not int or den!=43980465111040 or type(q) is not int or q<=0):
        raise ValueError('invalid bank identities or denominators')
    if any(len(row)!=9 or any(type(v) is not int for v in row) or row[0]!=1 for row in a):
        raise ValueError('invalid affine design')
    if len(set(map(tuple,a)))!=n or record['lookup_error_exact']!=0:
        raise ValueError('lookup injectivity not established')
    if any(len(row)!=10 or any(type(v) is not int for v in row) for row in ys):
        raise ValueError('invalid response bank')
    if len(bs)!=10 or len(p['upper_bounds'])!=10 or any(len(b)!=9 or any(type(v) is not int for v in b) for b in bs):
        raise ValueError('invalid primal coefficient or output dimensions')
    if type(k) is not int or not 0<=k<10:
        raise ValueError('invalid witness output')
    rows=c['rows'];u=[Q(v) for v in c['weights']]
    if len(rows)!=len(u) or len(rows)!=c['support_size'] or len(set(rows))!=len(rows) or any(type(i) is not int or not 0<=i<n for i in rows):
        raise ValueError('invalid witness support')
    if c['exact_design_rows']!=[a[i] for i in rows] or c['exact_response_numerators']!=[ys[i][k] for i in rows] or c['response_denominator']!=den:
        raise ValueError('witness not bound to complete subbank')
    if record['witness_original_bank_row_ids']!=[ids[i] for i in rows]:
        raise ValueError('incorrect original witness row mapping')
    if sum(abs(v) for v in u)!=1 or any(sum(u[t]*a[i][j] for t,i in enumerate(rows)) for j in range(9)):
        raise ValueError('invalid exact witness normalisation or annihilation')
    lower=sum(u[t]*Q(ys[i][k],den) for t,i in enumerate(rows))
    if lower!=Q(c['lower_bound']) or lower!=Q(record['joint_lower_bound']):
        raise ValueError('incorrect exact lower bound')
    uppers=[]
    for output in range(10):
        err=max(abs(ys[i][output]*q-den*sum(a[i][j]*bs[output][j] for j in range(9))) for i in range(n))
        upper=Q(err,den*q)
        if upper!=Q(p['upper_bounds'][output]):raise ValueError('incorrect exact primal upper')
        uppers.append(upper)
    if max(uppers)!=Q(record['joint_upper_bound']) or not 0<=max(uppers)-lower<Q(1,10**7):
        raise ValueError('incorrect joint minimax bracket')
    if record['rejects_primary_tolerance']!=(lower>Q(record['primary_tolerance'])):
        raise ValueError('incorrect primary rejection decision')
    return {'seed':seed,'subbank_rows':n,'support':len(rows),'lower':str(lower),'upper':str(max(uppers)),'rejects_epsilon1':lower>1}

def main():
    paths=sorted(ROOT.glob('seed-*.json'))
    if {p.name for p in paths}!={f'seed-{seed}.json' for seed in SEEDS}:
        raise ValueError('unexpected certificate file set')
    verified=[];rejected=0
    for path in paths:
        r=json.loads(path.read_text())
        if path.name!=f"seed-{r['seed']}.json":
            raise ValueError('certificate seed does not match its filename')
        verified.append(check(r))
        for kind in ['response','coefficient_truncation','target_truncation','upper_count','mapping']:
            bad=json.loads(json.dumps(r))
            if kind=='response':bad['certificate']['exact_response_numerators'][0]+=1
            elif kind=='coefficient_truncation':bad['primal']['coefficient_numerators'][0].pop()
            elif kind=='target_truncation':bad['primal']['response_numerators'].pop()
            elif kind=='upper_count':bad['primal']['upper_bounds'].pop()
            else:bad['witness_original_bank_row_ids'][0]+=1
            try:check(bad)
            except ValueError:rejected+=1
            else:raise ValueError('negative control escaped: '+kind)
    result={'status':'PASS','method':'independent standard-library exact fraction and integer arithmetic',
            'certificates':len(verified),'primal_output_bounds':10*len(verified),'rejected_mutations':rejected,
            'verified':verified,'test_outcomes_accessed':False,
            'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'verifies retained finite tables and exact bounds; replay_model.py separately binds tables to exported model and discovery images'}
    (ROOT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
