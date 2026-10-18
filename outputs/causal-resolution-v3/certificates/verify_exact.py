"""Independent standard-library checker for retained contrast certificates.

Usage: python verify_exact.py PATH/TO/certificates
No NumPy, SciPy, model framework, or optimisation solver is imported.
"""
from fractions import Fraction as Q
from pathlib import Path
import gzip, hashlib, json, sys

EXPECTED_FEATURES={17:89,29:94,43:87}
EXPECTED_FULL_ROWS=1904
EXPECTED_OUTPUTS=10


def check(c):
    a=c['exact_design_rows']; y=c['exact_response_numerators']
    u=[Q(v) for v in c['weights']]; den=c['response_denominator']
    ids=c['rows']
    if type(den) is not int or den<=0 or not a or len(a)!=len(y) or len(a)!=len(u):
        raise ValueError('incompatible dimensions')
    if (len(ids)!=len(a) or c['support_size']!=len(a) or
        any(type(i) is not int or i<0 for i in ids) or len(set(ids))!=len(ids)):
        raise ValueError('incorrect support dimensions or row identities')
    if sum(abs(t) for t in u)!=1:
        raise ValueError('incorrect normalisation')
    width=len(a[0])
    if any(len(r)!=width for r in a):raise ValueError('ragged matrix')
    if any(type(x) is not int for row in a for x in row):raise ValueError('noninteger design')
    if any(type(x) is not int for x in y):raise ValueError('noninteger target')
    for j in range(width):
        if sum(t*row[j] for t,row in zip(u,a)):
            raise ValueError('nonzero annihilation residual')
    bound=sum(t*Q(v,den) for t,v in zip(u,y))
    if bound!=Q(c['lower_bound']) or bound<0:raise ValueError('incorrect lower bound')
    return bound


def check_full_library(claim,p):
    """Check complete shapes before arithmetic, so no zip can truncate evidence."""
    seed=claim['seed']
    if seed not in EXPECTED_FEATURES or p['seed']!=seed:
        raise ValueError('primal/dual seed mismatch')
    if claim['feature_count']!=EXPECTED_FEATURES[seed]:
        raise ValueError('unexpected full-library feature count')
    width=EXPECTED_FEATURES[seed]+1
    a=p['design'];ys=p['response_numerators'];betas=p['coefficient_numerators']
    uppers=p['upper_bounds'];d=p['response_denominator'];q=p['coefficient_denominator']
    if (type(d) is not int or type(q) is not int or d<=0 or q<=0 or
        len(a)!=EXPECTED_FULL_ROWS or len(ys)!=len(a)):
        raise ValueError('incorrect full-library row dimensions or denominators')
    if any(len(row)!=width or any(type(v) is not int for v in row) for row in a):
        raise ValueError('incorrect full-library design dimensions or type')
    if not all(row[0]==1 for row in a):
        raise ValueError('full-library affine intercept missing')
    if any(len(row)!=EXPECTED_OUTPUTS or any(type(v) is not int for v in row) for row in ys):
        raise ValueError('incorrect full-library target dimensions or type')
    if len(betas)!=EXPECTED_OUTPUTS or len(uppers)!=EXPECTED_OUTPUTS:
        raise ValueError('incorrect full-library output or upper-bound count')
    if any(len(beta)!=width or any(type(v) is not int for v in beta) for beta in betas):
        raise ValueError('incorrect full-library coefficient dimensions or type')
    k=claim['output'];c=claim['certificate'];lower=check(c)
    if type(k) is not int or not 0<=k<EXPECTED_OUTPUTS:
        raise ValueError('invalid full-library witness output')
    if c['response_denominator']!=d or any(i>=len(a) for i in c['rows']):
        raise ValueError('full-library witness not linked to primal response bank')
    if (c['exact_design_rows']!=[a[i] for i in c['rows']] or
        c['exact_response_numerators']!=[ys[i][k] for i in c['rows']]):
        raise ValueError('full-library witness rows disagree with complete primal bank')
    if lower<=1:
        raise ValueError('full-library witness does not reject primary tolerance')
    evaluated=[]
    for output in range(EXPECTED_OUTPUTS):
        beta=betas[output]
        mx=max(abs(ys[i][output]*q-d*sum(a[i][j]*beta[j] for j in range(width)))
               for i in range(len(a)))
        upper=Q(mx,d*q)
        if upper!=Q(uppers[output]):raise ValueError('primal residual mismatch')
        evaluated.append(upper)
    if max(evaluated)!=Q(claim['exact_upper']):raise ValueError('joint upper mismatch')
    if not 0<=max(evaluated)-lower<Q(1,10**7):
        raise ValueError('claimed tight bracket not established')
    return len(evaluated)


def records_by_seed(records,label):
    if not isinstance(records,list) or len(records)!=len(EXPECTED_FEATURES):
        raise ValueError('incorrect '+label+' record count')
    seeds=[record['seed'] for record in records]
    if any(type(seed) is not int for seed in seeds) or set(seeds)!=set(EXPECTED_FEATURES):
        raise ValueError('incorrect or duplicate '+label+' seeds')
    return {record['seed']:record for record in records}


def main(root):
    counts={'certificates':0,'injective_report_certificates':0,'rejected_mutations':0,
            'full_library_certificates':0,'full_library_primal_output_bounds':0}
    bounds=[]
    discovery_files=sorted(root.glob('seed-*/discovery_audit.json'))
    if {p.parent.name for p in discovery_files}!={f'seed-{seed}' for seed in EXPECTED_FEATURES}:
        raise ValueError('incorrect discovery seed files')
    for p in discovery_files:
        d=json.loads(p.read_text())
        if p.parent.name!=f"seed-{d['seed']}" or [row['step'] for row in d['rounds']]!=list(range(5)):
            raise ValueError('incorrect discovery seed or round count')
        for row in d['rounds']:
            c=row['certificate'];check(c);counts['certificates']+=1
            corrupted=json.loads(json.dumps(c));corrupted['exact_response_numerators'][0]+=1
            try:check(corrupted)
            except ValueError:counts['rejected_mutations']+=1
            else:raise ValueError('response mutation escaped detection')
        c=d['pure_expressivity']['certificate'];bound=check(c)
        if len(set(map(tuple,c['exact_design_rows'])))!=len(c['exact_design_rows']):
            raise ValueError('claimed injective subbank has duplicate report rows')
        if not all(row[0]==1 for row in c['exact_design_rows']):
            raise ValueError('claimed affine certificate has no intercept')
        if c['support_size']<3 or bound<=1:
            raise ValueError('claimed pure expressivity rejection not established')
        counts['certificates']+=1;counts['injective_report_certificates']+=1
        bounds.append({'seed':d['seed'],'bound':str(bound),'support':c['support_size']})
    if counts['certificates']!=18 or counts['rejected_mutations']!=15 or counts['injective_report_certificates']!=3:
        raise ValueError('missing expected discovery certificates')
    full=records_by_seed(json.loads((root/'full_library_certificates.json').read_text()),'full-library dual')
    with gzip.open(root/'full_library_primal.json.gz','rt') as f:
        primals=records_by_seed(json.load(f),'full-library primal')
    for seed in sorted(EXPECTED_FEATURES):
        counts['full_library_primal_output_bounds']+=check_full_library(full[seed],primals[seed])
        counts['certificates']+=1;counts['full_library_certificates']+=1
    if (counts['certificates']!=21 or counts['full_library_certificates']!=3 or
        counts['full_library_primal_output_bounds']!=30):
        raise ValueError('missing expected full-library certificates or output bounds')
    files=discovery_files+[root/'full_library_certificates.json',root/'full_library_primal.json.gz']
    result={'status':'PASS','method':'independent Fraction dot products; standard library only',
            **counts,'injective_bounds':bounds,
            'input_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'validates saved exact witnesses and complete full-library primal banks; does not independently reconstruct model training or responses from model weights, or establish semantics or priority'}
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':main(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent)
