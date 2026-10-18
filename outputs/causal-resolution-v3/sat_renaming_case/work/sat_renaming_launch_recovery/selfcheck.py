"""Deterministic non-formula arithmetic checks. No RNG, checkpoint, fitting or native inference."""
import argparse
import ast
from itertools import product
import analyze as a
from protocol import *


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (ROOT / args.output).exists(): raise RuntimeError("Refusing to overwrite self-check receipt")
    count = 0
    def check(condition, name):
        nonlocal count
        if not condition: raise AssertionError(name)
        count += 1
    for path in sorted(ROOT.glob("*.py")):
        ast.parse(path.read_text(), filename=str(path)); check(True, "AST "+path.name)
    cases = [([1,0,0,0],[0,1,0,0],[0,.5,.5]),([0,0,0,0],[0,0,0,0],[0,0,0]),([1,0,0,0],[0,0,0,1],[.5,.5,0])]
    for source,target,expected in cases:
        bits=np.array([source,target],dtype=np.int8)
        errors, contrasts, primary=a.endpoint_statistics(bits,bits,bits,np.array([0,1]))
        check(np.array_equal(errors[:,0],expected),"mapping orientation")
        check(np.array_equal(errors[:,0],errors[:,1]) and np.array_equal(errors[:,0],errors[:,2]),"singleton bounds")
        check(np.array_equal(contrasts[:,0],np.array(expected[1:])-expected[0]),"shared contrast")
        values=np.array([[0,0,0,0],source,[0,0,0,0],target],dtype=np.int8)
        changes,_=a.change_statistics(values,values,values,np.array([0,1]))
        check(np.array_equal(changes[:,0],expected),"signed change orientation")
    nominal=np.zeros((2,4),dtype=np.int8); lo=nominal.copy(); hi=np.ones_like(lo)
    errors,contrasts,_=a.endpoint_statistics(nominal,lo,hi,np.array([0,1]))
    brute=[]
    perms=[(1,0,3,2),(0,1,2,3),(3,2,1,0)]
    for values in product([0,1],repeat=8):
        x,y=values[:4],values[4:]
        e=[sum(y[j]!=x[p[j]] for j in range(4))/4 for p in perms]
        brute.append(e+[e[1]-e[0],e[2]-e[0]])
    brute=np.array(brute)
    check(np.array_equal(errors[:,1],brute[:,:3].min(axis=0)),"all-endpoint error lower")
    check(np.array_equal(errors[:,2],brute[:,:3].max(axis=0)),"all-endpoint error upper")
    check(np.array_equal(contrasts[:,1],brute[:,3:].min(axis=0)),"joint contrast lower")
    check(np.array_equal(contrasts[:,2],brute[:,3:].max(axis=0)),"joint contrast upper")
    signed,_=a.change_statistics(np.zeros((4,4),dtype=np.int8),np.zeros((4,4),dtype=np.int8),np.ones((4,4),dtype=np.int8),np.array([0,1]))
    check(np.array_equal(signed[:,1],[0,0,0]) and np.array_equal(signed[:,2],[2,2,2]),"signed range retains negative changes")
    b,l,u=a.bit_bounds(np.array([0.0,0.5,0.6]))
    check(np.array_equal(b,[0,1,1]) and np.array_equal(l,[0,0,1]) and np.array_equal(u,[0,1,1]),"inclusive source threshold")
    good=a.correctness(b,l,u,np.array([0,1,0]))
    check(np.array_equal(good,[[1,1,1],[1,0,1],[0,0,0]]),"semantic correctness intervals")
    tn=np.array([14,13,15]); fp=np.array([16-tn[0],16-tn[2],16-tn[1]])
    check(np.array_equal(fp,[2,1,3]),"FP inverse correctness bounds")
    check(a.classify_conditions({"tp":a.minimum_condition([58,57,58],58)})=="UNRESOLVED_NUMERICAL_SENSITIVITY","minimum ambiguous")
    check(a.classify_conditions({"fp":a.maximum_condition([51,50,52],51)})=="UNRESOLVED_NUMERICAL_SENSITIVITY","maximum adverse bound")
    check(a.classify_conditions({"fp":a.maximum_condition([52,52,53],51)})=="CRITERION_NOT_MET","maximum favourable failure")
    check(a.classify_conditions({"mapping":a.minimum_condition([.2,0,.3],0,True)})=="UNRESOLVED_NUMERICAL_SENSITIVITY","strict positive mapping")
    logits=np.zeros((3,16)); logits[0,13]=1; logits[1,13]=logits[1,14]=1; logits[2,0]=1
    pred,possible,correct=a.task_bounds(logits,np.array([13,13,13]))
    check(np.array_equal(pred,[13,13,0]),"full vocabulary argmax")
    check(np.array_equal(possible.sum(axis=-1),[1,2,1]),"task tie overincluded")
    check(np.array_equal(correct,[[1,1,1],[1,0,1],[0,0,0]]),"task guaranteed and possible correctness")
    fixed=np.array([.1,.4,.5,.9]); q=np.quantile(fixed,[.025,.975],method="linear")
    check(np.all(np.quantile(fixed-.1,[.025,.975],method="linear")<=q) and np.all(q<=np.quantile(fixed+.1,[.025,.975],method="linear")),"linear quantile bound monotonicity")
    save_json(args.output,{"status":"PASS","checks":count,"utc":utc(),"scope":"Deterministic non-formula bit/logit arithmetic and AST syntax checks only",
        "new_formulas":0,"native_forwards":0,"fits":0,"scientific_bootstrap_draws":0,
        "tested_hashes":{name:sha(ROOT/name) for name in ["analyze.py","protocol.py","selfcheck.py"]},"numpy":np.__version__})
    print(json.dumps(read_json(args.output)))


if __name__ == "__main__":main()
