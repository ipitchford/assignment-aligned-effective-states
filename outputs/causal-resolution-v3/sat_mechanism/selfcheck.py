"""Pre-extraction synthetic tests; never loads a neural checkpoint or pilot data."""
from fractions import Fraction
from itertools import product
import json
import numpy as np
import analyze
from protocol import *


def main():
    tests=0
    for scores in [np.array([-3,-2,-1,0],np.float32),np.array([-2,-2,-1,-1],np.float32),np.zeros(4,np.float32)]:
        for labels in product([False,True],repeat=4):
            labels=np.array(labels,dtype=bool);rule=threshold_rule(scores,labels)
            predicted=apply_rule(scores,rule);n1=int(labels.sum());n0=4-n1
            if not n1 or not n0:
                assert not rule["fit_estimable"] and not predicted.any()
            else:
                options=[]
                for tau in [-float("inf"),*sorted(set(map(float,scores))),float("inf")]:
                    p=scores>tau
                    error=(Fraction(int((labels&~p).sum()),n1)+Fraction(int((~labels&p).sum()),n0))/2
                    options.append(((error,int(p.sum()),-tau),p))
                expected=min(options,key=lambda x:x[0])[1]
                assert np.array_equal(predicted,expected)
                assert abs(rule["balanced_error"]-float(min(options,key=lambda x:x[0])[0][0]))<1e-15
            tests+=1
    assert support_metrics([False,False],[False,True])["balanced_accuracy"] is None
    assert support_metrics([False,True],[False,True])["balanced_accuracy"]==1
    rows=json.loads((SOURCES/"sat_published_neuron_rules.json").read_text())["single_assignment_general_units"]
    values=[.8]*17;values[2]=None
    macro=assignment_macro(values,rows);assert macro["assignments"]["TTTFF"] is None and macro["macro"] is None
    fake=[];q=[];y=[]
    for i,row in enumerate(rows):
        for origin in range(4):
            fake.append(dict(split="assessment",origin_index=origin,unit_index=i))
            q.append(1 if origin<2 else 2);y.append([origin<2,False])
    q=np.array(q);y=np.array(y,dtype=bool);pred=np.repeat(y[:,:,None],len(MODELS),axis=2).astype(float)
    pred[:,:,1]=0;d=y[:,0].astype(int)-y[:,1].astype(int)
    scratch=ROOT/"_selfcheck";scratch.mkdir(exist_ok=True)
    saved_root=analyze.ROOT;analyze.ROOT=scratch
    try:boot=analyze.clustered_bootstrap(fake,np.ones(len(fake),bool),y,pred,d,q,rows)
    finally:analyze.ROOT=saved_root
    assert boot["intervals"]["interaction"]["na_draws"]>0
    assert boot["intervals"]["interaction"]["valid_draws"]<1900
    assert boot["intervals"]["interaction"]["interval_2_5_97_5"] is None
    raw=np.load(scratch/"bootstrap_raw.npz",allow_pickle=False)
    finite=raw["macro_interaction"][np.isfinite(raw["macro_interaction"])]
    assert np.all(finite==1)
    assert np.allclose(raw["macro_balanced_accuracy"][:,0][np.isfinite(raw["macro_balanced_accuracy"][:,0])],1)
    assert np.allclose(raw["macro_balanced_accuracy"][:,1][np.isfinite(raw["macro_balanced_accuracy"][:,1])],.5)
    receipt=dict(status="PASS",synthetic_threshold_cases=tests,missing_class_and_duplicate_assignment_na_checks=True,bootstrap_synthetic_test=boot["intervals"]["interaction"],no_neural_inference=True,no_pilot_bank_access=True)
    write_json("selfcheck.json",receipt);print(json.dumps(receipt))


if __name__=="__main__":main()
