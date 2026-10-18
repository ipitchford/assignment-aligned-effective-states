"""Independent-process replay of the saved fork, with a token-level Boolean oracle."""
from itertools import product
import json
import time
import numpy as np
import torch
from runtime import load_model,ROOT,sha256


def token_oracle(tokens):
    assert len(tokens)==41 and int(tokens[40])==12
    clauses=[]
    for slot in range(10):
        begin,left,right,end=map(int,tokens[4*slot:4*slot+4])
        assert begin==10 and end==11 and 0<=left<10 and 0<=right<10
        assert left%5<right%5
        clauses.append((left,right))
    satisfying=[]
    for values in product([False,True],repeat=5):
        def literal(token): return values[token] if token<5 else not values[token-5]
        if all(literal(left) or literal(right) for left,right in clauses):
            satisfying.append("".join("1" if v else "0" for v in values))
    names=[f"x{i}" for i in range(5)]+[f"~x{i}" for i in range(5)]
    return satisfying,[f"({names[a]} OR {names[b]})" for a,b in clauses]


def infer(model,tokens,units):
    cache={}
    model.blocks[1].mlp.hook_post.add_hook(lambda tensor,name: cache.update(z=tensor[:,-1,units].detach().cpu().numpy().copy()))
    try:
        with torch.inference_mode(): out=model(torch.from_numpy(tokens))[:,-1,:].cpu().numpy()
    finally: model.remove_all_hooks()
    return out,cache["z"]


def main():
    start=time.monotonic()
    witness=np.load(ROOT/"primary_witness.npz",allow_pickle=False)
    tokens=witness["original_tokens"];edited=witness["edited_tokens"]
    slot=int(witness["slot"])
    assert np.array_equal(tokens[0,4*slot:4*slot+4],tokens[1,4*slot:4*slot+4])
    assert np.array_equal(edited[0,4*slot:4*slot+4],edited[1,4*slot:4*slot+4])
    assert not np.array_equal(tokens[0,4*slot:4*slot+4],edited[0,4*slot:4*slot+4])
    keep=np.ones(41,dtype=bool);keep[4*slot:4*slot+4]=False
    assert np.array_equal(tokens[:,keep],edited[:,keep])
    before=[token_oracle(row) for row in tokens]
    after=[token_oracle(row) for row in edited]
    assert all(x[0] for x in before)
    assert bool(after[0][0])!=bool(after[1][0])
    units=witness["published_units"].astype(int)
    model,config,metadata=load_model()
    logits,acts=infer(model,tokens,units)
    edited_logits,_=infer(model,edited,units)
    bits=acts>=0.5
    assert np.array_equal(bits[0],bits[1]) and bits[0].any()
    assert np.array_equal(bits,witness["original_bits"])
    assert np.allclose(acts,witness["original_activations"],atol=1e-4,rtol=1e-5)
    expected=np.array([13 if row[0] else 14 for row in after])
    assert np.all(logits.argmax(axis=1)==13)
    assert np.array_equal(edited_logits.argmax(axis=1),expected)
    assert np.abs(acts-0.5).min()>=1e-4
    differences=dict(original_logit_max_abs=float(np.abs(logits-witness["original_logits"]).max()),edited_logit_max_abs=float(np.abs(edited_logits-witness["edited_logits"]).max()),activation_max_abs=float(np.abs(acts-witness["original_activations"]).max()))
    # Diagnostic alternate arithmetic, not an interval proof of real arithmetic.
    model.double()
    logits64,acts64=infer(model,tokens,units)
    edited64,_=infer(model,edited,units)
    double_check=dict(same_original_bits=bool(np.array_equal(acts64>=0.5,bits)),original_correct=bool(np.all(logits64.argmax(axis=1)==13)),edited_correct=bool(np.array_equal(edited64.argmax(axis=1),expected)))
    assert all(double_check.values())
    receipt=dict(status="PASS",scope="independent-process original-checkpoint replay and independent token-level exact Boolean enumeration; float64 diagnostic is not exact-real formal verification",source=metadata,witness_sha256=sha256(ROOT/"primary_witness.npz"),verifier_sha256=sha256(__file__),origin_indices=witness["origin_indices"].tolist(),slot_zero_based=slot,old_clause=before[0][1][slot],new_clause=after[0][1][slot],original_formulas=[x[1] for x in before],original_satisfying_valuations=[x[0] for x in before],edited_satisfying_valuations=[x[0] for x in after],valuation_order="x0,x1,x2,x3,x4;0=false,1=true",published_units=units.tolist(),identical_original_bits=bits[0].astype(int).tolist(),active_published_units=units[bits[0]].tolist(),original_activations=acts.tolist(),original_logits=logits.tolist(),edited_logits=edited_logits.tolist(),minimum_threshold_distance=float(np.abs(acts-.5).min()),replay_differences=differences,float64_diagnostic=double_check,elapsed_seconds=time.monotonic()-start)
    (ROOT/"primary_witness.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:receipt[k] for k in ["status","origin_indices","old_clause","new_clause","original_satisfying_valuations","edited_satisfying_valuations","active_published_units","minimum_threshold_distance","replay_differences","float64_diagnostic"]}))


if __name__=="__main__":main()
