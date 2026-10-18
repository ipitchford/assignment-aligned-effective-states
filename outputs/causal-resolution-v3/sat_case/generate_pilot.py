"""One frozen development pilot: generate origins, reports and candidate forks."""
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pilot_common import *
from runtime import load_model


def infer(model, tokens, activations=False, batch_size=128):
    outputs=[]; hidden=[]
    hook=model.blocks[1].mlp.hook_post
    cache={}
    if activations:
        hook.add_hook(lambda tensor,name: cache.update(z=tensor[:,-1,:].detach().cpu().numpy().copy()))
    try:
        with torch.inference_mode():
            for offset in range(0,len(tokens),batch_size):
                logits=model(torch.from_numpy(tokens[offset:offset+batch_size]))[:,-1,:]
                outputs.append(logits.cpu().numpy())
                if activations: hidden.append(cache["z"])
    finally:
        model.remove_all_hooks()
    return np.concatenate(outputs), np.concatenate(hidden) if activations else None


def main():
    check_contract()
    if (ROOT/"pilot_bank.npz").exists():
        raise RuntimeError("Pilot bank already exists; refusing to overwrite")
    start=time.monotonic()
    rng=np.random.default_rng(2026101511)
    formulas=[]; seen=set(); draws=0
    while len(formulas)<4096:
        candidate=rng.integers(0,40,size=10,dtype=np.int16)
        draws+=1
        key=tuple(int(x) for x in candidate)
        if key in seen or not any(python_truth_table(candidate)):
            continue
        seen.add(key); formulas.append(candidate)
    clauses=np.array(formulas,dtype=np.int16)
    tokens=encode(clauses)
    counts=semantic_counts(clauses)
    labels=all_edit_labels(clauses,counts)
    assert (counts==0).any(axis=1).all()
    assert np.array_equal(tokens[:,1:40:4] % 5,CLAUSES[clauses,0]%5)
    assert (tokens[:,1:40:4]%5 < tokens[:,2:40:4]%5).all()
    for i in range(8):
        assert np.array_equal(np.array(python_truth_table(clauses[i])),counts[i]==0)
        for slot in range(10):
            for new in range(40):
                edited=clauses[i].copy(); edited[slot]=new
                assert any(python_truth_table(edited))==bool(labels[i,slot,new])
    random_units=np.sort(np.random.default_rng(2026101512).choice(np.setdiff1d(np.arange(512),PUBLISHED_UNITS),size=34,replace=False))
    model,config,metadata=load_model()
    with torch.inference_mode():
        x=torch.from_numpy(tokens[:8])
        whole=model(x)
        first=model.blocks[0](model.pos_embed(model.embed(x)))
        mid=first+model.blocks[1].attn(first)
        mlp=F.relu(torch.einsum('md,bpd->bpm',model.blocks[1].mlp.W_in,mid)+model.blocks[1].mlp.b_in)
        resid=mid+torch.einsum('dm,bpm->bpd',model.blocks[1].mlp.W_out,mlp)+model.blocks[1].mlp.b_out
        decomposed=model.unembed(resid)
        compatibility_error=float((whole-decomposed).abs().max())
        assert torch.allclose(whole,decomposed,atol=1e-5,rtol=1e-6)
    logits,hidden=infer(model,tokens,True)
    codes=report_codes(hidden[:,PUBLISHED_UNITS]>=0.5)
    threshold_distance=np.abs(hidden[:,PUBLISHED_UNITS]-0.5).min(axis=1)
    margins=correct_margin(logits,np.full(4096,13))
    eligible=(logits.argmax(axis=1)==13)&(threshold_distance>=1e-4)&(margins>1e-4)&(codes!=0)
    np.savez_compressed(ROOT/"pilot_bank.npz",clauses=clauses,tokens=tokens,counts=counts,truth_tables=counts==0,post_sat=labels,logits=logits,activations=hidden,published_units=PUBLISHED_UNITS,random_units=random_units,codes=codes,eligible=eligible,threshold_distance=threshold_distance,original_margins=margins)
    candidates=[]; used_pairs=set()
    for code in sorted(int(c) for c in np.unique(codes[eligible])):
        origins=np.flatnonzero(eligible&(codes==code))
        if len(origins)<2: continue
        for slot in range(10):
            for old in range(40):
                members=origins[clauses[origins,slot]==old]
                if len(members)<2: continue
                for new in range(40):
                    if new==old: continue
                    outcomes=labels[members,slot,new]
                    if outcomes.all() or not outcomes.any(): continue
                    for u in range(len(members)):
                        for v in range(u+1,len(members)):
                            a,b=int(members[u]),int(members[v])
                            if outcomes[u]==outcomes[v] or (a,b) in used_pairs: continue
                            used_pairs.add((a,b))
                            candidates.append(dict(report_integer=code,slot=slot,old_clause=old,new_clause=new,origin_indices=[a,b],semantic_post_sat=[bool(outcomes[u]),bool(outcomes[v])]))
                            if len(candidates)==128: break
                        if len(candidates)==128: break
                    if len(candidates)==128: break
                if len(candidates)==128: break
            if len(candidates)==128: break
        if len(candidates)==128: break
    first_accepted=None
    if candidates:
        edited=[]
        for c in candidates:
            for i in c["origin_indices"]:
                row=clauses[i].copy();row[c["slot"]]=c["new_clause"];edited.append(row)
        edit_logits,_=infer(model,encode(np.array(edited,dtype=np.int16)))
        np.savez_compressed(ROOT/"candidate_logits.npz",edited_clauses=np.array(edited,dtype=np.int16),edited_logits=edit_logits)
        for k,c in enumerate(candidates):
            target=np.where(c["semantic_post_sat"],13,14)
            result=edit_logits[2*k:2*k+2]
            correct=result.argmax(axis=1)==target
            post_margin=correct_margin(result,target)
            accepted=bool(correct.all() and (post_margin>1e-4).all())
            c.update(candidate_index=k,checkpoint_post_correct=correct.tolist(),post_full_vocab_margins=post_margin.tolist(),accepted=accepted)
            if accepted and first_accepted is None:
                first_accepted=k
                ids=c["origin_indices"]
                np.savez_compressed(ROOT/"primary_witness.npz",origin_indices=ids,original_clauses=clauses[ids],original_tokens=tokens[ids],original_logits=logits[ids],original_activations=hidden[ids][:,PUBLISHED_UNITS],original_bits=hidden[ids][:,PUBLISHED_UNITS]>=0.5,counts=counts[ids],truth_tables=counts[ids]==0,edited_clauses=np.array(edited[2*k:2*k+2]),edited_tokens=encode(np.array(edited[2*k:2*k+2])),edited_logits=result,post_sat=np.array(c["semantic_post_sat"]),slot=c["slot"],old_clause=c["old_clause"],new_clause=c["new_clause"],published_units=PUBLISHED_UNITS)
    (ROOT/"candidate_forks.json").write_text(json.dumps(candidates,indent=2)+"\n")
    receipt=dict(status="PASS",contract_sha256=CONTRACT_SHA256,source=metadata,generated=4096,draws=draws,initially_sat=4096,original_checkpoint_correct=int((logits.argmax(axis=1)==13).sum()),eligible_nonzero_correct_robust=int(eligible.sum()),unique_published_codes=len(np.unique(codes)),random_units=random_units.tolist(),candidate_count=len(candidates),accepted_candidate_count=sum(c["accepted"] for c in candidates),first_accepted_candidate=first_accepted,compatibility=dict(strict_load=True,canonical_token_checks=True,independent_semantic_checks=8*401,whole_vs_three_stage_max_abs=compatibility_error),elapsed_seconds=time.monotonic()-start,code_sha256={p:sha256(ROOT/p) for p in ["runtime.py","pilot_common.py","generate_pilot.py"]},bank_sha256=sha256(ROOT/"pilot_bank.npz"))
    (ROOT/"pilot_generation.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:receipt[k] for k in ["status","generated","original_checkpoint_correct","unique_published_codes","candidate_count","accepted_candidate_count","first_accepted_candidate","elapsed_seconds"]}))


if __name__=="__main__": main()
