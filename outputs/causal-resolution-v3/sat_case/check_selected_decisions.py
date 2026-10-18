"""Frozen 128-menu checkpoint audit after fixed development decoder fits."""
import json
import time
import numpy as np
from pilot_common import *
from runtime import load_model
from generate_pilot import infer


def main():
    check_contract()
    start=time.monotonic()
    bank=np.load(ROOT/"pilot_bank.npz",allow_pickle=False)
    predictions=np.load(ROOT/"predictive_outputs.npz",allow_pickle=False)
    names=predictions["report_names"].tolist()
    menus=np.sort(np.random.default_rng(2026101515).choice(10240,size=128,replace=False))
    origins=3072+menus//10;slots=menus%10
    originals=bank["clauses"][origins]
    edited=[];labels=[]
    for r,name in enumerate(names):
        new=predictions["selected_new_clauses"][r,menus//10,slots]
        variant=originals.copy()
        variant[np.arange(128),slots]=new
        edited.append(variant)
        labels.append(bank["post_sat"][origins,slots,new])
    edited=np.concatenate(edited)
    labels=np.array(labels)
    model,config,metadata=load_model()
    logits,_=infer(model,encode(edited))
    logits=logits.reshape(len(names),128,16)
    expected=np.where(labels,13,14)
    correct=logits.argmax(axis=-1)==expected
    original_correct=bank["logits"][origins].argmax(axis=-1)==13
    summary={}
    for r,name in enumerate(names):
        stratum=correct[r]&original_correct
        summary[name]=dict(menus=128,original_checkpoint_correct=int(original_correct.sum()),edited_checkpoint_correct=int(correct[r].sum()),both_original_and_edited_correct=int(stratum.sum()),semantic_unsat_selected=int((~labels[r]).sum()),semantic_unsat_selected_in_both_correct_stratum=int((~labels[r]&stratum).sum()),stratum_semantic_selection_rate=float((~labels[r])[stratum].mean()) if stratum.any() else None,checkpoint_unsat_selected=int((logits[r].argmax(axis=-1)==14).sum()))
    np.savez_compressed(ROOT/"selected_decision_check.npz",menu_indices=menus,origin_indices=origins,slots=slots,report_names=np.array(names),edited_clauses=edited,edited_logits=logits,semantic_post_sat=labels,checkpoint_correct=correct,original_checkpoint_correct=original_correct)
    receipt=dict(status="PASS",scope="preregistered128developmentmenus only; not protected confirmation",contract_sha256=CONTRACT_SHA256,source=metadata,menu_seed=2026101515,metrics=summary,code_sha256=sha256(__file__),predictions_sha256=sha256(ROOT/"predictive_outputs.npz"),elapsed_seconds=time.monotonic()-start)
    (ROOT/"selected_decision_check.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps(summary,sort_keys=True))


if __name__=="__main__":main()
