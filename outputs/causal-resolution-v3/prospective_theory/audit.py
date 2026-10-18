#!/usr/bin/env python3
"""Exact ranks and prospective output aliases, using weights only.

LP proposes input perturbations. Exact rational projection and rescaling make
the final aliases independently verifiable without trusting an LP status.
No train/test observations or outcomes are loaded.
"""
from __future__ import annotations
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
import sympy as sp

ROOT=Path(__file__).resolve().parent
APP=ROOT.parent/"application"
PRIME=1000003
DEN=65536*10*4096


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def determinant_mod(a):
    a=[[int(v)%PRIME for v in row] for row in a]
    n=len(a); result=1
    for c in range(n):
        r=next((i for i in range(c,n) if a[i][c]),None)
        if r is None: return 0
        if r!=c:
            a[c],a[r]=a[r],a[c]; result=-result
        pivot=a[c][c];result=result*pivot%PRIME
        inverse=pow(pivot,PRIME-2,PRIME)
        for i in range(c+1,n):
            factor=a[i][c]*inverse%PRIME
            for j in range(c+1,n):a[i][j]=(a[i][j]-factor*a[c][j])%PRIME
    return result%PRIME


def rank_witness(matrix):
    original=[[int(v) for v in row] for row in matrix]
    a=[[v%PRIME for v in row] for row in original]
    ids=list(range(len(a))); rank=0; columns=[]; rows=[]
    for c in range(len(a[0])):
        selected=next((i for i in range(rank,len(a)) if a[i][c]),None)
        if selected is None:continue
        a[rank],a[selected]=a[selected],a[rank]
        ids[rank],ids[selected]=ids[selected],ids[rank]
        inv=pow(a[rank][c],PRIME-2,PRIME)
        for i in range(rank+1,len(a)):
            factor=a[i][c]*inv%PRIME
            for j in range(c+1,len(a[0])):a[i][j]=(a[i][j]-factor*a[rank][j])%PRIME
            a[i][c]=0
        rows.append(ids[rank]);columns.append(c);rank+=1
        if rank==min(len(a),len(a[0])):break
    minor=[[original[r][c] for c in columns] for r in rows]
    determinant=determinant_mod(minor)
    assert determinant!=0
    return {"rank_lower":rank,"minor_rows":rows,"minor_columns":columns,
            "prime":PRIME,"minor_determinant_mod_prime":determinant}


def matvec(a,x):
    return [sum((F(int(v))*z for v,z in zip(row,x)),F()) for row in a]


def exact_projection(E,proposal,certificate):
    rows=certificate["minor_rows"];cols=certificate["minor_columns"]
    free=[j for j in range(E.shape[1]) if j not in cols]
    candidate=[F(float(v)).limit_denominator(1000000) for v in proposal]
    square=sp.Matrix([[int(E[i,j]) for j in cols] for i in rows])
    rhs=[-sum((F(int(E[i,j]))*candidate[j] for j in free),F()) for i in rows]
    rhs_s=sp.Matrix([sp.Rational(v.numerator,v.denominator) for v in rhs])
    solution=square.inv(method="DM")*rhs_s
    for j,v in zip(cols,solution):candidate[j]=F(int(v.p),int(v.q))
    assert all(v==0 for v in matvec(E,candidate))
    return candidate


def realised_alias(seed,model,C,groups):
    W=np.asarray(model["w1"],dtype=np.int64)
    bias=np.asarray(model["b1"],dtype=np.int64)
    centre=8*np.ones(64,dtype=np.int64)
    pre=centre@W+16*bias
    assert np.all(pre!=0)
    active=(pre>0).astype(np.int64)
    E=(C*active[:,None]).T@W.T
    rank=rank_witness(E.tolist())
    assert rank["rank_lower"]==9 and np.all(E.sum(axis=0)==0)
    # Symmetric endpoints centre +/- d remain in one closed ReLU cell.
    inequality=np.vstack([W.T,-W.T]).astype(float)
    limits=np.r_[np.abs(pre),np.abs(pre)].astype(float)
    row_scale=np.maximum(np.max(np.abs(inequality),axis=1),1)
    best=None
    for g,group in enumerate(groups):
        masked=np.zeros_like(C);masked[group]=C[group]*active[group,None]
        prospective=masked.T@W.T
        for k in range(10):
            objective=-prospective[k].astype(float)/DEN
            eq=E[rank["minor_rows"]].astype(float)
            eqscale=np.maximum(np.max(np.abs(eq),axis=1),1)
            result=linprog(objective,A_ub=inequality/row_scale[:,None],
                           b_ub=limits/row_scale,A_eq=eq/eqscale[:,None],
                           b_eq=np.zeros(9),bounds=[(-8,8)]*64,method="highs")
            assert result.success,result.message
            score=-result.fun
            if best is None or score>best[0]:best=(score,g,k,result.x)
    _,g,k,proposal=best
    direction=exact_projection(E,proposal,rank)
    changes=matvec(W.T,direction)
    ratios=[F(8)/abs(v) for v in direction if v]
    ratios += [F(abs(int(pre[j])))/abs(v) for j,v in enumerate(changes) if v]
    scale=min(F(1),*ratios)
    direction=[scale*v for v in direction]
    changes=matvec(W.T,direction)
    plus_input=[F(8)+v for v in direction]
    minus_input=[F(8)-v for v in direction]
    assert all(0<=v<=16 for v in plus_input+minus_input)
    plus_h=[max(F(0),F(int(pre[j]))+changes[j]) for j in range(32)]
    minus_h=[max(F(0),F(int(pre[j]))-changes[j]) for j in range(32)]
    plus_head=matvec(C.T,plus_h);minus_head=matvec(C.T,minus_h)
    assert plus_head==minus_head
    output_bias=np.asarray(model["b2"],dtype=np.int64)
    centred_bias=[F(10*int(v)-int(output_bias.sum()),10*4096) for v in output_bias]
    full_current=[v/DEN+b for v,b in zip(plus_head,centred_bias)]
    group=groups[g]
    full_gap=sum((F(int(C[j,k]))*(plus_h[j]-minus_h[j]) for j in group),F())/DEN
    assert full_gap>0
    return {"seed":seed,"source":"two continuous input vectors in [0,16]^64, constructed from weights; no sampled images",
            "centre_pixel_value":8,"group_index":g,"output_coordinate":k,
            "group_units":group,"active_units_at_centre":[j for j in range(32) if active[j]],
            "input_plus_exact":[str(v) for v in plus_input],
            "input_minus_exact":[str(v) for v in minus_input],
            "hidden_plus_integer_scale_exact":[str(v) for v in plus_h],
            "hidden_minus_integer_scale_exact":[str(v) for v in minus_h],
            "common_current_centred_projection_exact":[str(v/DEN) for v in plus_head],
            "centred_output_bias_exact":[str(v) for v in centred_bias],
            "common_full_current_centred_logits_exact":[str(v) for v in full_current],
            "full_ablation_strength_exact":"1",
            "half_ablation_strength_exact":"1/2",
            "full_ablation_effect_gap_exact":str(full_gap),
            "full_ablation_effect_gap_float":float(full_gap),
            "full_ablation_decoder_error_lower_exact":str(full_gap/2),
            "half_ablation_decoder_error_lower_exact":str(full_gap/4),
            "exact_head_nullspace_rank_witness":rank,
            "status":"all input bounds, ReLU evaluations, equal head outputs, and action gap checked with fractions"}


def verify_alias(record,model,C,groups):
    x=[F(v) for v in record["input_plus_exact"]]
    y=[F(v) for v in record["input_minus_exact"]]
    assert all(0<=v<=16 for v in x+y)
    W=np.asarray(model["w1"]).T.tolist();bias=model["b1"].tolist()
    def hidden(v):return [max(F(0),s+16*int(b)) for s,b in zip(matvec(W,v),bias)]
    hx,hy=hidden(x),hidden(y)
    assert [str(v) for v in hx]==record["hidden_plus_integer_scale_exact"]
    assert [str(v) for v in hy]==record["hidden_minus_integer_scale_exact"]
    assert matvec(C.T,hx)==matvec(C.T,hy)
    group=groups[record["group_index"]];k=record["output_coordinate"]
    gap=sum((F(int(C[j,k]))*(hx[j]-hy[j]) for j in group),F())/DEN
    assert gap==F(record["full_ablation_effect_gap_exact"]) and gap>2


def main():
    contract_path=APP/"contract.json"
    groups=json.loads(contract_path.read_text())["interventions"]["groups"]
    assert sorted(j for group in groups for j in group)==list(range(32))
    records=[];aliases=[]
    for seed in (17,29,43):
        precision_path=APP/f"seed-{seed}/precision-gain-1.npz"
        model_path=APP/f"seed-{seed}/model.npz"
        with np.load(precision_path,allow_pickle=False) as p:
            C=np.asarray(p["C"],dtype=np.int64);M=np.asarray(p["M"],dtype=np.int64)
        with np.load(model_path,allow_pickle=False) as p:
            model={key:p[key] for key in ("w1","b1","w2","b2")}
        assert np.all(C.sum(axis=1)==0) and np.all(M>0)
        head=rank_witness(C.T.tolist());assert head["rank_lower"]==9
        block=[rank_witness(C[group].T.tolist()) for group in groups]
        assert [b["rank_lower"] for b in block]==[8]*4
        stacked=[]
        for group in groups:
            z=np.zeros_like(C);z[group]=C[group]
            stacked.extend(z.T.tolist())
        action=rank_witness(stacked);assert action["rank_lower"]==32
        W=model["w1"]
        probe=np.vstack([np.zeros((1,64),dtype=np.int64),16*np.ones((1,64),dtype=np.int64),
                         16*np.eye(64,dtype=np.int64),16*(W.T>0).astype(np.int64)])
        hidden=np.maximum(0,probe@W+16*model["b1"])
        span=rank_witness(hidden.tolist());assert span["rank_lower"]==32
        affine_span=rank_witness((hidden[1:]-hidden[0]).tolist())
        assert affine_span["rank_lower"]==32
        # Starting with current output and adding the first m group actions.
        progression=[];progression_certificates=[]
        for count in range(5):
            bank=C.T.tolist()+stacked[:10*count]
            certificate=rank_witness(bank)
            progression.append(certificate["rank_lower"])
            progression_certificates.append(certificate)
        merged=[groups[0]+groups[1],groups[2]+groups[3]]
        merged_certificates=[rank_witness(C[group].T.tolist()) for group in merged]
        merged_ranks=[r["rank_lower"] for r in merged_certificates]
        record={"seed":seed,"model_sha256":sha(model_path),"precision_source_sha256":sha(precision_path),
                "current_output_rank":9,"current_output_rank_upper_reason":"ten centred rows sum to zero",
                "current_output_rank_witness":head,"group_block_ranks":[8]*4,
                "group_rank_witnesses":block,"prospective_one_step_bank_rank":32,
                "prospective_action_closed_rank":32,"prospective_rank_witness":action,
                "current_output_plus_first_m_actions_ranks":progression,
                "current_output_plus_first_m_actions_rank_witnesses":progression_certificates,
                "coarsened_two_group_ranks":merged_ranks,
                "coarsened_two_group_rank_witnesses":merged_certificates,
                "coarsened_two_group_action_closed_rank":sum(merged_ranks),
                "constructed_input_probe_recipe":"zero; all16;64 single-coordinate16 basis probes;32 sign-positive W1 unit-maximising corners",
                "constructed_probe_count":len(probe),"constructed_hidden_span_rank":32,
                "constructed_span_rank_witness":span,
                "constructed_hidden_affine_span_rank":32,
                "constructed_affine_span_rank_witness":affine_span}
        alias=realised_alias(seed,model,C,groups)
        verify_alias(alias,model,C,groups)
        records.append(record);aliases.append(alias)
        print(json.dumps({"seed":seed,"head_rank":9,"action_rank":32,"probe_span":32,
                          "rank_progression":progression,"full_action_gap":alias["full_ablation_effect_gap_float"],
                          "group":alias["group_index"],"output":alias["output_coordinate"]}),flush=True)
    receipt={"status":"PASS","analysis":"retrospective weight-only prospective sufficiency audit",
             "test_outcomes_read":False,"groups":groups,"contract_sha256":sha(contract_path),
             "script_sha256":sha(Path(__file__)),"rank_certificates":records,"exact_realised_aliases":aliases}
    (ROOT/"results.json").write_text(json.dumps(receipt,indent=2)+"\n")


if __name__=="__main__":main()
