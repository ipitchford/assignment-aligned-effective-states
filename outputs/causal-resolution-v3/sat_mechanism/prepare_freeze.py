"""Select semantic queries and freeze complete code BEFORE any extraction."""
import datetime
import hashlib
import json
import shutil
import numpy as np
from protocol import *


def main():
    if (ROOT/"freeze.json").exists():raise RuntimeError("Already frozen")
    verify_pilot()
    assert sha(SOURCES/"sat_followon_design.md")==CONTRACT_SHA
    assert sha(SOURCES/"sat_published_neuron_rules.json")==MAPPING_SHA
    assert sha(PILOT/"pilot_bank.npz")==BANK_SHA
    shutil.copyfile(SOURCES/"sat_followon_design.md",ROOT/"contract.md")
    shutil.copyfile(SOURCES/"sat_published_neuron_rules.json",ROOT/"source_mapping.json")
    rows=mapping();bank=np.load(PILOT/"pilot_bank.npz",allow_pickle=False)
    clauses=bank["clauses"];counts=bank["counts"]
    reverse={tuple(map(int,row)):i for i,row in enumerate(CLAUSES)}
    digest=lambda s:hashlib.sha256(s.encode("utf-8")).hexdigest()
    queries=[];shortfalls=[];tokens=[]
    for ui,row in enumerate(rows):
        unit=row["unit"];a=row["truth_table_index"]
        for split,start,end in [("fit",0,3072),("assessment",3072,4096)]:
            for qclass in ["one","many"]:
                origins=[i for i in range(start,end) if (counts[i,a]==1 if qclass=="one" else counts[i,a]>=2)]
                prefix=lambda i:f"sat-native-v1|{split}|{unit}|{qclass}|{i}"
                origins=sorted(origins,key=lambda i:(digest(prefix(i)),i))
                selected=origins[:32]
                if len(selected)<32:shortfalls.append(dict(unit=unit,split=split,qclass=qclass,available=len(origins),selected=len(selected)))
                for origin in selected:
                    options=[]
                    for slot,old in enumerate(clauses[origin]):
                        old=int(old)
                        if CLAUSE_TRUTH[old,a]:continue
                        for kept in CLAUSES[old]:
                            kept=int(kept)
                            for newvar in range(5):
                                if newvar in (CLAUSES[old]%5):continue
                                true_literal=newvar if ((a>>(4-newvar))&1) else newvar+5
                                false_literal=(true_literal+5)%10
                                repair=reverse[tuple(sorted([kept,true_literal],key=lambda x:x%5))]
                                neutral=reverse[tuple(sorted([kept,false_literal],key=lambda x:x%5))]
                                h=digest(prefix(origin)+f"|{slot}|{kept%5}|{newvar}")
                                options.append((h,slot,kept%5,newvar,old,repair,neutral))
                    chosen=min(options)
                    h,slot,kept,newvar,old,repair,neutral=chosen
                    assert old!=repair and old!=neutral and repair!=neutral
                    assert not CLAUSE_TRUTH[old,a] and CLAUSE_TRUTH[repair,a] and not CLAUSE_TRUTH[neutral,a]
                    variants=[]
                    for new in [repair,neutral]:
                        v=clauses[origin].copy();v[slot]=new;variants.append(v)
                    # Full 32-valuation enumeration, used only for fixed query validation.
                    qnew=(~CLAUSE_TRUTH[np.array(variants)]).sum(axis=1)
                    assert int(qnew[0,a])==int(counts[origin,a])-1
                    assert int(qnew[1,a])==int(counts[origin,a])
                    encoded=encode(np.array(variants));tokens.extend(encoded)
                    queries.append(dict(query_id=len(queries),unit_index=ui,unit=unit,assignment=row["assignment"],assignment_index=a,split=split,qclass=qclass,origin_index=origin,original_q=int(counts[origin,a]),origin_hash=digest(prefix(origin)),edit_hash=h,slot=slot,old_clause=old,kept_variable=kept,new_variable=newvar,repair_clause=repair,neutral_clause=neutral,repair_q=int(qnew[0,a]),neutral_q=int(qnew[1,a]),repair_sat=bool((qnew[0]==0).any()),neutral_sat=bool((qnew[1]==0).any())))
    token_array=np.array(tokens,dtype=np.int64)
    unique,inverse=np.unique(token_array,axis=0,return_inverse=True)
    np.savez_compressed(ROOT/"query_tokens.npz",all_tokens=token_array,unique_tokens=unique,query_action_to_unique=inverse.reshape(-1,2))
    write_json("selected_queries.json",dict(selection_only_uses_symbolic_counts_and_sha256=True,rows=queries,shortfalls=shortfalls,paired_queries=len(queries),edited_rows_before_dedup=len(token_array),distinct_edited_tokens=len(unique),edit_order=["repair","neutral"]))
    assert len(queries)<=2176 and len(token_array)<=4352
    files=CODE_FILES+["contract.md","source_mapping.json","selected_queries.json","query_tokens.npz","implementation_notes.md"]
    outside=["../sat_case/pilot_bank.npz","../sat_case/runtime.py","../sat_case/pilot_common.py","../sat_case/MANIFEST.json","../sat_case/upstream/model.py","../sat_case/upstream/models/layers_2_heads_1_4_1M/900.pth"]
    write_json("freeze.json",dict(status="frozen before preactivation extraction",created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),contract_sha256=CONTRACT_SHA,mapping_sha256=MAPPING_SHA,bank_sha256=BANK_SHA,sha256={name:sha(ROOT/name) for name in files+outside},selected_paired_queries=len(queries),distinct_edited_tokens=len(unique),native_outcomes_inspected=False))
    print(json.dumps(dict(status="FROZEN",paired_queries=len(queries),edited_rows=len(token_array),unique_edited_tokens=len(unique),freeze_sha256=sha(ROOT/"freeze.json"))))


if __name__=="__main__":main()
