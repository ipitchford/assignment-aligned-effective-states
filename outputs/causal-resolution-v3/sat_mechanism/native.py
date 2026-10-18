"""CPU extraction and separately invoked numerical replay; fixed 34 units only."""
import argparse
import json
import time
import numpy as np
import torch
from protocol import *
from runtime import load_model

assert torch.__version__.split("+")[0]=="2.1.2" and np.__version__=="1.26.4", "Pinned CPU runtime drift"


def infer(model,tokens):
    gathered={"pre":[],"post":[],"logits":[]};cache={}
    def capture(key):
        def hook(tensor,name):cache[key]=tensor[:,-1,PUBLISHED_UNITS].detach().cpu().numpy().copy()
        return hook
    model.blocks[1].mlp.hook_pre.add_hook(capture("pre"))
    model.blocks[1].mlp.hook_post.add_hook(capture("post"))
    try:
        with torch.inference_mode():
            for offset in range(0,len(tokens),128):
                logits=model(torch.from_numpy(tokens[offset:offset+128]))[:,-1,:].cpu().numpy()
                gathered["logits"].append(logits)
                for key in ["pre","post"]:gathered[key].append(cache[key])
    finally:model.remove_all_hooks()
    return {key:np.concatenate(values) for key,values in gathered.items()}


def extract(which):
    frozen();start=time.monotonic()
    target=ROOT/f"{which}_native.npz"
    if target.exists():raise RuntimeError("Refusing repeat extraction: "+which)
    if which=="original":
        bank=np.load(PILOT/"pilot_bank.npz",allow_pickle=False);tokens=bank["tokens"]
    else:
        assert read_json("stage_a.json")["gate"]["pass"]
        tokens=np.load(ROOT/"query_tokens.npz",allow_pickle=False)["unique_tokens"]
    model,config,metadata=load_model(threads=2)
    result=infer(model,tokens)
    rectification_difference=float(np.abs(result["post"]-np.maximum(result["pre"],0)).max())
    assert rectification_difference==0
    assert all(np.isfinite(v).all() for v in result.values())
    diagnostics=dict(rectification_max_abs=rectification_difference)
    if which=="original":
        difference=float(np.abs(result["post"]-bank["activations"][:,PUBLISHED_UNITS]).max())
        diagnostics.update(pilot_post_max_abs=difference,pilot_logits_max_abs=float(np.abs(result["logits"]-bank["logits"]).max()))
        assert np.allclose(result["post"],bank["activations"][:,PUBLISHED_UNITS],atol=1e-4,rtol=1e-5)
    np.savez_compressed(target,**result,units=PUBLISHED_UNITS,pre_distance_to_zero=np.abs(result["pre"]),post_distance_to_half=np.abs(result["post"]-.5))
    write_json(f"{which}_extraction.json",dict(status="PASS",which=which,forwards=len(tokens),retained_units=PUBLISHED_UNITS.tolist(),source=metadata,diagnostics=diagnostics,minimum_pre_zero_distance=float(np.abs(result["pre"]).min()),minimum_post_half_distance=float(np.abs(result["post"]-.5).min()),elapsed_seconds=time.monotonic()-start,result_sha256=sha(target),freeze_sha256=sha(ROOT/"freeze.json")))
    print(json.dumps(dict(status="PASS",stage=which,forwards=len(tokens),diagnostics=diagnostics)))


def replay():
    frozen();start=time.monotonic()
    if (ROOT/"replay.json").exists():raise RuntimeError("Replay already completed")
    model,config,metadata=load_model(threads=2)
    bank=np.load(PILOT/"pilot_bank.npz",allow_pickle=False)
    original=np.load(ROOT/"original_native.npz",allow_pickle=False)
    inputs=[("original",bank["tokens"][:8],original)]
    if (ROOT/"edited_native.npz").exists():
        tokens=np.load(ROOT/"query_tokens.npz",allow_pickle=False)["unique_tokens"][:8]
        inputs.append(("edited",tokens,np.load(ROOT/"edited_native.npz",allow_pickle=False)))
    receipts=[];arrays={}
    for label,tokens,saved in inputs:
        fresh=infer(model,tokens)
        for key,value in fresh.items():arrays[label+"_"+key]=value
        differences={key:float(np.abs(fresh[key]-saved[key][:len(tokens)]).max()) for key in fresh}
        same_events=bool(np.array_equal(fresh["post"]>=.5,saved["post"][:len(tokens)]>=.5))
        same_clipping=bool(np.array_equal(fresh["pre"]<0,saved["pre"][:len(tokens)]<0))
        compatible=all(np.allclose(fresh[k],saved[k][:len(tokens)],atol=1e-4,rtol=1e-5) for k in fresh)
        receipts.append(dict(which=label,forwards=len(tokens),max_abs_differences=differences,same_published_events=same_events,same_clipping=same_clipping,compatible=compatible))
    np.savez_compressed(ROOT/"replay_raw.npz",**arrays)
    write_json("replay.json",dict(status="PASS" if all(r["compatible"] and r["same_published_events"] and r["same_clipping"] for r in receipts) else "FAIL",scope="separate-process numerical replay, not exact-real formal verification",source=metadata,receipts=receipts,total_forwards=sum(r["forwards"] for r in receipts),elapsed_seconds=time.monotonic()-start))
    print(json.dumps(receipts))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["original","edited","replay"])
    args=parser.parse_args();replay() if args.stage=="replay" else extract(args.stage)
