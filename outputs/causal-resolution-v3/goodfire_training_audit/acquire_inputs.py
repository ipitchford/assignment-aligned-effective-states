#!/usr/bin/env python3
"""Copy prior source bytes and fetch only six frozen public JSON URLs, once each."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PRIOR = ROOT / "work/next_phase/goodfire_release_availability_20261016_sources"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
records = []
source_base = "https://already-clean.goodfire.com/assets/early-training/"
copies = [("r3_results.md", "RESULTS.md", source_base+"RESULTS.md"),
          ("r3_protocol.md", "PROTOCOL.md", source_base+"PROTOCOL.md"),
          ("r3_manifest.json", "manifest.json", "https://already-clean.goodfire.com/assets/training-map/manifest.json")]
copies += [("r3_tiny_seed0_"+kind+".json", "fresh-tiny-seed0/"+kind+".json", source_base+"fresh-tiny-seed0/"+kind+".json")
           for kind in ("metrics", "lineage", "causal")]
for old_name, new_name, url in copies:
    src = PRIOR / old_name
    target = HERE / "inputs" / new_name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    shutil.copy2(src, target)
    records.append({"mode": "unchanged_copy_of_prior_public_retrieval", "original_path": str(src.relative_to(ROOT)),
                    "url": url, "path": str(target.relative_to(HERE)), "bytes": target.stat().st_size, "sha256": sha(target)})

allowlist = json.loads((HERE / "retrieval_allowlist.json").read_text())
assert allowlist["additional_retrieval_cap"] == len(allowlist["requests"]) == 6
assert sha(HERE / "audit_scope.md") == allowlist["scope_sha256"]
for request in allowlist["requests"]:
    url = request["url"]
    target = HERE / request["local_path"]
    assert url.startswith(source_base) and url.endswith(("/metrics.json", "/lineage.json", "/causal.json"))
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"mode": "additional_public_GET", "url": url, "path": str(target.relative_to(HERE)),
               "requested_utc": datetime.now(timezone.utc).isoformat()}
    try:
        with urlopen(Request(url, headers={"User-Agent": "SourceArithmeticAudit/1.0"}), timeout=30) as response:
            body = response.read(2_000_001)
            if len(body) > 2_000_000:
                raise ValueError("Frozen 2 MB JSON body cap exceeded")
            json.loads(body)
            receipt.update({"status": response.status, "final_url": response.url,
                            "headers": {key: response.headers[key] for key in ("Content-Type", "Content-Length", "ETag", "Last-Modified", "Date") if key in response.headers}})
            with target.open("xb") as stream:
                stream.write(body)
            receipt.update({"bytes": len(body), "sha256": sha(target)})
    except Exception as exc:
        receipt["error"] = str(exc)
        if isinstance(exc, HTTPError):
            receipt["status"] = exc.code
        # Preserve failure; no retry, alternate path, or credential request.
    records.append(receipt)
    print(json.dumps({k:receipt[k] for k in ("url", "status", "bytes", "sha256", "error") if k in receipt}), flush=True)
    time.sleep(1)

with (HERE / "input_provenance.json").open("x") as stream:
    json.dump({"completed_utc": datetime.now(timezone.utc).isoformat(), "records": records}, stream, indent=2)
    stream.write("\n")
paths = [HERE / "audit_scope.md", HERE / "retrieval_allowlist.json", HERE / "acquire_inputs.py", HERE / "input_provenance.json"]
paths += sorted(p for p in (HERE / "inputs").rglob("*") if p.is_file())
freeze = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "retrospective_audit": True,
          "scope": "Freeze of source bytes before the complete arithmetic audit; not an experimental preregistration",
          "files": [{"path": str(p.relative_to(HERE)), "bytes": p.stat().st_size, "sha256": sha(p)} for p in paths]}
with (HERE / "input_freeze.json").open("x") as stream:
    json.dump(freeze, stream, indent=2)
    stream.write("\n")
