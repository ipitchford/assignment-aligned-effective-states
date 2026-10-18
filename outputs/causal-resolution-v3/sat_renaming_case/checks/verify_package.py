"""Read-only local package identity verification; standard library only."""
from pathlib import Path
import hashlib
import json


def verify(package):
    package = Path(package).resolve()
    manifest_path = package / "PACKAGE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    seen = set()
    total = 0
    for entry in manifest["files"]:
        rel = entry["path"]
        path = (package / rel).resolve()
        if rel in seen or not path.is_relative_to(package) or path.is_symlink():
            raise RuntimeError("Unsafe or duplicate package path: " + rel)
        seen.add(rel)
        data = path.read_bytes()
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise RuntimeError("Package identity mismatch: " + rel)
        total += len(data)
    actual = {p.relative_to(package).as_posix() for p in package.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    if actual != seen | {"PACKAGE_MANIFEST.json"}:
        raise RuntimeError("Package file inventory differs: " + repr(sorted(actual ^ (seen | {"PACKAGE_MANIFEST.json"}))))
    return {"status": "PASS", "files": len(seen), "bytes": total,
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
