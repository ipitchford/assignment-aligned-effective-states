"""Write the deterministic pre-inference bank audit once."""
import json
import platform
import numpy as np
from protocol import HERE, bank_preflight, sha256


def main():
    target = HERE / "preflight.json"
    result = bank_preflight()
    result["environment"] = {"python": platform.python_version(), "numpy": np.__version__}
    result["protocol_sha256"] = sha256(HERE / "protocol.py")
    result["source_sha256"] = {"bank": sha256(__import__("protocol").BANK),
                               "prior_native_receipt": sha256(__import__("protocol").PRIOR_NATIVE_RECEIPT),
                               "prior_analysis": sha256(__import__("protocol").PRIOR_ANALYSIS)}
    with target.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True); handle.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
