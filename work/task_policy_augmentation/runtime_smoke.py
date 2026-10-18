#!/usr/bin/env python3
"""One synthetic compatibility fit; reads no research cache and creates no freeze."""
from pathlib import Path
import argparse
import json
import os
import platform
import sys

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"

import numpy as np
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
import screen as s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    expected = {"python": "3.11.14", "numpy": "2.4.6",
                "scipy": "1.17.1", "sklearn": "1.9.0"}
    if versions != expected:
        raise RuntimeError(f"Analysis runtime changed: {versions!r}")
    rng = np.random.default_rng(2026101607)
    x = rng.normal(size=(256, 84)).astype(np.float32)
    y = (x[:, 0] + x[:, 68] > 0)
    settings = dict(s.SETTINGS)
    settings["max_iter"] = 3
    settings["max_leaf_nodes"] = 5
    model = HistGradientBoostingClassifier(**settings).fit(x, y)
    probability = model.predict_proba(x)[:, list(model.classes_).index(True)]
    size = s.fitted_model_size(model)
    if size["iterations"] != 3 or size["trees"] != 3 or size["leaves"] != 15:
        raise AssertionError(size)
    if probability.shape != (256,) or not np.isfinite(probability).all():
        raise AssertionError("Invalid synthetic prediction")
    receipt = {"status": "PASS", "scope": "Synthetic compatibility fit only",
               "versions": versions, "rows": 256, "features": 84,
               "thread_environment": {key: os.environ[key] for key in s.THREAD_KEYS},
               "bytecode_writes_disabled": bool(sys.dont_write_bytecode),
               "model_size": size, "research_caches_opened": 0,
               "research_fits": 0, "synthetic_fits": 1, "native_forwards": 0,
               "freeze_created": False, "run_created": False}
    with args.output.open("x") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
