"""Cached-only comparison primitives. No checkpoint loader or neural inference."""
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
MECH = ROOT.parent / "sat_mechanism"
PILOT = ROOT.parent / "sat_case"
DESIGN = ROOT.parent / "next_phase" / "sat_margin_baseline_contract.md"
CONTRACT_SHA = "bc330905a32edccb90d2ab511e5c2bc5714b7e6156d661085a7188ba131efdde"
MECH_MANIFEST_SHA = "6e5a46e1466339ccc08f5ce8b91a5cf5381bf3c99f789e3aa40d145b7f42b181"
PILOT_MANIFEST_SHA = "b81f234d4c20185955760ae01ffdfea3239b4c6c1d204666792819b315524e82"
UNITS = [10,41,55,77,81,185,224,245,261,291,317,326,380,411,416,492,495]
SCOPES = ["primary_assessment", "all_selected_assessment", "primary_fit", "all_selected_fit"]
CODE_FILES = ["common.py", "prepare_freeze.py", "selfcheck.py", "run_comparison.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    def native_scalar(item):
        if isinstance(item, np.generic): return item.item()
        if isinstance(item, np.ndarray): return item.tolist()
        raise TypeError(f"Unsupported JSON value: {type(item)}")
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False,
                                    default=native_scalar) + "\n")


def versions():
    assert (np.__version__, scipy.__version__, sklearn.__version__) == ("2.4.6", "1.17.1", "1.9.0")
    return dict(python=platform.python_version(), python_full=sys.version, executable=sys.executable,
                numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                platform=platform.platform())


def estimator():
    return HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.05, max_iter=200, max_leaf_nodes=7,
        max_depth=3, min_samples_leaf=5, l2_regularization=1.0, max_bins=255,
        categorical_features=None, early_stopping=False, random_state=2026101522)


def verify_originals():
    verified = {}
    for folder, expected in [(MECH, MECH_MANIFEST_SHA), (PILOT, PILOT_MANIFEST_SHA)]:
        assert sha(folder / "MANIFEST.json") == expected, f"Manifest changed: {folder}"
        manifest = read_json(folder / "MANIFEST.json")
        for name, entry in manifest["files"].items():
            assert sha(folder / name) == entry["sha256"], f"Prior artifact changed: {folder / name}"
        verified[folder.name] = dict(manifest_sha256=expected, files_verified=len(manifest["files"]))
    return verified


def verify_freeze():
    frozen = read_json(ROOT / "freeze.json")
    for name, expected in frozen["local_sha256"].items():
        assert sha(ROOT / name) == expected, f"Frozen implementation changed: {name}"
    for name, expected in frozen["input_sha256"].items():
        assert sha(ROOT / name) == expected, f"Frozen input changed: {name}"
    assert estimator().get_params() == frozen["estimator_parameters"]
    assert versions() == frozen["runtime"]
    return frozen


def design_matrix(pre, effect, context):
    """Exact float32-to-float64 conversion; [score,effect,slot,old,new]."""
    pre = np.asarray(pre)
    effect = np.asarray(effect)
    context = np.asarray(context)
    assert pre.dtype == np.float32 and pre.ndim == 1
    assert effect.shape == (len(pre),) and context.shape == (len(pre), 3)
    assert np.issubdtype(context.dtype, np.integer)
    assert np.isfinite(pre).all() and np.isin(effect, [0, 1]).all()
    assert ((context[:,0] >= 0) & (context[:,0] < 10)).all()
    assert ((context[:,1:] >= 0) & (context[:,1:] < 40)).all()
    x = np.zeros((len(pre), 92), dtype=np.float64)
    x[:,0] = pre
    x[:,1] = effect
    k = np.arange(len(pre))
    x[k,2 + context[:,0]] = 1
    x[k,12 + context[:,1]] = 1
    x[k,52 + context[:,2]] = 1
    assert np.array_equal(x[:,0].astype(np.float32), pre)
    assert np.isfinite(x).all()
    assert np.all(x[:,2:12].sum(axis=1) == 1)
    assert np.all(x[:,12:52].sum(axis=1) == 1)
    assert np.all(x[:,52:92].sum(axis=1) == 1)
    return x


def metrics(y, pred):
    y = np.asarray(y, dtype=bool)
    pred = np.asarray(pred)
    positive = int(y.sum()); negative = int((~y).sum())
    base = dict(n=len(y), positive=positive, negative=negative)
    if not np.isfinite(pred).all():
        return dict(base, tp=None, tn=None, fp=None, fn=None, sensitivity=None,
                    specificity=None, balanced_accuracy=None, reason="unestimable predictor")
    assert np.isin(pred, [0,1]).all()
    p = pred.astype(bool)
    tp = int((y & p).sum()); tn = int(((~y) & (~p)).sum())
    sensitivity = tp/positive if positive else None
    specificity = tn/negative if negative else None
    return dict(base, tp=tp, tn=tn, fp=negative-tn, fn=positive-tp,
                sensitivity=sensitivity, specificity=specificity,
                balanced_accuracy=(sensitivity+specificity)/2 if positive and negative else None,
                reason=None if positive and negative else "missing outcome class")


def strict_mean(values):
    return float(np.mean(values)) if values and all(v is not None and np.isfinite(v) for v in values) else None


def assignment_macro(values, rows):
    by = {a:strict_mean([values[i] for i,r in enumerate(rows) if r["assignment"] == a])
          for a in sorted({r["assignment"] for r in rows})}
    return dict(assignments=by, macro=strict_mean(list(by.values())))


def interval(values):
    values = np.asarray(values)
    valid = values[np.isfinite(values)]
    return dict(draws=len(values), valid_draws=len(valid), na_draws=len(values)-len(valid),
                interval_2_5_97_5=np.quantile(valid,[.025,.975]).tolist() if len(valid) >= 1900 else None)


def interpretation(delta, limits):
    if delta is None or limits is None:
        result = "UNRESOLVED_OR_UNSUPPORTED"
    else:
        lower, upper = limits
        if lower > .02: result = "ONE_BIT_LOSES_MATERIAL_ACCURACY"
        elif lower >= -.02: result = "CONTINUOUS_MEETS_DEVELOPMENT_NONINFERIORITY"
        elif upper < -.02: result = "THIS_COMPARATOR_MISSES_DECLARED_PARITY"
        else: result = "UNRESOLVED_OR_UNSUPPORTED"
    return dict(category=result, resolution=.02,
                readiness_meets_development_noninferiority=limits is not None and limits[1] <= .02,
                descriptive_parity=limits is not None and limits[0] >= -.02 and limits[1] <= .02)
