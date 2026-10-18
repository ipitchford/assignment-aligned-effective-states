"""Archive code, cached-input identities and fixed learner metadata before fitting."""
import datetime
import inspect
import os
import shutil
from pathlib import Path
import threadpoolctl
import sklearn.ensemble._hist_gradient_boosting as hgb
from common import *


def main():
    assert not (ROOT / "freeze.json").exists(), "Already frozen"
    assert not (ROOT / "execution.json").exists(), "Execution already started"
    assert sha(DESIGN) == CONTRACT_SHA
    originals = verify_originals()
    shutil.copyfile(DESIGN, ROOT / "contract.md")
    runtime = versions()
    library = ROOT / "library_source"
    library.mkdir(exist_ok=False)
    source_root = Path(inspect.getfile(hgb)).parent
    modules = []
    # Python source retained; compiled-extension identities retained with versions.
    for path in sorted(source_root.iterdir()):
        if path.is_file() and path.suffix in [".py", ".pyx", ".pxd", ".so"]:
            item = dict(name=path.name, path=str(path), bytes=path.stat().st_size, sha256=sha(path))
            if path.suffix in [".py", ".pyx", ".pxd"]:
                shutil.copyfile(path, library / path.name)
                item["archived"] = str((library/path.name).relative_to(ROOT))
            modules.append(item)
    with threadpoolctl.threadpool_limits(limits=1):
        pools = threadpoolctl.threadpool_info()
        assert all(p["num_threads"] == 1 for p in pools)
    thread_settings = dict(limit=1, mechanism="threadpoolctl.threadpool_limits(1)",
        environment={key:os.environ.get(key) for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "PYTHONDONTWRITEBYTECODE"]},
        observed_pools=pools, threadpoolctl_version=threadpoolctl.__version__)
    write_json(ROOT / "library_identity.json", dict(modules=modules, runtime=runtime, thread_settings=thread_settings,
        estimator_parameters=estimator().get_params(), neural_imports=False, fits_performed=0))
    inputs = ["../next_phase/sat_margin_baseline_contract.md", "../sat_mechanism/MANIFEST.json",
              "../sat_mechanism/stage_b_raw.npz", "../sat_mechanism/bootstrap_raw.npz",
              "../sat_mechanism/selected_queries.json", "../sat_mechanism/source_mapping.json",
              "../sat_mechanism/stage_a.json", "../sat_mechanism/stage_b.json", "../sat_mechanism/cost_ledger.json",
              "../sat_mechanism/protocol.py", "../sat_case/pilot_common.py", "../sat_case/MANIFEST.json"]
    local = CODE_FILES + ["contract.md", "implementation_notes.md", "selfcheck.json", "library_identity.json"]
    local += [str(p.relative_to(ROOT)) for p in sorted(library.iterdir())]
    write_json(ROOT / "freeze.json", dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status="FROZEN_BEFORE_ANY_COMPARISON_FIT", contract_sha256=CONTRACT_SHA,
        local_sha256={name:sha(ROOT/name) for name in local}, input_sha256={name:sha(ROOT/name) for name in inputs},
        runtime=runtime, estimator_parameters=estimator().get_params(), thread_settings=thread_settings,
        prior_manifests_verified=originals, maximum_fits=17, neural_forwards=0, new_formulas=0,
        protected_cases_accessed=0, assessment_passes=1))
    print(json.dumps(dict(freeze_sha256=sha(ROOT / "freeze.json"), fits_performed=0, neural_forwards=0)))


if __name__ == "__main__": main()
