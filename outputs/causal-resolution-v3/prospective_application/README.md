# Prospective report audit

Read `report.md` for the study, `analysis_contract.json` for its declared interface and status, and `all_metrics.csv` for all 84 model/bank/method rows. “Prospective” means that the prediction uses the report before the action. This study was designed after previous results on the same dataset split had been inspected; it is exploratory and is not a new untouched test.

The complete runnable bundle must retain this directory beside `application/` and `nonlinear_application/`. The exact external model, report and data dependencies are hashed in `MANIFEST.json`. The independently developed structural results are in sibling `prospective_theory/`; their files are not required to replay this audit.

From the containing research workspace, with Python 3, NumPy and Matplotlib available:

```sh
OPENBLAS_NUM_THREADS=1 python3 work/prospective_application/decision_certificates.py
OPENBLAS_NUM_THREADS=1 python3 work/prospective_application/verify.py
MPLCONFIGDIR=work/application/.mpl-cache OPENBLAS_NUM_THREADS=1 python3 work/prospective_application/report_and_figures.py
python3 work/prospective_application/build_manifest.py
```

The first command enumerates all deterministic action policies on each report/context fibre; the second independently rebuilds exact responses and verifies predictions use only the permitted inputs. It is a separate arithmetic implementation by the same team, not independent scientific replication. The third rebuilds descriptive tables and the fixed two-image figures. Rebuilding PDF files may alter metadata and file hashes without changing their visual content.

`audit.py` builds the exploratory encoders, fits the constrained decoders and records all evaluated outcomes. It refuses to overwrite a directory containing `results.json`. To repeat the fitting analysis, create a clean sibling tree containing copies of the original `application/` and `nonlinear_application/` dependencies, place `audit.py` in its empty `prospective_application/` directory, and run that copy. Preserve the original results and manifests. No neural model is retrained by this script.

The exact same-report classification witness uses shallow seed 43, images 1305 and 253. Only action 2 flips the first from 8 to 1; only action 3 flips the second. The seed 143 witness is a different result: it certifies distinct best actions for the fixed 1-minus-8 margin, while action 1 changes both full predictions through other classes. The figures and JSON distinguish these claims.
