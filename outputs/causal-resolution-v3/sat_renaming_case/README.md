# Frozen SAT renaming study and saved-data replay

The completed recovery has status **CRITERION_NOT_MET**. The original attempt
stopped before any neural forward because its launched interpreter could not
import PyTorch. Both records are preserved separately. This supplement packages
existing evidence; its preparation generated no formulas, model outputs, fits or
scientific random draws.

For the fixed checkpoint and selected finite cohort, the first nominated unit
pair had 63/64 positive bits and 31/32 exact positive patterns; the second had
56/64 and 24/32, below the predeclared thresholds of 58/64 and 26/32. Both pairs
had 448/448 specificity. These author-chosen thresholds are not guarantees made
by the source paper. The two pairs are not a sample from a population of neurons.
The recovered run retained all 512 task rows and 2,048 unit events, with 512
initial and 16 separately replayed input forwards, and zero learner fits.

## Read and replay

Start with `work/sat_renaming_launch_recovery/REPORT.md` and `analysis.json`.
`work/next_phase/sat_renaming_execution_contract.md` is the frozen contract.
The original failed launch is in `work/sat_renaming/`; the recovery admission and
freeze explicitly connect it to `work/sat_renaming_launch_recovery/`.

With Python and NumPy installed, run from any working directory:

```sh
python3 -B /path/to/sat_renaming_case/replay_saved.py --output-dir /path/to/NEW-replay-output
```

The output directory must not exist and must be outside this supplement. The
helper verifies `PACKAGE_MANIFEST.json`, runs the unchanged independent Boolean
checker on **saved** formulas and ledger entries, then replays event, mapping,
task, saved-bootstrap, interval and secondary affine arithmetic. It uses the
2,000 saved bootstrap index arrays; it generates no random values. The checkpoint
is read only for SHA-256 identity. No checkpoint deserialization, PyTorch,
training, native inference, new formula generation, fitting or downloads occur.
It neither imports nor runs `run_study.py`, `recovery_driver.py`, `construct.py`,
`analyze.py`, or `native.py`.

The independently authored checker and original receipts are unchanged in
`archived_independent_audit/`. `checks/recompute_saved.py` is a separately labelled
portable adaptation. `checks/portable_adaptation.diff` records every change:
explicit paths, no RNG regeneration, and corresponding scope fields. Numerical
comparisons are unchanged. The archived audit checked the declared RNG stream;
the default portable replay does not repeat that check. Model-generated review
is not independent investigators' reproduction or peer review.

## Component layout and provenance

- `work/sat_renaming/`: complete original failed attempt, frozen code and bank.
- `work/sat_renaming_launch_recovery/`: complete admitted recovery, all 66 saved
  batches, readouts, tables, indices, manifests and journal.
- `work/sat_case/` and `work/sat_mechanism/`: exact prior files required by the
  freeze and known-input exclusion ledger, including canonical model source and
  checkpoint. They are not complete copies of the earlier studies.
- `work/next_phase/`: all frozen design, contract and review references, plus the
  separate recovery admission.
- `FROZEN_DEPENDENCIES.json`: all 33 original freeze references resolved locally.
- `SOURCE_COPY_MAP.json`: source-to-package paths, byte lengths and hashes.

The earlier source is [Palumbo et al., ICML 2025](https://proceedings.mlr.press/v267/palumbo25a.html)
and the [authors' repository](https://github.com/nilspalumbo/axiomatic-validation)
at commit `35d1e4105265ab0ca74bc7daf9371fc8f82bba81`. Its retained MIT license is at
`work/sat_case/upstream/LICENSE`. Frozen model source/checkpoint identities are
resolved in `FROZEN_DEPENDENCIES.json`. Files copied from the existing packaged
release were selected only when they matched the required frozen hashes.

Historical study READMEs may describe a then-pending freeze. They remain exact
copies and do not override the final reports. Original execution and analysis
scripts are archival: invoking them can perform a new study or overwrite/fail on
existing artifacts and is not part of this replay. Requirements for those
original runtimes are archival too; the safe replay requires NumPy only.

Passing this replay establishes the stated saved-data identities and arithmetic.
It does not rerun the neural experiment, prove interpretation correctness on new
inputs, regenerate the proposal pool or establish generality. The secondary
affine clamp is arithmetic on saved arrays, not patched-native execution or an
upstream mediation result. Numerical sensitivity intervals are engineering
conventions, not proven floating-point error bounds.
