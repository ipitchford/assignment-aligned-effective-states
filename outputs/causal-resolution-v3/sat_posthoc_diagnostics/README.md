# Portable saved-data SAT diagnostics

These eight exploratory analyses answer decision-changing questions raised during
review of the manuscript. They operate on the exact retained arrays in `inputs/`,
make zero new transformer forwards, and do not alter the protected status of the
earlier SAT-renaming execution.

## Components

- `sat_information_audit.py` reproduces the finite-bank lookup bounds and
  separates logical target ambiguity from report-specific loss.
- `report_collision_sensitivity.py` repeats that decomposition on deterministic
  nested origin banks and by the number of active published bits.
- `action_menu_encoding_factorial.py` crosses two menu regimes with one-hot and
  natural-binary edit encodings and reproduces the original pilot cell.
- `readiness_semantic_baseline.py` compares the frozen readiness rule with the
  declared semantic-action-effect baseline under the saved 2,000 resamples.
- `renaming_threshold_audit.py` reports each direction's score distribution,
  AUROC, zero-false-positive sensitivity and restoration cross-tab.
- `clamp_pair_null.py` enumerates all 130,816 unordered final-block unit pairs on
  the two source-matched positive cohorts.
- `direction_specific_clamp_null.py` ranks every single unit and unordered pair
  separately for each of the four 32-endpoint construction directions.
- `renaming_specificity_and_causal_controls.py` stratifies the negative cohort,
  performs converse single-unit patch-in, selects controls from activation-change
  profiles and audits unit 201 as a compensation candidate for unit 411.

Run the scripts from this directory with Python 3. The retained full environment
is recorded in `requirements.txt`; only the action-menu factorial imports SciPy
and scikit-learn, while the other seven scripts require NumPy alone. Each script
writes its named JSON, Markdown and, where applicable, NPZ output beside itself.
Existing outputs are the retained reference results; use a copied directory if
preserving their bytes matters. All eight packaged replays passed after the path
adaptation; the factorial used the retained 2.4.6/1.17.1/1.9.0 stack and the other
seven also passed under NumPy 2.4.2.

## Interpretive boundary

The lookup analyses quantify finite-bank occupancy and do not estimate a
population failure rate. The action-menu result depends on representation and one
learner seed. The readiness comparison does not equalise supervision. The
renaming negatives contain no multi-solution SAT case. Patch-in and restoration
establish source-aligned effects at the saved final affine readout, under
potentially off-manifold interventions; they do not identify an upstream route.
Qualified activation-change controls exist for only two original directions.

`MANIFEST.json` records the SHA-256 digest and size of every other packaged file.
