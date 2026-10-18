# Evidence guide

Read `paper.pdf` for the central case and `supplement.pdf` for supporting proofs,
calibrations and earlier studies. The author is Anonymous. This is an unrefereed
case study of one checkpoint and an engineered input bank.

## Trace the main findings

Paths below are relative to the extracted release directory.

| Paper finding | Evidence | Interpretation boundary |
|---|---|---|
| Seventeen published singleton labels rank strongly, while the frozen combined and threshold-fidelity decisions fail | `work/next_phase/sat_all17_validation_v3_20261018/protocol.md`, `freeze.json`, `analysis.json`, and `posthoc_diagnostics_v2.json` / `posthoc_diagnostics_v3.json` in that directory | The protocol and primary result are locally protected; later error localisation is retrospective. The bank is not a sample of checkpoints or a natural input distribution. |
| The source-nominated 34 units constitute the bank-effective state; their assignment alignment exceeds fixed rotations and mixtures | `work/next_phase/sat_full_layer_basis_v7_20261018/analysis.json`, `posthoc_diagnostics_v2.json`, and `effective_live_state_analysis_v2.json` | The full-space comparison is protected. Effective-state localisation and geometry controls are retrospective. Fixed rotations are reference ensembles, not exchangeability-based significance tests. |
| Learned NMF slightly improves mean assignment matching while recovering most native axes | `work/next_phase/sat_full_layer_basis_v7_20261018/effective_state_followup_v6.py` and `effective_state_followup_v6.json` | These decompositions are fitted and scored on the same bank. The five seeds assess numerical stability, not a model population. V6 corrects the earlier ICA reconstruction. |
| The final affine readout adds positive SAT evidence; grouped removals cross decision margins | `work/next_phase/sat_full_layer_basis_v7_20261018/analysis.json` and `effective_state_followup_v4.json` | Final-cut logit accounting is distinct from upstream causal routing and from nonlinear changes in the argmax decision. |
| Refitted bits give 64 errors and continuous activations 26 under matched standardisation and nested regularisation selection | `work/next_phase/sat_full_layer_basis_v7_20261018/effective_state_followup_v6.py` and `effective_state_followup_v6.json` | Both use orbit-grouped outer folds and training-only scaling/selection. The unscaled sensitivity gives 64 versus 21 errors. These state-only predictors differ from fixed-native-readout substitution, which retains residual information. |
| Current truth and the published report do not determine every legal edit response | `work/next_phase/sat_review_repairs_20261018/sat_information_audit.py` and `sat_information_audit.json`; original source materials under `outputs/causal-resolution-v3/sat_case/` | An exact witness establishes failure of universal sufficiency for this target; it does not refute a source claim about a different target. |

## Supporting evidence

- The exact affine and nonlinear calibration checkers are under
  `outputs/causal-resolution-v3/certificates/` and `nonlinear_certificates/`.
  `matched_controls/verify.py`, `prospective_theory/verify.py` and
  `prospective_application/verify.py` cover their separate, stated contracts.
- `outputs/causal-resolution-v3/src/verify_bundle.py` runs nine supporting
  checks in a fresh copy. They are part of the public replay's 18 stages (five image-dependent historical stages are unavailable).
- The preceding SAT-renaming and Goodfire-export audits have scoped portable
  checkers in `outputs/causal-resolution-v3/sat_renaming_case/` and
  `goodfire_training_audit/`.
- The deterministic visual construction is under
  `work/next_phase/fixed_seam_sham/`. Its numerical and saved-pixel checks pass;
  the separately retained model-assessor reports judged the visual stimuli
  ambiguous. This avenue stopped without an edited-image neural forward pass.

## Replay and review records

`reproduce.py` checks all released file identities, creates a new workspace,
executes all 7,252 retained inputs through the packaged checkpoint, regenerates
the main analyses and runs the supporting checks. It does not retrain a model
or create a new experimental population. See `README.md` for pinned runtimes.

`verification/candidate-replay/` retains the completed candidate replay's logs,
regenerated results and fresh native outputs. Its receipt identifies that
candidate's manifest. The separate verification receipt delivered beside the
final ZIP binds the final archive and its own clean-extraction replay.
The original extracted inputs remain unchanged; declared zero timestamps for
the blinded visual packet are reconstructed only in the disposable copy because
ordinary ZIP extraction cannot preserve them.

`model_generated_reviews/` contains the complete submitted requests, input
hashes, returned text, model identity, token usage and completion status. Those
records document model-generated technical criticism. They are not human peer
review or independent experimental replication.

The current reader-facing paper and supplement are authoritative for the claim
scope. Earlier diagnostic versions are retained to expose the correction history,
not as competing current estimates. `effective_state_followup_v6.json` supersedes
V5's ICA reconstruction and unmatched-preprocessing readout comparison. Internal
editorial commentary omitted from four unfrozen research memos is recorded in
`editorial_omissions.json`; frozen scientific records are preserved.


## Corrected-release additions

The machine-readable claim_evidence_map.json supplies validated JSON-pointer links for occupancy/activity, source-rule linkage and pair/union ablation counts. nmf_revision_diagnostic.json contains all five per-assignment comparisons and every axis anchor count/fraction. The environments directory records the original 23-stage environment assignment; the public runner executes 18 of them to installed-package snapshots. source_verification.json records licence and preprint access checks. revision_response.md documents the full-review corrections.
