# Frozen-ready protocol for the cached task-policy screen

This implementation realizes the development screen in
`work/next_phase/task_policy_augmentation_admission.md`. It is not an execution
record and does not authorize fresh native inference.

For each of the 4,096 cached initially satisfiable origins, choose one clause
slot by interpreting the first eight bytes of
`SHA256("already-clean-task-policy-augmentation-v1\0" || canonical-token-bytes)`
as an unsigned big-endian integer modulo ten. The 3,072 fit origins and 1,024
assessment origins use the contiguous split prospectively fixed in the original
`work/sat_case/contract.md` and implemented in the bound
`work/sat_case/fit_pilot.py`: indices 0--3,071 are fit origins and 3,072--4,095
are assessment origins. The freeze binds those two source files as evidence; it
does not reopen `predictive_outputs.npz` to infer a partition. Each chosen slot
supplies all 39 alternative clauses, giving 119,808 fit and 39,936 assessment
rows.

The action descriptor is 16 fixed bits: four-bit natural-binary slot ID,
six-bit natural-binary old-clause ID and six-bit natural-binary new-clause ID.
The report is computed per origin before this descriptor is appended.

The seven learned arms are context-only, published 34 bits, the 68-bit negative
score candidate, 68-bit four-bin preactivation, 68-bit four-bin postactivation,
continuous 34-value preactivation and continuous 34-value postactivation. Every
model input reserves 68 report columns plus the common 16 action columns. Missing
report columns are exact zeros. This equalizes learner geometry, not information
cost.

Negative thresholds are lower medians of strictly negative fit preactivations.
The candidate's added bit is zero if a unit has no negative fit value. Four-bin
preactivation then uses minus infinity for that unit's lower boundary. Upper
postactivation thresholds are lower medians conditional on values at least 0.5,
or exactly 0.5 if that set is empty. Thresholds retain float32 bytes and flags.

Each arm uses one `HistGradientBoostingClassifier`: 100 iterations, at most 15
leaves, learning rate 0.08, L2 regularization one, 63 bins, no early stopping,
random seed 2026101606 and no class reweighting. The response is semantic UNSAT.
No hyperparameter, threshold, feature order or arm may change after the freeze.

Actions rank by predicted UNSAT probability, with clause ID breaking exact ties.
Report forced-choice success, reachable-menu success, regret, and coverage and
conditional success at 0.5 and 0.8. At 0.8, utility is +1 for success, -4 for
failure and zero for abstention. The exact semantic control chooses the lowest
successful clause, or the lowest clause if none succeeds.

For every report arm, also compute the exact finite policy ceiling over classes
with identical report bytes, slot and old clause. Within each class, select the
clause with the largest number of semantic successes. This yields the exact
minimum reachability regret for a deterministic policy on the assessment bank
and diagnoses report-information loss separately from the learned policy's gap
to that ceiling. It is not a population bound. The bound and its set-intersection
criterion are fixed in `work/next_phase/set_valued_policy_sufficiency.md`.
Every ceiling receipt also records the complete class-size histogram, singleton
fraction and reachable-member class-size histogram so fragmentation and
near-singleton saturation can be assessed without reopening a research cache.

Development GO requires the candidate to exceed both published bits and
four-bin postactivation by at least five percentage points, lie no more than two
points below four-bin preactivation, and have at least ten percent coverage and
positive mean utility at 0.8. Its exact finite policy ceiling must also exceed
the published-bit and four-bin postactivation ceilings by at least five
percentage points on all assessment menus. Failure stops this branch. Passing
permits only a separately reviewed prospective design; it does not itself
authorize native work.

`prepare-freeze` requires an independent static-admission document outside the
implementation directory. The document must contain `DECISION: ADMIT_STATIC`
and transcribe the exact hashes of all bound code and governance inputs. The
freeze binds that admission hash, this protocol, all executable code, the
immutable cached inputs, the admission card, cache-alignment audit, theory note,
prospective claim ladder, decisive contribution bridge, literature boundary,
authoritative static receipt, runtime-smoke receipt, implementation handoff,
original SAT contract and original pilot implementation.
Execution refuses if any bound code, governance or evidence document changes.
`execute` refuses absent or changed bindings and writes into a new `run/`
directory. Static checks use synthetic arrays only and open no research caches.
Execution also requires the recorded Python 3.11.14, NumPy 2.4.6, SciPy 1.17.1
and scikit-learn 1.9.0 analysis stack; a version mismatch stops before fitting.
Before constructing any research design, execution checks the declared token,
clause, semantic-outcome and native-array shapes, the Boolean outcome dtype,
and clause range. It records all five numerical thread environment values in
the calculated result.
The executable arms a 1,800-second process alarm and checks process peak resident
memory against 4 GiB before and after each fit/prediction. It counts realized
trees and leaves from the fitted estimators and stops above 700 trees or 10,500
leaves in total. It rechecks the caps after exclusive serialization and output
hashing. `terminal.json` is committed through an fsynced, exclusively linked
pending file and is the sole completion record containing the final observed
resource peak; a calculated `result.json` is explicitly uncommitted
until that terminal record exists. On failure, `error.json` retains the stage,
arm, attempted and completed fit counts, available metrics and ceilings, model
sizes, versions and final observed resources. These boundary checks audit the
declared cap; they do not prevent every transient allocation at the OS level.
`runtime_smoke.py` performs one three-tree fit on generated 84-column Gaussian
data solely to verify the pinned library API and realized-tree accounting. It
does not open either research cache and is not a development result.

