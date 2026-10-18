# Protected all-unit SAT label validation protocol

## Question and boundary

This study tests the seventeen published final-block units whose general rule is a
single satisfying assignment in Palumbo et al. (2025). For a nominated unit
`u_a`, the published Boolean abstraction predicts activation at the source
threshold 0.5 when and only when assignment `a` satisfies the input formula. The
primary test is threshold-free discrimination across five semantic strata. A
secondary bank applies every variable transposition whose transformed assignment
also has a published singleton unit.

The study does not test Cammarata's prevalence claims, recursive development,
architectural equivariance, a population of checkpoints, or a natural upstream
causal route. It does not treat raw activation equality under renaming as a source
prediction. The source rules are approximate and distribution-dependent.

## Fixed population and strata

The source mapping contains seventeen units and sixteen distinct assignments;
units 55 and 416 both map to `TTTFF`. For each distinct assignment, select exactly
32 formulas in each stratum:

1. `sole_positive`: the assignment is the formula's unique solution;
2. `multi_positive`: it satisfies the formula and the formula has at least two solutions;
3. `unsat_negative`: the formula has no solution;
4. `sole_negative`: the formula has one different solution;
5. `multi_negative`: the formula has at least two solutions, excluding the assignment.

For each SAT-negative stratum, select 16 formulas whose closest satisfying
assignment is at Hamming distance one from the target and 16 whose closest
satisfying assignment is at distance at least two. This fixed near/far balance
tests whether a unit merely responds to a neighbourhood of its proposed label.

Formulas contain ten independently sampled clause IDs from the complete native
40-clause grammar, including repetition as in the source generator. Selection
uses a fixed PCG64 stream of 250,000 proposals, seed 2026101807, and a
domain-separated SHA-256 order. It reads no checkpoint, activation, logit, prior
study outcome or fitted model. Every selected token sequence must be absent from
the complete prior native-query ledger, including the earlier renaming study.
Within an assignment, no formula may fill two strata. Reuse across assignments is
allowed and retained as dependence, because one formula may deliberately test
several labels.

## Secondary transposition bank

Use all ten swaps of two variables. For every directed assignment edge `a -> b`
where both assignments have a published singleton unit, scan the frozen 32-case
base stratum in its fixed hash order and retain the first eight cases whose
transformed formula differs from the original and is absent from the historical
ledger. Do not replenish after native outcomes. The transformed formula must
belong to the corresponding semantic stratum for `b`; otherwise construction
stops before inference. Fixed-assignment edges test surface-renaming stability;
non-fixed edges test label transport. Duplicate TTTFF units are evaluated
separately and are not forced into a unique hidden-unit permutation.

## Outcomes and criteria

The primary score is exact pairwise AUROC of preactivation for each of seventeen
units on its 64 positive and 96 negative target-specific base cases. Report each
unit, the mean across the sixteen assignment summaries after averaging units 55
and 416, the median, range, and 2,000 formula-orbit-clustered bootstrap draws
(seed 2026101808), stratified by the assignment-independent solution-count
classes zero, one, and two-or-more. A cluster is the complete orbit under all 120
variable permutations. Also report at the source threshold `preactivation >= 0.5`, separately by stratum, every
numerator/denominator for sensitivity or false-positive rate.

The protected result supports **broad assignment-label discrimination** only if:

- the sixteen-assignment macro AUROC is at least 0.95;
- every unit AUROC is at least 0.80;
- at least 15/17 units have AUROC at least 0.90; and
- no unit has source-threshold false-positive rate above 0.10 on the
  multi-solution negative stratum.

This broad-discrimination gate set combines three threshold-free ranking
requirements with one source-threshold specificity guard on multi-solution
negatives; it is not full validation of the source threshold. Separately,
source-threshold calibration is supported only if
every unit detects at least 24/32 cases in each positive stratum, every unit has
at most 3/32 false positives in each negative stratum, and at least 15/17 units
have pooled threshold balanced accuracy at least 0.85. These gates are
study-defined and are not guarantees from the source paper.
Failure must be reported by unit and stratum; labels, thresholds, quotas,
transpositions and strata may not be changed after responses.

For each unit, report the sole- and multi-solution negative strata separately at
near distance one and far distance at least two: threshold false-positive
numerators/denominators and descriptive AUROCs against the pooled 64 positives.
These near/far summaries are mandatory descriptive specificity diagnostics, not
additional protected gates.

On the secondary bank, compute one pooled per-unit transformed AUROC over distinct
unit/input events. Split edge-weighted threshold counts into fixed-assignment and
assignment-switching edges; unit 41 (`FFFFF`) has no switching edge and that cell
is structurally not applicable. Report base-minus-transformed AUROC descriptively.
Do not use equality of raw scores, McNemar tests
between different units, or ordinary accuracy pooled across the imbalanced
semantic events as primary evidence.

The full sixteen-way task prediction is retained for every formula and reported
by semantic stratum. The published source defines the five-variable SAT token as
`2*nvars+3 = 13` and UNSAT token as `2*nvars+4 = 14`; `analyze.py` asserts this
mapping, while every execution stage verifies the hashes of upstream `helpers.py`,
`interpretation.py`, and the recovered primary-table unit mapping. Those sources
also bind the `x0`-most-significant-bit assignment convention and all seventeen
unit/assignment pairs. Unit-label analysis includes task failures. Exact source-
threshold distance and winning-logit gaps are retained. Eight replay formulas are
fixed by construction before inference and must reproduce threshold bits and
argmax labels under a separate model load.

## Dependence, computation and stopping

The full variable-permutation orbit is the dependence unit. Transposed copies and
appearances under several assignment-specific strata remain linked by stored
proposal and orbit identifiers. Bootstrap resampling is descriptive and
solution-count-stratified; it is not an
independent checkpoint or neuron-population analysis.

Construction may draw at most 250,000 formulas. Native execution may evaluate
only the deduplicated selected base and transformed token rows plus the eight
fixed replay rows, on the unchanged epoch-900 CPU float32 checkpoint with two
threads. No learner is fitted. Any quota shortfall, semantic mismatch, historical
overlap, environment drift, non-finite output, replay-bit mismatch or missing
analysis cell stops the study without relaxing the contract.

Even if every discrimination gate passes, the conclusion is restricted to one
checkpoint and this balanced conditional mixture. It does not validate the full
source threshold unless the separate calibration criterion passes; it does not
establish equivariance, specificity outside the declared near/far-balanced cohort,
prevalence, causal role, or training-set exclusion.

The frozen packet and review are local chronology evidence, not external
preregistration. Model-generated static review is not human peer review.
