# SAT renaming: implementation awaiting authorized freeze

This directory implements the fixed contract in `../next_phase/sat_renaming_execution_contract.md` (SHA-256 `4b9dc2df3002dbbe56cbcc5d38c119ace801f16e2f3bd018d345deb4eda7af8c`). No result is implied by the existence of the implementation. The reviewed draft remains unchanged.

The narrow question is conditional source-fixed assignment transport for units 261↔291 and 411↔492 under x1↔x2, with variables indexed x0 through x4. One matched block contains a unique-SAT repair and an edited-UNSAT control with the same action, each in original and renamed form. All four fixed units are measured on all eight inputs. Equal semantic labels do not equalize native confidence; semantic scalar-margin implementations remain compatible with success.

## Execution boundary

Do not run preparation or scientific entrypoints until the complete implementation has been reviewed and root has authorized the freeze. After that authorization, preparation hashes every code file, the contract and admissions, independent oracle/checker, original encoder/runtime/model/checkpoint, and all five prior input files. It builds the historical ledger using existing inputs only. Missing files prevent the freeze.

The historical ledger has six array sources: pilot `tokens`, mechanism `unique_tokens`, candidate-logit `edited_clauses`, decision-check `edited_clauses`, and witness `original_tokens`/`edited_tokens`. Clause arrays use the unchanged encoder. Token identity is exactly 41 raw uint8 bytes, irrespective of storage dtype. Ledger rows are deduplicated and sorted bytewise. The invalid-grammar all-zero smoke input is recorded separately and cannot equal a valid selected formula.

After root authorizes execution, the entrypoints are:

```sh
/Users/admin/.venvs/gaia-tess-microlensing/bin/python -B prepare_freeze.py --authorize-freeze
/Users/admin/.venvs/gaia-tess-microlensing/bin/python -B run_study.py --execute
```

Run from this directory. The runner uses the frozen analysis interpreter and the existing `../sat_case/.venv/bin/python` native interpreter. It does not install dependencies, modify an environment, resume, retry, or overwrite any artifact. Each stage checks the implementation freeze; native and analysis stages also check the selected-data freeze. Partial batch results and attempted-forward charges are retained on failure.

Construction has one PCG64 seed, at most 100,000 candidates and a 20-minute cap. It must produce 32 distinct matched descriptors per pair. An insufficient pool stops before model loading. Native extraction is 64 eight-row batch calls plus two fixed replay calls in separate processes, at most 528 input forwards and ten total minutes. Analysis has 2,000 fixed block-bootstrap draws, no fits and a five-minute cap. Selection never reads model outputs. There is no permission to substitute a smaller feasible bank.

## Exact selection encoding

The clause array is the unchanged encoder's 40-clause order, restricted to its 36 IDs whose unordered variable pair is not {1,2}. The proposal call is `Generator(PCG64(2026101604)).integers(0,36,size=(100000,10),dtype=int16)` followed by lookup in the saved restricted IDs. Candidate formulas generated twice retain the first trial index. Candidate/selected arrays retain original 0..39 clause IDs.

Every hash starts with its ASCII domain and one zero byte. Each subsequent field is encoded as its eight-byte unsigned big-endian byte-length followed by raw bytes. Pair IDs are one byte 0/1. An action is five bytes `(slot,old_clause_id,kept_literal,new_variable,new_clause_id)`. Formula identity is its 41 token bytes. A canonical orbit is the bytewise minimum of original and renamed tokens; no anchor orientation is changed by this identity convention.

The action-minimum domain is `sat-renaming/action-v1` and its fields are `(pair,orbit,action)`. Ties use the five-integer action tuple. Descriptor ordering uses `sat-renaming/descriptor-v1` with `(pair,action)` and then the action tuple. Candidate-origin ordering uses `sat-renaming/origin-v1` with `(pair,action,cohort,orbit,original_tokens)`, cohort byte0 positive/1 negative, then trial ID. One eligible action is chosen per origin/pair before matching. Each descriptor uses its first positive candidate and the first compatible negative; an unusable first positive or exhausted negative list skips that descriptor without choosing another positive/action. P1 and P2 selection alternates for 32 rounds. Global endpoint/orbit and complete historical-ledger overlaps are rejected.

## Saved data interface

`selected_bank.npz` has `clauses[64,2,4,10]`, `tokens[64,2,4,41]`, `counts[64,2,4,32]`, `labels[64,2,4]` (SAT13/UNSAT14), `expected_unit_bits[64,2,4,4]`, `pair_index[64]` and `trial_ids[64,2]`. Cohorts are positive then negative. States are original, edited, renamed original, renamed edited. Unit order is `[261,291,411,492]`. Flattening each block in C order gives the native eight-row batch. The first two blocks, one per pair, are the replay batches.

The independently authored `independent_oracle.py` and `verify_bank.py` rederive truth, count, action, renaming, uniqueness, identity, source-ledger and denominator checks without the vectorized sampler oracle. Their PASS receipt is required before `selected_data_freeze.json` is written. Candidate and partial-selection diagnostics are retained even on shortfall.

Native full postactivation/residual arrays exist solely for the fixed affine parity check; they must not be used for an unregistered hidden-unit search. `analyze.py` scores the source-labelled bits, full16-way classes, fixed correct/identity/wrong maps, signed changes, paired block uncertainty and conservative numerical bounds. Each mapping contrast is enumerated jointly so shared endpoint bits remain shared. Off-pair FPR bounds invert TN bounds. The same saved bootstrap indices propagate numerical bounds. Primary sensitivity crossings produce an unresolved benchmark; secondary affine parity failure affects only the clamp interpretation.

The bootstrap target is the equal-weight mixture of selected descriptors under this selection algorithm. Matching and deduplication do not establish independent random blocks. No result identifies a unique mechanism, excludes arbitrary assignment-selective margins, revives the stopped baseline comparison, or establishes a broader generality.
