# Published 2-SAT checkpoint: feasibility and frozen development pilot

**Outcome.** The original published transformer is executable locally. A reproducible, nonzero-code witness shows that its published 34-neuron Boolean report does not determine a new clause-edit response, even when both original and edited checkpoint answers are correct. The frozen predictive pilot **did not establish an advantage for retaining the 34 post-ReLU activation magnitudes**. No protected final cases were generated or opened, no neuron selection or hyperparameter tuning occurred, and no follow-on mechanism experiment has run.

This is implementation and exploratory research evidence, not an independent manuscript review or an official external scholarly assessment. The earlier independent review is unchanged.

## Source and executable identity

The case is [Palumbo et al., *Validating Mechanistic Interpretations: An Axiomatic Approach*, ICML 2025](https://proceedings.mlr.press/v267/palumbo25a.html), [arXiv v2](https://arxiv.org/abs/2407.13594v2), using the [authors' repository](https://github.com/nilspalumbo/axiomatic-validation) at commit `35d1e4105265ab0ca74bc7daf9371fc8f82bba81`.

| Artifact | Identity |
|---|---|
| Published checkpoint | `models/layers_2_heads_1_4_1M/900.pth`, 1,635,210 bytes, epoch 900 |
| Checkpoint SHA256 | `d587446c64b7a8d68de93808ccf84a84364b85dcaf2149b8983b77895e24046d` |
| Model source SHA256 | `7555e1f2f27546225397aa12de4fde1f6c903e9804766de477fe4c02770691ea` |
| Frozen development contract SHA256 | `36a8f2763d4e299d19620db4e58a30e19788ae404f56939225f73cf95c489104` |

The cloned upstream checkout is unchanged and contains no `AGENTS.md`. Its MIT license is retained. `runtime.py` executes the exact original model-class ASTs while omitting unrelated plotting/training imports. It checks the complete source and checkpoint hashes, uses `torch.load(..., weights_only=True)`, and loads every state entry strictly. All 403,840 parameters and 21 state entries match. The CPU runtime uses Python 3.11.14, PyTorch 2.1.2, NumPy 1.26.4 and two threads. A separate existing analysis environment provides NumPy 2.4.6, SciPy 1.17.1 and scikit-learn 1.9.0. Both dependency sets are recorded. GPU arithmetic from the original training run is not reproduced bit for bit.

## Exact task grammar and training provenance

The model has two transformer blocks, one then four attention heads, width 128, ReLU MLP width 512, vocabulary 16 and input length 41. There is no layer-normalization operation in its forward computation. Each formula has five variables and ten clauses. The generator samples two distinct variables per clause, prints them in increasing variable-index order, and assigns independent signs. This gives 40 canonical clause types. Clause repetitions are allowed; tautological or repeated-variable clauses are outside this generated grammar.

The input is ten four-token groups `(`, literal, literal, `)`, followed by `:`. Positive literals have IDs 0–4, negative literals 5–9, parentheses 10 and 11, colon 12, SAT 13, UNSAT 14 and padding 15. A stored labelled row contains 42 tokens; inference drops the label. Primary correctness in this pilot requires full-vocabulary argmax exactly 13 or 14 as appropriate, which is stricter than treating every non-SAT argmax as UNSAT.

`train.log` records 2,000,001 rows: 1,000,000 SAT and 1,000,001 UNSAT. The seeded, label-balanced 60/40 split contains 1,200,000 training rows and 800,001 test rows. Source settings are AdamW, learning rate 0.001, weight decay 1, betas (0.9,0.98), ten-epoch linear warmup, batch size 10,000 and 1,000 epochs. The supplied checkpoint is epoch 900. The final-position cross-entropy helper computes float32 despite its high-precision name/comment.

The current generator notebook's training cell contains the syntax error `n_ones = 1kjjj000`; it cannot be executed verbatim to recreate training. The datasets are nevertheless publicly archived on [Zenodo](https://zenodo.org/records/11239102): analysis 67.2 MB and training 672.0 MB. Neither was downloaded or used here. The original fitted trees, means and analysis cache are absent from the repository. Rebuilding them would require a separately labelled recomputation. The distributed interpretation notebook directly records the 34 neuron IDs, which were used unchanged.

## Prior-art boundary

The original paper already validates component/prefix equivalence and replaceability. It studies different decision-tree capacities and alternative abstraction/reconstruction choices; its Appendix F includes random clause substitution. Appendix D derives downstream dependencies on clause multiplicities. Its first abstract state is the complete ordered clause list, which reconstructs the input and therefore has no unrestricted information-loss obstruction on this grammar. The source also includes previous-clause and colon-token context approximations.

This pilot instead asks whether the **final 34-bit report, computed before an input edit, can be reused to answer a new edit query**. The authors do not claim that property. The witness is an additional use-case limitation, not a refutation of their published tests, and this transformer is not the original Goodfire system. The symbolic closure of violated-clause counts is classical and is used only as an oracle/calibration.

## Contract and compatibility

`contract.md` was hashed and communicated before valid-formula inference. The 4,096 distinct development origins are all initially SAT, sampled from independent uniform canonical clauses conditional on satisfiability. Entire origins split into 3,072 fit and 1,024 assessment origins. All 390 single-clause edit labels are calculated exactly by enumeration of 32 assignments. Each predictor sees only its report and `(slot,old_clause,new_clause)`. Each decision independently selects among 39 alternatives within one stated slot and old-clause context. It never combines all ten old-clause menus, which would leak the whole formula.

Compatibility checks passed: strict parameter loading; canonical grammar/tokens; 3,208 independent Boolean checks on the first eight origins and all their edits; whole-model versus explicit three-stage decomposition; and a separate-process witness replay. Source/checkpoint and output hashes are retained. The first smoke check used an invalid-grammar all-zero row and only inspected tensor shapes and finiteness.

The checkpoint correctly classifies 4,073/4,096 original SAT formulas. Their published 34-bit reports have 1,428 distinct values. The frozen candidate search examines nonzero report collisions with the same action and threshold/logit-margin guards. Of its first 128 distinct origin pairs, 125 retain correct checkpoint classifications on both edited inputs. These are deterministic candidate-search results, not an estimated population prevalence.

## Replayed witness

Origins 1112 and 1623 have exactly the same satisfying assignment before editing: `10000`, in order x0,x1,x2,x3,x4. Their published report has only unit 10 active. Both begin with `(x0 OR ~x1)`. Replacing that first clause by `(~x0 OR x1)` yields a SAT formula with sole satisfying assignment `01111` for origin 1112 and an UNSAT formula for origin 1623. Thus the pair also shares the complete original 32-valuation truth table.

| Original | Unit 10 post-ReLU value | Other 33 report values | Edited exact status | Edited correct-label margin |
|---|---:|---|---|---:|
| 1112 | 0.7510638237 | all zero | SAT | 3.0279517 |
| 1623 | 1.6261007786 | all zero | UNSAT | 6.0506182 |

The smallest distance of a report activation from threshold 0.5 is 0.2510638. Independent-process float32 replay matches saved logits and activations exactly. Repeating the two originals and two edits in float64 preserves the report equality and all correctness claims. This is a numerical robustness diagnostic, not an interval or exact-real proof of the transformer.

The token-level Boolean verifier is independent of the vectorized count-update code. Full formulas, tokens, activations, logits, valuations and action are in `primary_witness.json` and `primary_witness.npz`; `replay_primary.py` recomputes the claim. The identical report/action but different response establishes that no deterministic decoder of this report/action can answer both cases correctly. The pair does **not** establish that unit 10's magnitude alone supports a general decision rule.

## Fixed predictive pilot, including its negative result

All learned reports used the same frozen histogram-gradient-boosting settings, 61,440 fit action rows and 399,360 assessment action rows. No second fit or tuning followed these results. There are 10,240 assessment menus, of which 8,433 have at least one UNSAT-producing edit.

| Report | Response AUROC | Log loss | Selected edit succeeds, all menus | Succeeds when an edit exists |
|---|---:|---:|---:|---:|
| Action context only | 0.5106 | 0.3375 | 11.494% | 13.957% |
| Published 34 bits | 0.9457 | 0.2574 | 62.451% | 75.833% |
| Published 34 magnitudes | 0.9405 | 0.2561 | 62.266% | 75.608% |
| Random 34 bits | 0.5106 | 0.3375 | 11.494% | 13.957% |
| Random 34 magnitudes | 0.5108 | 0.3375 | 11.865% | 14.408% |
| Exact truth table, 32 bits | 0.9630 | 0.2517 | 66.963% | 81.312% |

Retaining magnitudes slightly improves log loss, but worsens AUROC and selected-edit success relative to the Boolean report. The 0.1855-percentage-point selection difference favors bits. This pilot therefore does not support a claim that finer precision of these post-ReLU values repairs the edit limitation. Failure under this fixed decoder does not prove that all possible magnitude-based decoders fail.

The response distribution is 89.447% SAT. Almost every response is classified SAT at probability threshold 0.5, so ordinary accuracy and balanced accuracy are uninformative here: all models have roughly 89.45% accuracy and 0.5 balanced accuracy. The selection table is a forced-choice ranking result. At an UNSAT-confidence threshold of 0.5, the published-bit model acts on only 0.098% of menus, and the magnitude model on only 0.586%. The full coverage and accepted-decision results are retained in `predictive_metrics.json`; the ranking table must not be described as a calibrated, generally actionable policy.

The random units were fixed prospectively and match dimension/decoder capacity, but their threshold codes are all zero in this bank. They are therefore a weak activity control. Their poor performance does not establish specificity against an activity-matched alternative.

On the entire finite development bank, any deterministic decoder of the 34-bit report and action must make at least 25,188 errors among 1,597,440 semantic edit queries (1.5768%); the truth-table report has a corresponding 22,512-error minimum (1.4093%). These are finite-bank collision bounds, not population bounds and not bounds on neural responses outside checkpoint-correct cases. Exact violated-clause counts supply a zero-error symbolic oracle through the identity `q'_v=q_v−[v violates old]+[v violates new]`.

The prescribed checkpoint audit sampled 128 assessment menus. All original inputs were checkpoint-correct. For selected actions, both original and edited answers were correct in 127/128 menus for published bits, 127/128 for magnitudes, 126/128 for the truth-table baseline, 127/128 for random magnitudes and 128/128 for context/random bits. The audit and raw logits are in `selected_decision_check.*`. Headline semantic metrics remain unconditional.

## Untested next mechanism hypothesis

A source-informed hypothesis is that **pre-ReLU values in the same fixed 34 units** retain information about assignments that nearly satisfy the formula. Appendix D relates preactivations to clause multiplicities, and Appendix G connects units with assignment predicates. ReLU maps every negative preactivation to zero; changing precision of post-ReLU zeros cannot recover distinctions between mildly and strongly negative values. In the witness, the newly possible assignment `01111` was false in both original truth tables, making information about currently false assignments relevant to the new edit.

This is an hypothesis, not an outcome. No new pre-ReLU features were extracted or evaluated. The manuscript's mapping from units to assignments must be read exactly before a separate study. A credible follow-on would freeze the same units and action menus, compare pre- and post-ReLU reports with matched capacity and stated bit budgets, and test whether any benefit specifically tracks assignments one violated clause away from satisfying. Such a comparison could distinguish rectification loss from threshold-quantization loss. It must retain a null result and must not silently select new units or reuse this assessment split as protected confirmation.

## Reproduction and freeze

`runtime_smoke.json`, `pilot_generation.json`, `primary_witness.json`, `predictive_metrics.json` and `selected_decision_check.json` contain checks, versions, hashes and runtimes. Logs preserve the PyTorch typed-storage deprecation warning; it did not affect execution. The initial isolated dependency install encountered restricted-network DNS failure and succeeded through the authorized network approval path. All writes were confined to the workspace.

To replay the principal claim from this folder, use the pinned CPU environment to run `replay_primary.py`. Regeneration order in a fresh copy is `generate_pilot.py`, then `fit_pilot.py` in the pinned analysis environment, then `check_selected_decisions.py`. Generation and fitting refuse to overwrite existing completed artifacts. Include the unmodified upstream model/checkpoint/license when packaging; exclude `.venv`, `uv-cache`, `__pycache__` and `upstream/.git`.

The pilot is frozen at this point. No further fits, protected generation, neuron selection or mechanism experiments are authorized within this contract.
