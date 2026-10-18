# Exploratory nonlinear discovery expressivity certificates

This is a **separate exploratory supplement** to the frozen nonlinear architecture study. It uses only the existing discovery bank and fixed native-eight-predicate report for each model. It performs no model training, report-feature selection, or test-outcome access. It does not amend the earlier independent review or its input snapshot.

For each model, retain the first discovery case for each distinct native-eight-predicate signed-difference vector. All three resulting subbanks contain 15 distinct reports. The unrestricted empirical lookup error is therefore zero. An affine decoder includes an intercept and the eight signed-difference columns. Exact signed contrasts nonetheless prove the following lower bounds on its simultaneous-output maximum error:

| Seed | Cases in witness | Output coordinate (zero-based) | Exact lower bound | Approximation |
|---:|---:|---:|:--|---:|
| 117 | 4 | 5 | `42493082332223/21990232555520` | 1.9323616622 |
| 129 | 4 | 2 | `25218001206429/8796093022208` | 2.8669548108 |
| 143 | 4 | 5 | `21485656030733/10995116277760` | 1.9541090324 |

Each bound exceeds the declared tolerance one. Each record also retains a dyadic primal decoder for **all ten outputs**, with a joint exact upper bound less than `10^-7` above the lower bound. Thus the diagnostic is a tight finite-subbank affine minimax bracket, not merely a failed fitted predictor. It is not a prevalence estimate or a held-out result.

Each `seed-*.json` is self-contained for the numerical claim: it includes all 15 design rows, all exact response numerators and their denominator, the sparse rational witness, mappings to original discovery rows and intervention metadata, all ten primal coefficient vectors, and every exact upper bound. The source bank is identified by SHA-256.

`generate.py` calls the existing affine-audit LP routines to propose witnesses and coefficients, then uses exact arithmetic to export the bounds. `verify_exact.py` is a separately written **standard-library-only** checker of the resulting records. It verifies dimensions, injectivity, exact nullspace annihilation, normalisation, model-row mappings within each retained bank, and every primal residual. Its receipt, `verification.json`, records three certificates, 30 primal output bounds, and 15 rejected mutation controls.

`replay_model.py` separately binds the numerical records to the frozen exported networks. It reads the discovery image split, model weights, native-eight report parameters, contract and freeze hashes, and discovery banks. Using Python arbitrary-precision integer arithmetic, it independently reconstructs the two hidden ReLU layers' endpoint computations, all 5,712 discovery responses, and every native-eight report difference. It does not import the study's target function. The retained subbanks and original row mappings match exactly. Its receipt is `model_replay.json`.

From this directory:

```text
python3 -B verify_exact.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B replay_model.py
```

The model replay requires NumPy and the sibling `nonlinear_application` directory. Regenerating proposed certificates additionally requires SciPy and the sibling `certificates/affine_audit.py`. The interpreter used here was `/Users/admin/.venvs/gaia-tess-microlensing/bin/python`; verification itself calls no optimiser.

These results establish a bounded expressivity failure beyond report collisions at a cut with nonlinear downstream computation. They do not establish semantic meaning, causal-model identification, transfer, minimum feature-repair cost, or a broader research quality rating. The earlier independent proof review remains unchanged and did not assess this supplement.
