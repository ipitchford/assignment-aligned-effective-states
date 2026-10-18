# SAT-renaming construction replay

This component independently regenerates the deterministic construction used by
the SAT variable-renaming study. It imports no module from the study sampler,
constructor, protocol or semantic oracle. The replay reimplements the native
five-variable clause grammar, ordinary Boolean evaluation, the declared
`PCG64(2026101604)` proposal stream, eligibility rules, domain-separated hash
ordering and prior-ledger exclusions.

The retained run regenerated all 100,000 proposals and reproduced all 16 saved
array or metadata comparisons, including candidate clauses and counts,
eligibility codes, action multiplicities, eligible-record order, selected trial
IDs, all 64 matched blocks, 128 origin orbits and 512 endpoint sequences. It used
NumPy 2.4.2; the protected construction used NumPy 2.4.6. The receipt is
`report.json`, and `replay_arrays.npz` preserves the independently generated
selection arrays. No neural forward or learner fit occurs.

Run from the root of an extracted `causal-resolution-v3` package:

```text
python3 -B sat_renaming_selection_replay/replay_selection.py
```

The distributed directory already contains the exclusive outputs from the
retained run, so execute the script in a fresh copy with `report.json` and
`replay_arrays.npz` removed, or change `HERE` to a new empty output directory.
The script reads the frozen contract and selected-study inputs from the embedded
`work/` tree in the sibling `sat_renaming_case/` component. Exact reconstruction
verifies the declared deterministic selection procedure. It does not establish
external preregistration, absence from checkpoint training, or representativeness
of the selected cohort.
