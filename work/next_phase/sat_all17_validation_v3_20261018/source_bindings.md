# Frozen source bindings

The execution code verifies these complete-file SHA-256 bindings before every
stage:

- upstream `helpers.py`: `ece43db7317fa8e3ad6cb6fb40bb1ba9f98ecd9191cbec6689b0bc67c0a268fe`;
- upstream `interpretation.py`: `b8f87ceaecb0c2449f3f626983ec028f2d3661c394d08f7d96641f4e06ba2177`;
- recovered primary-table mapping `source_mapping.json`:
  `5768a10f40de3c3a5a8b981a571992bd4d7f201b5034c87817f3c99221e9510d`.

The bound tokenizer source defines positive literals as tokens `0..4`, negated
literals as `5..9`, and then `s = 2*nvars+3`, `u = 2*nvars+4`. With the bound
model configuration `nvars = 5`, SAT is token 13 and UNSAT is token 14.

The bound interpretation source constructs truth-table rows with
`np.unpackbits(...)[-nvars:]` and renders them left-to-right as `x0,...,x4`.
Therefore `x0` is the most significant bit and, for example, `TFFFF` is index 16.

The bound recovered mapping is sourced from Palumbo et al. Appendix G Table 2
and records the exact seventeen single-assignment general units, their T/F names,
and truth-table indices. `verify_frozen_packet()` parses that file and requires
its ordered triples to equal `UNITS`, `ASSIGNMENT_NAMES`, and `ASSIGNMENT_IDS`.

The model runtime separately binds upstream `model.py` at
`7555e1f2f27546225397aa12de4fde1f6c903e9804766de477fe4c02770691ea`
and the epoch-900 checkpoint at
`d587446c64b7a8d68de93808ccf84a84364b85dcaf2149b8983b77895e24046d`.
