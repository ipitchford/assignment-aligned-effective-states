# Pinned runtime conventions

The custom Palumbo model is not TransformerLens. At the pinned source and
checkpoint hashes, the final block is `blocks[1]` of exactly two blocks. Its
state dictionary records `blocks.1.mlp.W_out` shape `(128,512)`,
`blocks.1.mlp.b_out` shape `(128,)`, and `unembed.W_U` shape `(128,16)`.
There is no unembedding bias and no final normalization: `Transformer.forward`
applies `unembed` directly to the final residual, and the runtime constructs the
model with `use_ln=False`.

The final MLP contribution is therefore `post @ W_out.T`, and the unit-to-logit
coefficient matrix is `W_out.T @ W_U`, with shape `(512,16)`. The native script
asserts every shape and absence above before evaluating a formula.

The custom `HookPoint.add_hook` invokes callbacks as `hook(module_output,
name=self.name)`. The extractor accepts this `name` keyword, asserts that every
hook fires, and checks the complete batch shapes. The four resolved attributes
are `blocks[1].mlp.hook_pre`, `blocks[1].mlp.hook_post`,
`blocks[1].hook_resid_mid`, and `blocks[1].hook_resid_post`. Before the forward
marker, the extractor verifies that each exposes callable `add_hook`, that the
model exposes callable `remove_all_hooks`, and that the runtime receipt is JSON
serialisable. The runtime pins the checkpoint,
model source and every executed class AST; `protocol.py` independently pins the
runtime, model source and checkpoint files.

`HookPoint.add_hook` wraps callbacks as PyTorch forward hooks and returns their
value; the extractor returns `None`, which leaves the module output unchanged.
`Transformer.remove_all_hooks()` removes both block and MLP hook handles. The
pinned loader calls `model.eval()`, disables gradients, selects CPU, and loads the
checkpoint with `strict=True`. The exact configuration fields asserted by the
extractor are `nvars`, `d_model`, `d_mlp`, `d_vocab`, and `use_ln`.

The source configuration has five variables and vocabulary size 16. The bound
source defines SAT token `2*nvars+3 = 13`, UNSAT token `2*nvars+4 = 14`, and the
activation event as final-block post-ReLU value `>=0.5`. Since the threshold is
positive, `pre >=0.5` and `post >=0.5` are identical. A singleton-assignment
unit's expected positive event is that its named assignment satisfies the formula,
including formulas with more than one satisfying assignment.
