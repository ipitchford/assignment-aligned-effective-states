# Cached action-menu × action-encoding factorial

This post-review diagnostic uses zero new formulas and zero neural forwards. It fits the same two 100-tree report arms in four cells and pairs uncertainty by assessment origin.

| Menu/training regime | Action encoding | Published-bit success | Continuous-post success | Continuous minus bits | Paired 95% interval |
|---|---|---:|---:|---:|---:|
| all_slot_sparse_fit | one_hot | 0.624512 | 0.622656 | -0.001855 | [-0.017480, 0.013872] |
| all_slot_sparse_fit | natural_binary | 0.450195 | 0.485352 | 0.035156 | [0.013184, 0.058694] |
| selected_slot_dense_fit | one_hot | 0.651367 | 0.650391 | -0.000977 | [-0.020508, 0.019531] |
| selected_slot_dense_fit | natural_binary | 0.431641 | 0.497070 | 0.065430 | [0.037109, 0.091797] |

## Factorial contrasts

- `encoding_effect_on_continuous_minus_bits__all_slot_sparse_fit`: 0.037012, paired 95% interval [0.012888, 0.062405].
- `encoding_effect_on_continuous_minus_bits__selected_slot_dense_fit`: 0.066406, paired 95% interval [0.034155, 0.098633].
- `menu_effect_on_continuous_minus_bits__one_hot`: 0.000879, paired 95% interval [-0.020605, 0.022468].
- `menu_effect_on_continuous_minus_bits__natural_binary`: 0.030273, paired 95% interval [-0.004497, 0.064753].
- `menu_by_encoding_interaction`: 0.029395, paired 95% interval [-0.011040, 0.069438].

The all-slot/one-hot cell reproduces both original pilot selection rates exactly. The other cells use the same learner seed and report geometry to identify the menu and encoding contributions. This is an opened-data diagnostic, not a preregistered confirmation.
