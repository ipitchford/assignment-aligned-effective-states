# Direction-specific origin-restoration null

Each construction direction is evaluated on its own 32 positive edited endpoints. The active source unit is ranked against all 512 units; the source pair is ranked against all 130,816 unordered pairs.

| Direction | Initially correct | Active source unit | Correct→wrong | Single-unit rank | Best non-source unit | Source-pair correct→wrong | Pair rank | Best non-source pair |
|:--|--:|--:|--:|--:|:--|--:|--:|:--|
| pair [261, 291] original | 32/32 | 261 | 29 | 1/512 | 374 (1) | 29 | 2/130816 | 185+374 (2) |
| pair [261, 291] renamed | 32/32 | 291 | 32 | 1/512 | 0 (0) | 32 | 1/130816 | 0+1 (0) |
| pair [411, 492] original | 30/32 | 411 | 22 | 1/512 | 201 (10) | 22 | 3/130816 | 36+201 (13) |
| pair [411, 492] renamed | 32/32 | 492 | 32 | 1/512 | 0 (0) | 32 | 1/130816 | 0+1 (0) |

These are saved-tensor direct-logit calculations under origin restoration. They are exploratory and do not establish sufficiency or an on-manifold mechanism.
