# Saved-tensor all-pair clamp null

This post-review calculation uses the saved 512-coordinate final-block vectors and exact affine readout. It performs zero neural forwards and evaluates all 130,816 unordered unit pairs.

| Source pair | Positive endpoints | Initially correct | Correct→wrong | Full-null rank | Activity-matched rank | Mean signed SAT−UNSAT contribution | Full-null rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| 261+291 | 64 | 64 | 61 | 1/130816 | 1/512 | 9.164239 | 1/130816 |
| 411+492 | 64 | 62 | 54 | 1/130816 | 1/512 | 8.319411 | 1/130816 |

Across the two source-matched endpoint sets, the nominated pairs change 115 decisions from correct to wrong among 126 initially correct positive endpoints.

The activity-matched set contains every unit pair whose mean two-coordinate activation-change RMS is within a factor of 1.5 of the source pair. It is a descriptive post hoc comparator, not a prespecified test.

The calculation is direct-logit attribution under the stated saved-tensor clamp. It does not show upstream mediation, unique causal routing or necessity under an on-manifold intervention.
