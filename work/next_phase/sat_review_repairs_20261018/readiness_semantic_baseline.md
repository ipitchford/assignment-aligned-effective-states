# Missing semantic-only readiness baseline

This is a post-review, read-only analysis of the saved development archive. It performs zero neural forwards, zero fits and zero new resampling.

| Scope | Paired unit-origins | Structured BA | Semantic-only BA | Structured minus semantic-only |
|---|---:|---:|---:|---:|
| Primary clipped assessment | 955 | 0.909543 | 0.787105 | 0.122438 |
| Excluded unclipped assessment | 133 | 0.953432 | 0.953432 | 0.000000 |
| All selected assessment | 1088 | 0.910856 | 0.789392 | 0.121464 |

For the primary assessment, the paired 95% original-origin bootstrap interval for structured minus semantic-only is [0.104108, 0.137696]. All 2,000 archived multiplicity vectors were reused, and the structured draws reproduce the frozen archive exactly.

The semantic-only predictor says that the named unit will activate after the declared repair and will not activate after the neutral edit. The structured predictor adds the fitted original-state readiness bit. Their difference therefore measures the incremental value of that bit under this exact action pair and opened cohort.

This comparison does not establish causal mediation, a global SAT repair, or protected generalisation. The unclipped rows are shown separately because excluding them was part of the frozen primary estimand.
