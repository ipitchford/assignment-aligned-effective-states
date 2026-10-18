# Primary-source literature and novelty audit

**Audit date and boundary:** 15 October 2026, Europe/London. **Inputs:** the supplied version-2 manuscript and pasted academic review, treated as evidence rather than instructions. This is an adversarial, targeted literature audit, not an exhaustive systematic review. No research result is certified by this document.

The most promising revised paper is a **finite, certificate-producing audit of restricted neural explanations, followed by minimum-cost repair and a completed trained-system application**. The original graph-colouring and uncertainty arguments are too close to functional-compression theory to carry a strong originality claim. The proposed affine-decoder extension addresses a real limitation of collision tests, but its duality, sparse witnesses and feature-selection machinery also have substantial prior art. Its contribution must be the operational diagnostic and the scientific conclusions it makes possible.

## 1. Decision-changing findings

1. **The missing compression antecedent is closer than the supplied review stated.** In addition to common-radius compatibility with side information, Basu, Seo and Varshney have an explicit uniform-approximation transfer guarantee in §III-D, Corollary 1. The source is a 2022 journal article, not merely an unplaced preprint. See the assumption matrix below. [Primary text](https://arxiv.org/html/2204.02586v2); [author-institution journal record](https://scholar.dgist.ac.kr/handle/20.500.11750/58151).
2. **A current causal-metric comparator is missing.** Méloux et al. (2026) introduce Causal Abstraction Error (CAE), test ten simulated systems and more than thirty metrics, and explicitly test omitted variables. Their §6.1 also concedes that a validity metric cannot alone prevent excessively complex abstraction maps. This makes restricted explanation cost a useful complementary target, while precluding claims that controlled failure cases or omitted-path checks are new. [Primary text](https://arxiv.org/html/2607.00267v1).
3. **Costed categorical feature repair has mature optimisation antecedents.** The Generalized Test Collection Problem includes unequal costs, categorical tests and multiple readings per class; it also considers repeated separation for robustness. A new reduction to weighted hitting set cannot be the central novelty. [Author manuscript](https://www.eng.tau.ac.il/~talraviv/Publications/GTCP_Sep2019.pdf).
4. **The proposed affine direction is more discriminating, but foundational claims need restraint.** Short signed contrasts annihilating the report matrix are the dual witnesses of linear Chebyshev approximation. The support-size argument is ordinary LP basic-solution sparsity. Use that mathematics as an established tool and demonstrate why the resulting witness changes an interpretability judgement. [Yang et al., §3](https://arxiv.org/html/2308.07636).

## 2. Assumption-by-assumption comparison with bounded-error functional compression

The following comparison is an inference from the inspected definitions and results, rather than a claim that every manuscript statement appears verbatim in the source.

| Assumption/objective | Basu–Seo–Varshney (2022) | Version 2 / proposed revision | Novelty consequence |
|---|---|---|---|
| Input domain | Discrete sources | Finite audit rows | Specialisation |
| Response geometry | Normed output space | Primarily coordinatewise maximum error | Specialisation |
| Compatibility | Common radius-ε ball, each supported context | Pairwise conflicts for maximum norm | Same geometry in this norm |
| Error objective | Asymptotic zero excess-distortion probability | Worst-case finite-bank guarantee | Different objective; limited increment alone |
| Side information | Decoder-side context | Declared decoder-only context | Direct antecedent |
| Encoder/decoder | Coding maps | Unrestricted labels in version 2; frozen reports and affine decoder proposed | Restriction is the substantive change |
| Approximate response knowledge | Uniform δ-approximation; Corollary 1 transfers codes | Certified measured-response bracket | Direct antecedent for transfer mechanism |
| Refinement | Includes successive-refinement coding settings | Must retain an existing report | Different constraint; not a new general refinement idea |
| Scientific output | Coding rates and constructions | Rejection witness, admissible repair and cost | Application contribution requiring demonstration |

Inspected: §II-A, Definition 3 in §II-B, §III-B and §III-D/Corollaries 1–2. No priority claim for graph compatibility, bounded distortion, decoder context or triangle-inequality transfer should remain. [Primary text](https://arxiv.org/html/2204.02586v2).

## 3. The new affine diagnostic: closest mathematical and algorithmic antecedents

Let the frozen report matrix be A (including permitted intercept/context columns) and the nominated response vector be y. The proposed diagnostic minimises ||Aβ−y||∞. A dual vector v with Aᵀv=0 and ||v||₁≤1 supplies a lower bound |vᵀy|. This can reject affine adequacy even when no two report rows collide. Those facts are classical; the research object is how to use the witness in a fair causal audit.

| Proposed ingredient | Closest inspected antecedent | Precise distinction / permitted claim |
|---|---|---|
| Minimax affine-decoder error and signed annihilator | Yang, Zhang & Zhang (2023), §3, dual LP in proof of Theorem 3.1 | Same optimisation primitive. Their ordered alternation theorem assumes a Haar system; arbitrary neural report matrices need only the more general LP argument and do not inherit uniqueness or ordered sign alternation. |
| Witness on at most rank(A)+1 rows | Same §3: a basic dual optimum uses at most the equality-system rank | A rank-specialised corollary, not a new sparsity theorem. Rank-deficient designs and zero optimum require explicit handling. |
| Small infeasible numerical subsystem | Gleeson & Ryan (1990), alternative-polyhedron vertex characterisation | Short failure explanations for linear constraints have direct precedent. Do not identify a sparse optimal witness with a minimum-cardinality witness without proof. |
| Select feature subset, solve continuous subproblem, add dual cuts | Benders (1962); Geoffrion (1972); Bertsimas & Van Parys (2020), §§2–3 | Exact feature selection using dual decomposition is established. The response bank, cost contract and human-readable witness interpretation can be distinctive. |
| Exact feature selection with a restricted predictor | Aytug (2015), SVM feature selection through generalized Benders | Different loss and predictor constraints, but a close counterexample to broad claims of first exact decoder-aware feature selection. |
| Exact rational proof after floating-point optimisation | Szeider (2026), VIPR certificates from black-box ILP; broader certifying-algorithm tradition | Independent replay is a reliability property, not mathematical originality. It proves only the encoded rational finite instance. |
| Repeated falsify/refine loop on a neural system | Elboher, Gottschlich & Katz (2020), §§3–4; Clarke et al. (2000) | Neural CEGAR already exists. Their abstraction overapproximates a network to verify properties; this proposal repairs a report/decoder explanation on a declared response bank. |

Primary sources: [Yang et al.](https://arxiv.org/html/2308.07636), [Gleeson–Ryan](https://pubsonline.informs.org/doi/10.1287/ijoc.2.1.61), [Benders](https://research.tue.nl/en/publications/partitioning-procedures-for-solving-mixed-variables-programming-p-2/), [Geoffrion author copy](https://www-apache.anderson.ucla.edu/faculty_pages/art.geoffrion/home/docs/GBD.pdf), [Bertsimas–Van Parys](https://arxiv.org/abs/1709.10029), [Aytug](https://www.sciencedirect.com/science/article/pii/S0377221715000077), [Szeider](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.CP.2026.52), [neural CEGAR](https://pmc.ncbi.nlm.nih.gov/articles/PMC7363186/), [original CEGAR author copy](https://www.cs.cmu.edu/~emc/papers/Conference%20Papers/Counterexample-guided%20Abstraction%20Refinement.pdf).

**No direct mechanistic-interpretability paper implementing this exact Chebyshev-contrast/optional-feature-repair workflow was located in the searches recorded below.** This is a bounded negative search result. It does not establish priority. The paper should describe a method assembled from established optimisation tools and make its research value visible in experiments.

### Distinctions that matter for the proposed repair theorem

- A decoder restriction is an explicit scientific assumption. Rejecting affine sufficiency does not reject information sufficiency, all nonlinear explanations or the Already-Clean hypothesis in general.
- XOR is a transparent demonstration that distinct reports can fail a linear decoder. XOR's nonlinearity is not itself a novel discovery.
- For unbounded coefficients, a candidate feature that has nonzero contrast can invalidate a particular dual witness, but this does not by itself solve every other obstruction. The algorithm must re-audit after repair.
- For bounded coefficients, the support-function allowance is essential: a tiny contrast cannot be made decisive through an arbitrarily large coefficient. Bound the intercept and permitted context consistently, or explicitly exempt them.
- Minimum cost means optimum over the frozen menu, fixed scales, precision, decoder class and response bank. It is not optimum over semantic explanations or over all possible native groups.
- A sparse witness and a short explanation are different objects. Rows can contain complex interventions; the actual intervention specification and side-state channels must remain inspectable.

These are consequences of the proposed contract, not additional claims of priority.

## 4. Feature/test selection and repair: why the classical reduction is insufficient

Vinterbo, Kim and Ohno-Machado (2005), §2.4, define a discernibility predicate, require selected descriptors to hit the sets distinguishing differently labelled examples, and identify the resulting minimum hitting-set problem. Their §2.5 uses weighted set cover for rule filtering. This is a particularly close historical analogy to an explanatory vocabulary chosen to preserve response distinctions. [Publisher full text](https://academic.oup.com/bioinformatics/article/21/9/1964/409100).

Sela Perelman et al. (2016), §3.2, convert minimum test cover to set cover over pairs of events and give greedy and augmented-greedy algorithms. This supplies an engineering precedent for sensor selection through distinguishability. [Author paper](https://cs.vuse.vanderbilt.edu/koutsoxd/www/Publications/1-s2.0-S000510981630231X-main.pdf).

Huang, Zheng and Chen's *Conflict hypergraphs for distinguishing feature selection* appeared online on **7 October 2026** (Information Sciences, article 124270; DOI 10.1016/j.ins.2026.124270). The publisher preview explicitly declines novelty for the hitting-set reduction. It instead contributes structural complexity results and exact dynamic programs, and evaluates downstream models. This is a timely warning that a new repair formulation needs either substantial new structure or demonstrated scientific utility. Full paywalled text was not inspected. [Publisher preview](https://www.sciencedirect.com/science/article/pii/S0020025526012016).

The affine extension should therefore be presented as resolving a failure of **pairwise discernibility as a sufficient criterion for a specified decoder**. The optional-feature optimisation remains part of sparse constrained approximation; the bridge to mechanistic audit is where a defensible contribution may lie.

## 5. Causal validity, complexity and identifiability

**Causal abstraction.** Geiger et al. (2025), §2, already supplies exact/approximate transformations, intervention algebras and simple, recursive and distributed interchange interventions. Output-response prediction is a narrower audit than validating a complete high-level mechanism. A report can predict outputs while proposing incorrect intermediate structure. [JMLR paper](https://jmlr.org/papers/volume26/23-0058/23-0058.pdf).

**Alignment complexity.** Sutter et al. (2025), Theorem 1 and Appendix F, establish a permissive-alignment result under countable inputs, layerwise input injectivity, strict output surjectivity and further structural/behavioural assumptions. These assumptions must not be suppressed. Their §6 discussion calls for an accuracy–complexity tradeoff; they also distinguish an existence construction from learnability and generalisation. A finite affine contract is a concrete response to that concern, not the first identification of it. [Primary text](https://arxiv.org/html/2507.08802v2).

**CAE.** The new comparison should be: CAE tests a specified causal model/map triple, whereas the proposed audit supplies a lower-bound witness and costed repair inside a restricted report/decoder class. CAE's constructive setting uses disjoint variable assignment and surjective value maps; its benchmark macro-models are deterministic. The proposed response audit can be more restricted in mechanism validation and more explicit in decoder feasibility. Both limitations belong in the paper. Inspected §§2.1, 3, 3.1, 5 and 6.1. [Primary text](https://arxiv.org/html/2607.00267v1).

**Non-identifiability.** Méloux et al. (2025), §§4–5, finds multiple accepted circuits and alignments, including a larger-model subnetwork example. Simplicity alone does not establish a unique ontology. Report an equivalence class or tied optimum when several repairs have the same certified cost. [Primary text](https://arxiv.org/html/2502.20914).

**Probe complexity.** Voita and Titov (2020) already compare accuracy with description length and learning effort, using variational and online coding. Fixed coefficient budgets and report quantisation differ from those MDL estimators, but charging decoder effort is established. An affine report should be accompanied by control tasks or random-feature controls, rather than suggesting that linearity alone resolves every probe confound. [ACL primary record](https://aclanthology.org/2020.emnlp-main.14/).

**Refinement.** Zečević et al. (2023), *Continual Causal Abstractions*, explicitly proposes revising a coarse abstraction after counterexamples; it is a short research-direction paper. A new automatic method can advance this programme but should not claim the refinement idea itself. [PMLR paper](https://proceedings.mlr.press/v208/zecevic23a/zecevic23a.pdf).

**Formal explanations.** Ignatiev, Narodytska and Marques-Silva (AAAI 2019) already computes subset/cardinality-minimal abductive explanations through constraint oracles. Those are local sufficient feature assignments for a prediction, rather than a global response decoder, but they preclude broad claims of first optimal or formally certified machine-learning explanations. [Primary preprint](https://arxiv.org/abs/1811.10656).

## 6. Native features, learned features and benchmark reliability

| Source | What is established / inspected | Consequence for this paper |
|---|---|---|
| Huang et al., RAVEL (ACL 2024) | Publisher abstract/record; controlled attribute disentanglement and distributed alignment method | Preserve target success and non-target preservation as separate outcomes; do not equate output change with clean causal control. |
| Karvonen et al., SAEBench (ICML 2025) | Publisher abstract/record; eight evaluation metrics, 200+ SAEs, proxy/practical discrepancies | A single new score should be shown to answer a specific additional question. |
| Shafran, Geiger & Geva (2026), compositional neuron groups | Full-text §5 and Appendices D–E; sparse semi-NMF groups, steering comparison, adapted rather than identical AxBench concept sets | Include native groups as a substantive competitor. Count group construction, support, precision and supervision. |
| Chanin (2026), benchmark reliability | Full-text synthetic benchmark results, §6 and discussion; synthetic oracle/ground-truth comparisons and training-trajectory checks | Include random baselines, known-answer failures, repeated seeds and uncertainty. Results are scoped to tested metrics/panels. |
| Korznikov et al. (2026), random-baseline checks | Primary abstract inspected; not full reproduction | Random features are a required sanity check; avoid detailed claims beyond inspected evidence. |
| Paulo & Belrose (2025), SAE feature instability | Primary abstract inspected; not full reproduction | Feature identity across seeds is not assured. Track stability separately from finite response adequacy. |
| Méloux et al. (ICML 2026), statistical estimation | Publisher record; circuit-discovery variance and perturbation analysis | Treat feature/circuit discovery as an estimator. One frozen checkpoint/menu is a bounded case study. |

Primary links: [RAVEL](https://aclanthology.org/2024.acl-long.470/), [SAEBench](https://proceedings.mlr.press/v267/karvonen25a.html), [native groups](https://arxiv.org/html/2506.10920v2), [benchmark reliability](https://arxiv.org/html/2605.18229), [random baselines](https://arxiv.org/abs/2602.14111), [feature instability](https://arxiv.org/abs/2501.16615), [variance analysis](https://proceedings.mlr.press/v306/meloux26a.html).

The Already-Clean homepage failed to load through the web fetch used by this audit. Consequently, this audit does not independently authenticate the latest live essay against the supplied archive. Its claims about that essay rely on the supplied manuscript/review and should be checked against a preserved source by the parent investigation.

## 7. Three promising contributions, ranked

### A. Certified decoder failure and minimum-cost repair on a trained system — strongest immediate route

Freeze a trained checkpoint, target responses, allowed contexts, feature menu, numerical precision and affine coefficient budget. Show a case where a plausible report has no collisions and is semantically legible, but a short contrast proves every permitted decoder fails. Use the contrast to search a menu containing native neurons, native groups and learned features; certify the optimal repair cost and test the fixed repaired decoder on untouched interventions. Include identity-state, unrestricted decoder, random-feature and shuffled-target controls.

**Distinctive claim to attempt:** an interpretable, independently replayable diagnosis-and-repair workflow that reveals a consequential distinction hidden by legibility, accuracy and collision checks. **Do not claim:** new minimax duality, a general solution to causal abstraction, global optimality over explanations, or a general SAE/native-neuron verdict.

**Evidence threshold:** at least one real trained-system decision, not only a manually arranged XOR network; a frozen hold-out test; a cost-sensitive result that survives a reasonable alternative budget; concrete rejection and repair witnesses. Stronger significance comes from repeating the phenomenon across seeds and at least one additional task or architecture. A small controlled model can establish proof of method, but prevalence in foundation models remains untested.

### B. Robust audit under bounded feature and response error — useful extension if proved and executed

The finite rational optimum should be connected to the actual activation/response numbers via declared error bounds. With bounded decoder coefficients, feature uncertainty contributes a computable allowance; without such bounds, small input discrepancies can be amplified without limit. Produce a certificate whose conclusion survives numerical rounding and a tolerance sweep, and show the exact boundary where it becomes inconclusive.

**Potential value:** a credible measurement procedure for the revision's restricted decoder, rather than the original uninstantiated error assumption. **Prior-art ceiling:** robust optimisation, support functions and uncertainty transfer are established. Novelty must be the contract-specific diagnostic, complete certificate and empirical consequence, unless a separate difficult theorem is proved.

### C. Budgeted intervention discovery that finds the failure and validates a repair — longer-term route

Use contrasts to propose additional interventions that distinguish tied or nearly tied candidate explanations, then protect inference with fresh evaluation data or a valid adaptive procedure. Compare query cost with uniform sampling and CAE-style failure detection. Include multi-site interventions so that agreement on single interventions cannot conceal interaction errors.

**Potential value:** reduce the experimental burden of discovering a useful explanation. **Prior-art ceiling:** CEGAR, active learning, test cover and causal intervention design already cover much of the idea; CAE explicitly lists adaptive sampling and explanation discovery as future work. A generic loop or union bound is insufficient. This route requires an actual performance or theoretical gain.

These are candidate research directions. They are not completed discoveries in this audit.

## 8. Positioning language and rejection tests

A defensible introduction could say: “We apply classical minimax duality and exact discrete optimisation to audit restricted neural explanations. The method turns decoder failure into a small signed contrast of interventions, and searches a declared feature menu for a minimum-cost repair. Our contribution is the resulting diagnostic-and-repair procedure and the trained-system findings it supports.” This wording still requires the method and findings to be executed and verified.

Reject or narrow these claims during revision:

- “First tolerance-aware causal resolution theorem”: compression antecedents are direct.
- “First uncertainty-aware compatibility graph”: generic approximation transfer is already present in the closest source.
- “First optimal explanation repair”: too broad relative to sparse regression, formal explanations and test collection.
- “Few-row certificate proves the mechanism”: it proves a restriction fails on stated rows.
- “Minimum-cost repair establishes the true features”: optimum is menu-relative and may be nonunique.
- “More examples/tests establish significance”: checking volume is not originality or significance.

An exact but narrowly engineered demonstration may remain incremental; the paper must demonstrate useful reach beyond its construction.

## 10. Search process and limitations

Search engines/tools: live web search and primary-page/full-text inspection; Exa for a complementary semantic search. Semantic Scholar was attempted and returned HTTP 429 without results, so no citation-graph search is claimed. No proprietary Web of Science/Scopus coverage, full backward/forward snowballing, author consultation, code reproduction or comprehensive non-English search was performed. Publisher access to some papers was limited to previews/abstracts. Search-engine crawl dates were not used as publication dates.

Exact representative queries (all executed on 15 October 2026):

- `"functional compression" "maximal distortion" hypergraph`
- `"causal abstraction" "counterexample" refinement interpretability`
- `"causal abstraction" "minimal" "cost" explanation`
- `"feature selection" "discernibility matrix" "set cover"`
- `"test cover problem" "weighted" feature selection`
- `"counterexample-guided abstraction refinement" Clarke Grumberg Jha Lu Veith 2000 pdf`
- `"mechanistic interpretability" "counterexample-guided"`
- `"discernibility" "hitting set" feature selection Skowron Rauszer pdf`
- `"minimum test cover" "set cover" paper`
- `"rational" "certificate" "set cover" "dual"`
- `"Conflict hypergraphs for distinguishing feature selection"`
- `"Validating Causal Abstraction Metrics on Simulated Complex Systems"`
- `"The Generalized Test Collection Problem"`
- `Méloux 2025 mechanistic interpretability identifiability boolean circuits`
- `"rational certificates" "integer programming" verified optimization`
- `"Information-Theoretic Probing with Minimum Description Length" ACL`
- `"sparse" "Chebyshev approximation" feature selection`
- `"feature selection" "Farkas" regression`
- `"Chebyshev regression" "subset"`
- `"Benders" "sparse regression"`
- `"linear Chebyshev approximation" "dual" "n+1"`
- `"mechanistic interpretability" "Chebyshev"`
- `"causal abstraction" "linear programming" "dual"`
- `"neural" "explanation" "Farkas" certificate`
- `"An Abstraction-Based Framework for Neural Network Verification"`
- `"Abduction-Based Explanations for Machine Learning Models"`
- `"smallest explanations" "hitting set" Ignatiev`
- `Gleeson Ryan 1990 identifying minimally infeasible subsystems inequalities`
- `"Aytug" "2015" "Benders" feature selection`

Semantic-search prompts asked for primary papers on constrained causal abstraction, counterexample refinement, non-identifiability and metric reliability through the cutoff. AI-generated summary sites were used only as discovery leads; substantive comparisons above rely on primary or author/publisher sources. Sources that surfaced but were not inspected closely enough are recorded separately in the ledger rather than treated as established comparisons.

The audit has identified several stronger counterarguments than those in the supplied review. It has not exhausted all possible avenues or established global novelty. A specialist in approximation/optimisation and an independent mechanistic-interpretability reviewer should examine the exact final claims and frozen results before any strong originality or prospective quality claim is made.
