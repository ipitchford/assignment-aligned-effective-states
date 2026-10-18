# Exact semantic calibration for the clause-replacement pilot

This note uses only the declared grammar and action contract. It does not inspect neural activations, checkpoint outcomes, development formulas, or protected cases, and it does not amend the frozen pilot. The script work/sat_semantic_theory/check.py is a standalone standard-library enumeration and certificate generator.

The strongest bridge is **clause-deletion backbone information**. For the initially SAT population, it exactly determines the 39 replacement labels in a fixed context. Current satisfying-assignment bits can lose that information, even when both colliding states admit a successful edit and require different choices. This gives a specific semantic hypothesis for information retained in amplitudes. It does not establish that the model implements that hypothesis.

## 1. Exact one-menu equivalence

Write an ordered formula as $F=(c_0,\ldots,c_{m-1})$. A valid action is $(i,c_i,d)$, with $d\ne c_i$. Clauses contain two distinct variables. Let $M(F)$ be its satisfying assignments, let

\[
S_i(F)=M(F\setminus\{\text{occurrence }i\}),
\qquad B_i(F)=\operatorname{Backbone}(S_i(F)),
\]

where a backbone records variables fixed to a common value in every assignment in a nonempty set. Removing one occurrence, rather than all copies of its clause, matters. Since $F$ is initially SAT, $S_i(F)$ is nonempty.

**Theorem (exact replacement-menu quotient).** For initially SAT $F$, a replacement $d=\ell_p\lor\ell_q$ produces UNSAT precisely when both $\neg\ell_p$ and $\neg\ell_q$ belong to $B_i(F)$. Consequently:

1. The whole menu is determined by $B_i(F)$.
2. If $B_i(F)$ fixes zero or one variable, every replacement remains SAT.
3. If it fixes at least two variables, its UNSAT-producing replacement clauses recover $B_i(F)$ exactly.

For formulas sharing the same valid context $(i,c)$, two menus therefore agree if and only if their deletion backbones are equal, or both fix at most one variable.

**Proof.** The edited formula has model set $S_i(F)\cap M(d)$. It is empty exactly when every member of $S_i(F)$ falsifies $d$, which is equivalent to every member falsifying each of its two literals. These are precisely the two specified backbone facts. If the backbone has $k$ fixed variables, exactly $\binom{k}{2}$ canonical 2-clauses are false throughout it: one for every pair of fixed variables. For $k\ge2$, the union of the negations of those clauses' literals recovers every backbone literal. For $k\le1$ there are no such clauses. The excluded old clause $c$ cannot be one of them because $S_i(F)\cap M(c)=M(F)$ is nonempty. Excluding $c$ from the menu thus loses no recovery information. ∎

This supplies a necessary-and-sufficient criterion for an arbitrary pre-edit report $T$: for every report fibre and every **common valid** context $(i,c)$, all deletion backbones within that context must have the same quotient value from the theorem. Necessity follows from two different menus sharing the same predictor input; sufficiency follows by assigning the common menu to each report/context pair. No decoder architecture is assumed. Different old clauses are different inputs and cannot be ignored in a collision claim.

For five variables and a fixed old clause, there are exactly

\[
3^5-3^3-2\cdot5=206
\]

possible menu classes in the ten-clause grammar. Of the $3^5$ ternary backbone patterns, $3^3$ would force both old literals false and are incompatible with initial satisfiability. The remaining zero/one-fixed-variable patterns number eleven and collapse to one menu, removing another ten classes. Every other compatible pattern is realisable: force an anchor literal $a$ using $(a\lor z)\land(a\lor\neg z)$; force each other required literal $b$ using $(\neg a\lor b)$; then repeat clauses to reach nine clauses and append the specified old clause. At most six clauses are needed before padding. For the empty backbone, nine copies of the old clause suffice. All clauses have distinct variables. Thus a fixed-context exact report requires at least eight fixed-length bits, and eight suffice by indexing the classes. The enumeration constructs all compatible backbones for all forty old clauses.

This is a **conditional** eight-bit statement. The actual encoder is a shared pre-edit report, built before the queried slot is selected. Supplying an oracle-computed eight-bit deletion backbone separately for each queried slot would change the report contract. Nor may the conditional lower bounds be summed over slots: the contexts are correlated and the old clauses are supplied as side information.

## 2. What the near-satisfying information contributes

Let $q_F(v)$ count violated clause occurrences, and let $\nu_c(v)$ be one when $v$ violates $c$. Directly,

\[
S_i(F)=M(F)\;\cup\;
\{v:q_F(v)=1,\ \nu_{c_i}(v)=1\}.
\]

Removing the queried clause admits precisely assignments whose sole violation was that occurrence. The menu theorem takes the backbone of this enlarged assignment set. Thus an exact assignment-wise indication of the $q=1$ layer is sufficient, but it is more information than the one-menu decision requires. The relevant aggregate is which current backbone literals survive the added near-solutions. Neither the complete violation-count vector nor every near-satisfying assignment is proved necessary.

Here is a stronger calibration than a SAT/UNSAT-label collision. Use variables $x_0,\ldots,x_4$, and let the common old clause be $c=x_0\lor x_1$ at slot zero. Write

\[
U(z)=(z\lor x_4)\land(z\lor\neg x_4).
\]

Construct

\[
\begin{aligned}
F&=c\land U(\neg x_1)\land U(x_2)\land(\neg x_0\lor x_3)
   \land(\neg x_1\lor\neg x_4)^4,\\
G&=c\land U(\neg x_1)\land U(x_3)\land(\neg x_0\lor x_2)
   \land(\neg x_1\lor\neg x_4)^4.
\end{aligned}
\]

The exponent denotes four repeated clause occurrences. Each formula has exactly ten canonical clauses. Both have the same complete model set:

\[
x_0=1,\quad x_1=0,\quad x_2=x_3=1,\quad x_4\text{ free}.
\]

Deleting the common old clause leaves backbones $\{\neg x_1,x_2\}$ and $\{\neg x_1,x_3\}$ respectively. Hence the only UNSAT-producing replacement for $F$ is $x_1\lor\neg x_2$, while the only one for $G$ is $x_1\lor\neg x_3$. The two menus are identical as presented to a policy using the current 32-assignment truth table and the common context, but their unique successful actions differ.

Any deterministic or randomized policy with that input succeeds with probability at most $1/2$ on the uniform pair; a fully informed oracle succeeds on both. Indeed the policy assigns probabilities $p_F,p_G$ to the two distinct successful clauses, with $p_F+p_G\le1$. Mean success is $(p_F+p_G)/2\le1/2$. Both menus are reachable, so this is not merely an inability to identify an origin on which no successful edit exists.

This is an exact symbolic semantic witness. It does **not** assert that the published 34 native bits collide on these formulas or that the checkpoint classifies any of their endpoints correctly. No checkpoint inference was performed.

## 3. One-step sufficiency does not make a closed update state

The known count update is

\[
q_{F'}(v)=q_F(v)-\nu_c(v)+\nu_d(v).
\]

For an action sequence of at most $L$ valid replacements, initial counts capped at $L+1$ suffice. If $q_F(v)>L$, at most $L$ removals cannot make it zero. If $q_F(v)\le L$, the cap retains it exactly. This argument answers the whole sequence from the initial capped state and the cumulative known changes. Equivalently, with $r$ edits remaining one may store $\min(q,r+1)$; after an edit the exact next state at the shorter horizon is

\[
\min\{\min(q,r+1)-\nu_c+\nu_d,\ r\}.
\]

The horizon decreases. A fixed low cap is not asserted to be an indefinitely closed update state.

There is a small sharpness family entirely within the frozen five-variable, ten-clause grammar. Put

\[
c=x_0\lor x_2,\quad d=x_0\lor\neg x_2,\quad
e=\neg x_0\lor x_1,\quad r=\neg x_0\lor\neg x_1.
\]

For each $L\in\{1,2,3,4\}$ let

\[
F_L=[c^L,d,e^{\,9-L}],\qquad
G_L=[c^L,d,e^{\,8-L},c].
\]

Both initially have exactly the models $x_0=x_1=1$. Their entire vectors $\min(q,L)$ agree. To see this, assignments fall into four groups:

| Assignment condition | $q_{F_L}$ | $q_{G_L}$ |
|---|---:|---:|
| $x_0=0,x_2=0$ | $L$ | $L+1$ |
| $x_0=0,x_2=1$ | 1 | 1 |
| $x_0=1,x_1=0$ | $9-L$ | $8-L$ |
| $x_0=x_1=1$ | 0 | 0 |

For these horizons $8-L\ge L$, so capping at $L$ makes the rows agree. Replace the first $L-1$ copies of $c$ by $d$, then the $L$th copy by $r$. Both formulas remain SAT before the last edit. The final $F_L$ is SAT, witnessed by $x_0=x_2=0$, while $G_L$ retains a copy of $c$ and is UNSAT. Every descriptor is valid for both formulas and supplies the same old clause $c$ in the same slot. The descriptors do not reveal the differing final slot or the full formulas.

Thus cap $L$ fails for $L$ replacements, while the known cap $L+1$ suffices. This sharpness is for the **uniform count-cap control**, not a proof that every sufficient representation must encode those counts.

For $L=2$, the two formulas additionally have identical deletion-backbone profiles at **all ten slots**. Their current model sets also agree. Nevertheless the same two-action sequence has different final labels. Therefore even the exact shared one-step semantic profile, augmented by the complete current truth table, is not a closed state for two-step composition. This is a concrete finite abstraction failure, not a new general theorem of dynamic SAT.

## 4. Representation and decoder cost controls

A shared deletion-backbone profile has a simple 55-bit encoding. Deletion only enlarges the model set, so every $B_i(F)$ is a subset of the current backbone. For each of the five variables, its fixed value can never disagree across deletion contexts. Store either “fixed nowhere”, or a value and a nonempty ten-slot mask indicating where it remains fixed. There are $1+2(2^{10}-1)=2047$ possibilities per variable, requiring eleven bits. The decoder reads the queried slot and tests the new clause's two literals. This is an upper bound, not an optimal global profile code, and computing the profile is an oracle operation.

A more consequential control is the complete **clause multiset**. There are

\[
N=\binom{40+10-1}{10}=\binom{49}{10}=8{,}217{,}822{,}536<2^{33}
\]

multisets. A 33-bit code therefore retains every multiplicity exactly. Given a valid action, decrement the old clause multiplicity and increment the new one; compute semantic SAT from the result. Repetition and arbitrary bounded compositions cause no loss.

For an executable encoding, sort the clause IDs $0\le a_0\le\cdots\le a_9<40$ and put $b_i=a_i+i$. These are ten distinct numbers in $\{0,\ldots,48\}$. Their colexicographic rank is

\[
\operatorname{rank}(a)=\sum_{i=0}^{9}\binom{b_i}{i+1}.
\]

The rank bijects multisets with $\{0,\ldots,N-1\}$. Inductively, subsets with maximum $t$ occupy the block from $\binom t{m}$ through $\binom{t+1}{m}-1$; the remaining $m-1$ elements index the block. Pascal's identity makes adjacent blocks meet without gaps. This also proves the greedy unranking algorithm used by the script.

| Exact semantic control | Explicit bits | Guarantee and cost |
|---|---:|---|
| Current satisfying-assignment truth table | 32 | Insufficient for the one-step witnesses above |
| Count vector capped at 2 | 64 | One-step labels; assignment-wise update and existential test |
| Shared deletion-backbone profile | at most 55 | All one-step menus on initially SAT formulas; oracle encoder, simple decoder; not closed for two edits |
| Full count vector, values 0–10 | 128 | Closed count updates; assignment-wise SAT test |
| Complete clause multiset, enumerative code | 33 | Exact semantic SAT under valid replacements and compositions; full multiplicity information, unranking and SAT computation |

The 33-bit control means the published 34-bit report is not excluded by cardinality from containing a sufficient semantic edit state. Failure would concern **which information it retains and what the decoder can do**, not merely having too few bits. A learned decoder receiving packed combinatorial bits may have a much harder computation than the explicit oracle decoder. Equal per-example bits do not establish equal encoder or decoder cost. The pilot's 34 float32 magnitudes have 1,088 nominal stored bits; that full-magnitude comparison is not a quantisation repair.

Three limits are essential. First, the multiset code retains formula content and must be named and charged as such; it is not an inexpensive abstract neural feature. Second, it cannot verify that a declared old clause occupies the queried slot. It relies on the frozen valid-action contract; invalid-action detection requires extra positional information. Third, **it cannot reconstruct token order or the order-sensitive checkpoint computation**. It is an exact control for the SAT oracle target, not a faithful native-neural emulator. Semantic success and checkpoint correctness remain different outcomes.

Presenting all ten menus together supplies every slot's old clause and reconstructs the ordered formula. That would bypass the report bottleneck. A composed query reveals only the old clauses of newly touched slots; revisiting a previously changed slot adds no new original clause information. Sequence length and supplied contexts must therefore remain explicit.

## 5. Mechanistic interpretation and prior-art limits

Backbones and their computation are established concepts: [Marques-Silva, Janota and Lynce, *On Computing Backbones of Propositional Theories*, ECAI 2010](https://sat.inesc-id.pt/~mikolas/ecai10.pdf). The effect of clause removal on a backbone is already central to [Singer, Gent and Smaill, *Backbone Fragility and the Local Search Cost Peak*, JAIR 12 (2000)](https://arxiv.org/abs/1106.0240). Backbone information in incremental SAT is also studied in [*Mining Backbone Literals in Incremental SAT*](https://ofers.dds.technion.ac.il/publications/sat15.pdf). The menu theorem above is an elementary application of the backbone definition, and the counts, horizon argument and multiset code use standard techniques. No foundational novelty is claimed.

The specific analytical bridge is narrower: under the pilot's new edit-selection contract, successful actions are determined by which backbone facts survive deleting the supplied old clause. An amplitude representation might retain this deletion sensitivity through near-satisfying assignments. It might instead retain clause multiplicities, positional features, parser information, confidence, or another correlated statistic. Predictive improvement alone cannot distinguish those explanations.

Useful future empirical evidence would include independently held-out decoding of the specified deletion-backbone facts; contrasts holding the current complete model set and action context fixed while changing deletion sensitivity; and targeted interventions or analytic component equations showing that the proposed native amplitude variables causally implement the relevant distinctions. Patching features of a newly fitted report decoder shows a dependence of that decoder, not automatically a causal mechanism of the original transformer. Any added feature search, quantiser, or outcome-directed contrast family must be separately frozen and labelled; none is licensed as a silent change to the present pilot.

[Palumbo et al., *Validating Mechanistic Interpretations: An Axiomatic Approach*, ICML 2025](https://proceedings.mlr.press/v267/palumbo25a.html) already studies replacement and compositional validation of mechanistic abstractions. The clause-edit task is a new use of the late report, not a refutation of the source's stated task. The results here sharpen that use and its controls; they do not establish that native amplitudes represent near-solutions, that an interpretation is causally faithful, or that the resulting research has attained a particular scholarly standing.

## 6. Exact replay

Run: python3 -B work/sat_semantic_theory/check.py. It imports only the Python standard library, writes work/sat_semantic_theory/results.json, and reads the two contract documents only to record hashes.

The completed replay checks all 20,736 ordered three-variable four-clause formulas, including 82,080 initially-SAT menus and 902,880 replacement labels; constructs every compatible five-variable deletion backbone for all forty old clauses; verifies the conflicting-action witness and all four horizon witnesses with direct Boolean evaluation; tests all 1,365 small-domain multiset ranks and 1,001 deterministic full-domain ranks; and exhausts 576, 5,184 and 46,656 valid action paths at horizons one, two and three on two-variable three-clause formulas. All checks pass. These are exact symbolic checks, not neural evidence or a claim to enumerate all ten-clause five-variable formulas.
