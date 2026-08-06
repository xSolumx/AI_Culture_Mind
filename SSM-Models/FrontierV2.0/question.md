# Four Research Frontiers Beyond the Current Gates

These are research questions, not established results. Each proposal is stated
with an explicit mechanism, a decisive test, and a boundary on what a positive
result would justify.

## 1. Spontaneous gauge discovery in standard sequence models

### Question

Do unconstrained transformers or state-space models organize intermediate
activations into low-dimensional noncommutative group orbits while solving
compositional tasks?

The experiment would apply the decoder-blind compiler to hidden states from a
fixed pretrained model during tasks such as entity tracking, bracket matching,
or short program execution. No group structure would be inserted into the
model.

### Decisive evidence

A convincing result would require all of the following:

- a compact action extracted without task labels at the compiler stage;
- stable group relations on held-out trajectories and longer compositions;
- a noncommutative signal that survives matched linear, permutation, and
  clustering baselines;
- causal evidence that intervening on the extracted orbit changes the relevant
  computation.

Such a result would show that a particular trained network discovered a useful
group-like mechanism. It would not, by itself, prove that Lie-group structure
is a universal attractor of sequence learning.

## 2. Continuous dynamics and syntactic monoids

### Question

When does a continuous recurrence admit a finite behavioral quotient equal to
the syntactic monoid of a formal language?

For a language \(L\subseteq\Sigma^*\), the syntactic congruence identifies two
prefixes \(u,v\in\Sigma^*\) when no continuation can distinguish them:

\[
u\equiv_L v
\quad\Longleftrightarrow\quad
\forall p,q\in\Sigma^*,
\;puq\in L\iff pvq\in L.
\]

The quotient \(M(L)=\Sigma^*/{\equiv_L}\) is the syntactic monoid. The research
goal is not to assume that every neural state manifold is a homogeneous space
\(G/H\). It is to find verifiable conditions under which a learned continuous
transition system has a stable finite quotient, and then prove that this
quotient is isomorphic to \(M(L)\).

### Decisive evidence

The first theorem should cover a tightly controlled class of deterministic,
continuous recurrences. It must state observability, robustness, and minimality
assumptions explicitly and supply both positive examples and counterexamples
when those assumptions are removed.

## 3. Infinite-group compilers and topological memory

### Question

Can a constant-state recurrence represent and expose useful invariants of an
infinite noncommutative group over lengths far beyond training?

Candidate tasks include word reduction in a free group \(F_n\), braid-group
relations in \(B_n\), and bounded-depth stack languages. Compact orthogonal
dynamics alone cannot encode an unbounded discrete state injectively: compact
state spaces necessarily admit arbitrarily close recurrent configurations.
The plausible target is therefore narrower—stable computation of a selected
invariant or quotient, not lossless storage of every group element.

### Decisive evidence

Require exact or tolerance-certified relation checks, dense length sweeps well
beyond training, adversarial near-collision searches, and comparison with
automata-, stack-, and fast-weight baselines. A compiler must recover a stated
invariant without silently receiving the relation table it is meant to
discover.

## 4. Triality as a scan-compatible binding primitive

### Question

Can the invariant triality map provide useful dynamic binding inside a
constant-state, parallel-scan architecture?

The algebra supplies a bilinear map

\[
\rho:\mathbf 8_{+}\otimes\mathbf 8_{-}\longrightarrow\mathbf 8_v,
\]

where \(\mathbf 8_v\), \(\mathbf 8_{+}\), and \(\mathbf 8_{-}\) are the vector
and two chiral eight-dimensional representations of
\(\operatorname{Spin}(8)\). For a single unit key, the induced bind/unbind map
is an exact isometry. Multiple pairs superposed in one eight-dimensional
channel, however, suffer severe crosstalk. Capacity must therefore come from
explicit multiplicity slots, temporal structure, or a decoder outside the
recurrent scan—not from vague high-dimensional-memory claims.

### Decisive evidence

Compare the triality recurrence against a same-width direct slot memory,
delta-rule memory, bilinear fast weights, Householder transport, and diagonal
complex recurrence. Triality earns architectural credit only if it improves
state efficiency, extrapolation, optimization reliability, or throughput at a
matched task and budget.

## Comparison table

| Candidate | Primary gate | Legitimate positive claim |
|---|---|---|
| Spontaneous gauge discovery | Causal, held-out group-action extraction | A trained model uses an emergent group-like mechanism |
| Syntactic-monoid quotient | Exact quotient isomorphism | The recurrence realizes the minimal language quotient |
| Infinite-group compiler | Adversarial long-horizon recovery | Constant state computes a selected group invariant |
| Triality binding | Matched-baseline retrieval and streaming efficiency | Triality is useful for a specified binding regime |

The immediate priority remains the last row because its algebraic contracts and
matched baselines already exist locally. The other three are valuable only if
their first experiments are designed to falsify the strongest interpretation,
not to decorate it.
