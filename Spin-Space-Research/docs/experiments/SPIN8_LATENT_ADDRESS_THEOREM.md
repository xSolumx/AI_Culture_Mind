# Birkhoff address-completion theorem for scan-compatible memory

Date: 2026-08-03.

This note isolates the exact mechanism tested by
`spin8_learned_address.py`.  The ingredients are classical simplex geometry
and the Birkhoff polytope; the candidate contribution is their use as a joint
family retraction for latent SSM addresses, not a claim that the underlying
theorem is new.

## Setup

Let `K = H`.  Logical key `k` has an address row

\[
 r_k \in \Delta^{H-1}, \qquad r_{kh}\geq 0,\quad \sum_h r_{kh}=1.
\]

From zero memory, write a nonzero payload `b` with row `r_k`, then query with
the same row.  Each slot contains `M_h = r_{kh} b`, so the linear query is

\[
 \widehat b = \sum_h r_{kh}M_h = \lVert r_k\rVert_2^2 b.
\]

For triality memory, `b = bind(p,n)` and exact unbinding by the unit key `p`
reduces the endpoint to the same equation.  Supplied orthogonal transport also
cancels from squared endpoint error.  Thus the argument applies identically
to direct and triality-bound memory.

## Theorem

Zero single-key endpoint error for every nonzero payload forces every address
row to a simplex vertex.  If rows are independently normalized, there are
`H^K` such zero-loss families and collisions are permitted.  If the complete
`K by H` family is doubly stochastic, every zero-loss family is a permutation
matrix and collisions are impossible.

## Proof

Exact retrieval requires `||r_k||_2^2 = 1`.  A probability vector has squared
norm at most one, with equality only at a vertex of the simplex.  Therefore
each zero-loss row is one-hot.

With independent row normalization, each of the `K` rows may choose any of
`H` vertices, giving `H^K` solutions.  When `K = H`, only `H!` are
collision-free, a fraction

\[
 \frac{H!}{H^H}.
\]

For `H=8` this is `40320 / 16777216 = 0.0024032593`.

If the family is also column-stochastic, one-hot rows place exactly `H` unit
entries into `H` columns whose sums must each equal one.  Every column
therefore contains exactly one unit entry.  The matrix is a permutation.

## Consequence for the held-out test

The training objective need never contain two logical keys in one episode.
It supplies the pressure that moves each row to a vertex.  Joint family
normalization removes the otherwise invisible collision slack.  Once the
learned matrix is a permutation, writes and queries for different keys occupy
disjoint slots, so arbitrary unseen multi-key interleavings and overwrites are
exact, up to numerical transport error.

This is a structural generalization result.  It does **not** show that a
content encoder can infer semantic key identity, nor that triality is better
than a same-width direct memory.  Those require the next alias/content and
baseline gates.

## Associative-scan compatibility

For input-dependent but state-independent `r_t`, action `A_t`, and payload
`b_t`, one step is the affine map

\[
 M_t = (1-r_t)\odot(A_tM_{t-1}) + r_t\odot b_t.
\]

Affine-map composition is associative.  Joint normalization is performed on
the token-action family before sequence execution; it does not inspect the
running state.  The recurrent state therefore remains `H x 8`, independent
of context length, and the same transitions admit a logarithmic-depth prefix
scan during training.
