# Learned endpoint-manifold compiler: prospective contract

Date fixed: 2026-08-03, after the endpoint curriculum seed-0 pass and before
any pre-retraction endpoint-manifold compiler training result. The exact
post-retraction seed-0 checkpoint was used only to validate that the proposed
finite-table reconstruction code accepts the exact anchor and rejects the
three non-group channels; it is not evidence for the gate below.

Prospective correction, still before any blind GPU result: the first draft
specified L4 words. A direct coverage audit showed the held-out-pair L4
language reaches only 48/60 states, making that corpus structurally
insufficient. The frozen corpus below was changed to L8, which covers all 60.
This correction is recorded rather than silently replacing the draft.

The corrected post-retraction ten-seed pipeline validation selects the known
exact anchor in 10/10 checkpoints. Seeds 6 and 8 also contain auxiliary
channels whose nearest-center products form the same exact abstract table, so
table validity is not required to be unique; the already-frozen alignment
score ranks the accepted candidates. This is a pipeline validation on
previously retracted models, not a blind-discovery result.

## Gate

Remove the separate 1,148-label endpoint query compiler. Train the rotor model
from the frozen endpoint curriculum, let its token actions move unconstrained
in their tangent charts, and infer a shared finite representation manifold
from learned word actions plus endpoint labels that the neural task has already
consumed.

No Cayley table, permutation action, token-to-element map, character vector,
irrep branch, inverse-token matching, representative extension, or additional
endpoint label enters the compiler.

## Frozen compiler

1. Reuse the first 64 L8 curriculum batches: 16,384 complete words and their
   one endpoint label each. These labels are part of the fixed 512,000 neural
   training labels; compiler overhead is zero additional labels.
2. At step 850 and every 50 steps thereafter through step 1,500, extract each
   channel's four learned 3x3 token rotations.
3. Compose the learned rotations along every saved word. For each of the 60
   anonymous endpoint labels, average its word products and project the mean to
   SO(3), producing 60 candidate class centers.
4. Infer identity as the center nearest the identity matrix. For every ordered
   center pair `(i,j)`, assign `j @ i` to its nearest center.
5. Reject unless the assigned table has permutation rows and columns, is
   exactly associative, has the inferred two-sided identity, and the four
   inferred token elements generate all 60 states.
6. Extract exact real 3D irreps from only that recovered regular table. Jointly
   align the learned token family to the nearest candidate and accept only if:

   - alignment RMS <= 0.08;
   - runner-up minus winner RMS >= 0.20;
   - token commutator separation >= 0.50;
   - minimum nearest-center assignment gap >= 0.10;
   - maximum center-product residual <= 0.20.

7. Retract the complete token family through the accepted single shared
   conjugacy. Tokens are never independently normalized or rounded.

The exact recovered-table isomorphism to A5 is computed only post hoc for
scoring. It is not an acceptance input.

## Frozen outcome taxonomy and threshold audit

This section was fixed after the seed-0 smoke and before any seed-1--9 result
was inspected. Seed 0 is comfortably inside every numerical boundary:
minimum assignment gap `0.506` versus the `0.10` floor, and maximum product
residual `0.0269` versus the `0.20` ceiling. The formal cohort will nevertheless
report the complete per-seed distributions of alignment RMS, runner-up gap,
commutator separation, assignment gap, consistency RMS, product RMS, and
maximum product residual, including rejected candidates. Thresholds and
compilation times will not be changed after seeing those distributions.

Every seed is classified into exactly one of three primary outcomes:

1. **accepted and A5-isomorphic**: compiler success, subject to every downstream
   numerical and behavioral gate below;
2. **numerically or structurally rejected**: compiler failure, with the failed
   margins and the best candidate reported;
3. **accepted but not A5-isomorphic**: wrong-structure compiler failure. This
   is distinct from failure of neural training to form any clusterable action.

An accepted wrong structure is never counted as a pass even if its decoder
behavior is strong. Conversely, a near-threshold rejection remains a failure
under this frozen experiment, but will be reported as a near miss rather than
silently merged with a gross failure.

## Frozen experiments and gates

Run seed 0 as an implementation smoke. If it fails, record the failure and do
not tune thresholds or compilation time within this experiment. If it passes,
run seeds 0--9 unchanged.

Every formal seed must:

1. use exactly zero additional compiler endpoint labels;
2. recover one accepted 60-state finite table by step 1,500;
3. be post-hoc exactly isomorphic to the hidden A5 environment;
4. compile below `1e-10` invariance and homomorphism RMS;
5. reach >=90% at every dense L16--L256 checkpoint on the original and
   untouched class-59 alphabets;
6. reach >=90% at L4096 on both and L16384 on class 59;
7. pass the 13-point class-59 L4096--L16384 sweep.

The formal headline is the three-way outcome distribution plus the all-gate
pass count. A success is stronger than the
1,148-query compiler result only along the compiler-supervision axis; all
other claim boundaries remain.

## Claim boundary

This is not unsupervised algebra discovery. The neural curriculum still uses
exact anonymous endpoint classes, includes L1/L2 short words, knows there are
60 output classes, and is generated by a hidden exact A5 environment. The
compiler recovers multiplication from the learned action geometry and reused
endpoint classes; it does not infer the existence or cardinality of the label
space from raw observations, tolerate noisy equivalences, or establish a
language-model mechanism.
