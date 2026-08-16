# The Mathematics in Plain Language

This chapter explains the current Dirac--Gram and triality results without
assuming advanced university mathematics. It distinguishes carefully among
four kinds of conclusion:

- theorems proved for an entire continuous family;
- finite classifications verified exhaustively;
- numerical searches that found no counterexample;
- conjectures that remain open.

## The basic question

A transformation in \(\operatorname{Spin}(8)\) has 28 local degrees of freedom.
Imagine trying to identify such an unknown transformation using only five
carefully chosen probes. Each probe reveals part of the action. The design
problem is to choose probes whose information overlaps as little as possible,
so that together they distinguish all nearby transformations.

The balanced \(\operatorname{Spin}(8)\) arrangement has the information score

\[
\det I=\frac{81}{1024}.
\]

The determinant is a local volume measure. A large value means that nearby
transformations produce clearly distinguishable measurements. A zero value
means that at least one infinitesimal direction of motion is completely
invisible. This is a local conditioning statement; by itself it does not
describe finite-noise estimation error or guarantee that an optimizer will
find the correct transformation.

The value \(81/1024\) now has a direct structural explanation. At the balanced
point, the 28-dimensional information operator separates into four independent
blocks. Their determinant contributions are

\[
\frac14,\qquad\frac9{16},\qquad\frac9{16},\qquad1.
\]

Therefore

\[
\det I
=\frac14\cdot\frac9{16}\cdot\frac9{16}\cdot1
=\frac{81}{1024}.
\]

The two factors \(9/16\) arise from blocks related by a fixed signed-permutation
symmetry, so their equality is forced rather than accidental. At either
calibrated endpoint of the Cayley parameter, one eigenvalue in each of three
blocks vanishes. This is why the observation map loses exactly three ranks.

Why is one Cayley number enough? A classical orbit theorem says that, after
the singleton probe is fixed, the underlying oriented four-plane is classified
by one signed Cayley coordinate \(c\in[-1,1]\). The repository then checks
exactly that the stabilizer of the four-plane acts as the full \(SO(4)\) inside
that plane, so dividing the four probes into two orthogonal pairs introduces no
second continuous coordinate. Reversing one pair changes \(c\) to \(-c\)
without changing the information operator. The observable family is therefore
parameterized by \(z=c^2\in[0,1]\). The global orbit classification is a cited
classical theorem; the split-isotropy and information calculations are the
exact local contributions of this repository.

## Why perpendicular probes are the natural benchmark

Let the four probe vectors be the rows of \(X\). Their Gram matrix is

\[
G=XX^{\mathsf T}.
\]

The entries of \(G\) are the pairwise inner products of the probes. If the rows
are orthonormal, then \(G=I_4\) and \(\det G=1\). As the probes become
correlated, the volume of the parallelepiped they span decreases, so \(\det G\)
decreases. If one probe becomes redundant, \(\det G=0\).

The Dirac--Gram conjecture says that correlation cannot secretly improve the
information score after the natural volume penalty is included:

\[
\det I(X)\leq \det(XX^{\mathsf T})^3\det I(Q).
\]

When the four rows are independent, \(Q=G^{-1/2}X\) is the orthonormal frame
obtained from \(X\) by removing this purely geometric distortion. On the
singular boundary \(G^{-1/2}\) is undefined; the exact theorem families use the
equivalent polynomial inequality and its continuous boundary extension. No
pseudoinverse is being hidden. The exponent 3 is not an adjustable penalty: it
is the order forced by the rank-loss geometry when one probe becomes redundant
and three independent observation directions disappear.

## The repeated-sensor gauge

The next bridge correlates two probes in the same triality representation. At
first this appears to introduce a new geometric degree of freedom, but part of
it is only a choice of coordinates.

Place the two probe vectors in a two-row matrix \(X\). Their combined
information depends only on the covariance \(X^{\mathsf T}X\). Replacing \(X\)
by \(UX\), where \(U\in O(2)\) mixes the two rows, leaves that covariance—and
therefore the information operator—unchanged.

If the two unit probes have inner product \(r\), their normalized sum and
difference directions are orthogonal, with squared weights \(1+r\) and
\(1-r\). Thus two correlated, equally weighted probes are exactly equivalent
to two orthogonal modes with unequal weights.

This does not prove the next inequality. It identifies the right variables:
the proof should use a rank-two covariance and its energy split, rather than
carry a redundant choice of sensor labels through a much larger polynomial.

## What has been proved

The inequality is proved for three increasingly large continuous families:

1. the signed star family, with three active correlations;
2. the Cayley-null edge family, with four active correlations;
3. the variable-Cayley one-edge family, which keeps those four correlations
   while allowing the Cayley angle to vary.

For the first family, the equality cases are now known exactly. After the
natural Gram-volume factor is divided out, equality occurs only when the star
is orthonormal or when the Cayley calibration reaches its singular endpoint.
Every other signed-star arrangement loses a strictly positive amount of
information relative to the bound.

The proof layers are described later. In brief, rational reconstruction and
off-grid determinant identities establish the target polynomials, while
Bernstein certificates establish their signs throughout the relevant
continuous parameter boxes. These are theorem certificates, not conclusions
drawn from a dense floating-point sample.

## A proof idea that failed

It was tempting to simplify a general correlated frame by turning off three
extra correlations one at a time. An exact rational counterexample shows that
this can make the normalized information score *smaller* by a factor of about
3.2168.

This does not disprove the main inequality. It disproves only that shortcut.
The lesson is simple: correlations interact, so removing them one by one can
move information around before it removes it.

## Why five probes can determine the whole action

The older experiment checked a local question: if we nudge the unknown action
by a tiny amount, do five measurements detect every possible nudge? They did.
That still left a loophole: perhaps a completely different, far-away action
could give the same five answers.

Triality closes that loophole for one exact arrangement. Start with one vector
probe and four spinor probes. Whenever two known objects are combined by the
triality multiplication rule, the result must also be fixed by any action that
fixed the original probes. Repeating this rule is like solving a Sudoku: every
known entry forces more entries.

For the exact probes

```text
vector:          e0
positive spinor: e0, e1, e2, e4
```

the forced vectors eventually form a complete basis of all three
eight-dimensional triality spaces. Each basis determinant is \(-1\), so none
of the directions is missing. Any compatible \(\operatorname{Spin}(8)\) action
that fixes the five probes must also fix everything generated from them by
triality multiplication. Because that closure contains a basis in all three
representations, the action is the identity in each representation and has no
residual group ambiguity.

Remove `e4` and the closure stops at four dimensions in each space. Three
independent rotations remain completely invisible. Their exact commutators are
those of `su(2)`, so this is a continuous family of wrong answers, not rounding
error or failed training.

This proves a global five-versus-four separation for the displayed probes.

There are 70 ways to choose four coordinate spinors. Finite enumeration checks
all of them. Fifty-six choices force all eight directions. Fourteen choices get
stuck in four dimensions. Those fourteen are the blocks of a famous binary
error-correcting code: the extended Hamming `[8,4,4]` code. Every set of three
coordinates belongs to exactly one exceptional block.

This does not mean the memory is performing error correction. It means the
geometry of the exceptional probe choices has the same exact combinatorial
pattern as that code.

The larger pattern is now known for every coordinate probe. Give the three
triality spaces three different two-bit labels and each of their eight
coordinates a three-bit label. Every coordinate probe then has a five-bit
barcode. Triality multiplication combines barcodes by XOR: a switch is on in
the answer exactly when it was on in one input but not both.

The finite coordinate atlas contains 52,752 multiview choices of four or five
probes, all of which were enumerated. Four barcodes can span at most four
independent binary directions,
so none sees the whole five-dimensional barcode space. Five probes see it all
exactly when their five barcodes are independent. The remaining invisible
continuous symmetry has dimension 8, then 3, then 0 as binary rank rises from
3 to 4 to 5. Exact bracket and Killing-form checks identify these stages as
`SU(3)`, `SU(2)`, and no continuous stabilizer.

This completely classifies the coordinate case. Arbitrary non-coordinate
probes form a continuous problem. That orbit theorem is now also proved.

Every way of distributing four probes among the three views has at least three
independent relationship measurements that Spin(8) cannot change. Those
relationships force at least three invisible rotation directions. Exact
representatives show the bound is sharp, and the compact principal-orbit
theorem guarantees that unusual probe choices can only have more symmetry,
never less. Thus no four-probe arrangement works, coordinate or continuous.

For five probes, the coordinate atlas supplies a trivial-stabilizer example in
every allocation that uses at least two views. Compact orbit theory then says
that a trivial stabilizer occurs on an open dense set: almost every such mixed
design works globally. Five probes in only one view still leave three rotations
unseen.

## How bilinear memory can remain parallel

Suppose two recurrent streams are updated independently. After both have been
computed, combine their current states with any bilinear rule and use that as
the input to a third recurrent stream. Each layer is still an ordinary affine
scan, so the construction needs two parallel scan stages and a fixed recurrent
state.

An exact proof adds the product of the two source states as temporary
coordinates. Those product coordinates make the whole update one linear
matrix multiplication. They are a proof device, not part of the streaming
cache.

The arrow direction matters. If the third stream feeds back into one source,
the polynomial degree grows forever. If it feeds both sources, the degree
doubles at every step. So triangular coupling is not an arbitrary engineering
choice; it is the boundary that preserves a finite exact scan.

## What the memory experiments found

The special triality binding can move information consistently between three
different eight-dimensional views. It does **not** create extra ordinary slot
capacity. When a direct slot memory and a triality-coded slot memory receive
the same correct address, both overwrite and recall exactly. Triality earns
scientific credit when the task asks for a missing cross-view action or
retrieval law, because one shared Spin(8) action constrains all three views.

A clean ten-seed follow-up tested that distinction directly. Both the shared
and independent action families fit every supplied observation, and the
independent family had three times as many adjustable action coordinates. On
the unobserved third view, however, the shared family remained essentially
perfect through length 2,048 while the independent family averaged about
0.543 cosine. Giving both families the same router did not change the gap, and
replacing direct slots with standard delta memory changed neither result. The
benefit therefore comes from the correct shared relationship between views,
not from extra storage or from the delta update rule.

The largest practical gain came from routing. First summarize long history in
coarse blocks, select a small set of promising blocks, and then read their fine
memory. On the frozen synthetic tasks this improves both direct and delta-rule
retrieval. Hard routing is perfect only while the learned alias groups remain
separable; sufficiently corrupted keys expose a sharp failure frontier.

A later test makes this less toy-sized: 64 overlapping keys are grouped into
eight blocks. One router shared across the three views fills in combinations
that were deliberately missing from each separate router's examples, and block
selection improves long-stream recall in all ten frozen runs. Merely gathering
the selected block was initially slower than a dense GPU operation because it
launched many tiny operations. Fusing routing, write, and read into one kernel
reversed that result: the sparse recurrence took about 50 microseconds and was
roughly 7.6--13 times faster than the corresponding dense eager operations on
the tested GPU. This is an inference-kernel result, not a training or capacity
theorem.

A second gain comes from changing coordinates. If the values are transported
by a changing invertible matrix, maintain the cumulative moving frame and
express every write in that frame. The difficult transported update then
becomes an ordinary gated-delta recurrence that an optimized FLA kernel can
run. Float32 matrix products and solves are essential before the fp16 kernel;
the naive fp16 inverse-transpose version becomes inaccurate at long length.

Spin(9) adds an exact way to bind a 9-dimensional key into a 16-dimensional
spinor and unbind it again, plus a natural Hopf-like coarse coordinate. This is
useful as a possible router, but expanding 9 numbers to 16 is not a same-width
memory-capacity advantage. None of these results requires the unrestricted
Dirac--Gram inequality: that is a separate global sensing theorem.

## The completed one-edge gate

The next family allows the Cayley angle to vary as well. Exact symmetry reduces
sixteen possible sign choices to four polynomial sectors. Those four numbers
are the eigenvalues of a symmetric 4 by 4 matrix. Proving the matrix is
positive is equivalent to proving every sign choice at once.

The lower-order matrix tests are proved. The difficult final test was the
determinant of that 4 by 4 matrix.

In ordinary cube coordinates its polynomial has 257,049 Bernstein controls and
only 21 are negative. A negative control is not a negative function value; it
means that particular coordinate net is too loose to prove positivity.

The useful change of coordinates is

\[
u=t y,\qquad v=t(1-y).
\]

Think of `t` as distance from a corner and `y` as the direction used to leave
the corner. This is called a Duffy chart. It separates "how far" from "which
way", exactly where ordinary boxes kept zooming forever.

The exact replay confirmed that every difficult control lives in the first two
distance layers. Those layers factor into already nonnegative pieces. All
layers from 2 through 24 are exactly nonnegative, and the complementary
triangle has no negative controls at all. The variable-Cayley theorem is
therefore proved on this whole family.

## From the gauge to the second residual edge

The first boundary factor of that proof is now exact. When the Cayley angle
reaches either endpoint, the observation system loses exactly three ranks.
That forces its information determinant to vanish with six powers of the
distance from the endpoint. In formulas, every orientation sector contains
`(1-c^2)^3`. The same factor occurs in the proposed upper bound, so it can be
cancelled before tackling the remaining polynomial.

There is also an exact reason that the remaining square roots fit together.
When every signed coordinate and every circle complement is audited together,
the 4,096 possible sign patterns collapse to a 512-element gauge group with
only eight allowed polynomial characters. Each visible orientation sector has
one unique hidden complement parity. Thus every apparent radical is a fixed
front factor; what remains is an ordinary polynomial in six squared numbers.

The first of those eight polynomials has now been recovered completely. Two
different exact 4-by-4-by-4-by-4-by-4-by-4 grids gave the identical rational
formula, and 32 new points matched calculations made directly from the
original matrices. The safe degree bound allowed 4,096 coefficient positions;
only 243 were used.

Then the formula revealed two extra pieces of structure. First, all 243 terms
share the factor `1-a^2`, reducing the real core to 162 terms. Second, every
term in which either of the last two coordinates appears also contains

```text
(1-d^2)(1-e^2)(1-g^2).
```

After removing that product, only a 28-term formula remains, and it is linear
in every squared coordinate. The subsequent complete reconstruction found the
same screening rule in **all eight** sectors. In everyday language:
information added late in this triangular coordinate chain can affect the
even polynomial core only if every intermediate direction is still available.
This is an exact dependency law, not a numerical correlation.

There is an even simpler picture. The eight sign cases form four pairs. In
each pair, flipping the new coordinate leaves four ingredients alone and
reverses the other four. The unchanged half is exactly the earlier one-edge
problem that has already been proved. So the new theorem problem is two
four-by-four blocks built as controlled deformations of a solved case—not
eight unrelated monsters.

The first danger point has also been settled. Starting from four perfectly
orthogonal probes, turning on the new correlation always makes the determinant
smaller, never larger. After exact simplification the difference is a positive
two-variable polynomial whose nine Bernstein coefficients are all positive;
the smallest is still `103`. Other exceptional boundary arrangements remain
to be checked.

On two outer faces, the remaining 162-term formula collapses to products of
only a few simple factors. One face is always nonpositive and another is always
nonnegative. That catches a tempting but wrong proof plan: the eight internal
sign components do not each need to be positive. They are like ingredients
with positive and negative flavors; only the eight final orientation totals
must be nonnegative.

## The second edge: what is now exact

The first one-edge theorem has a line of equality: a family of perfectly
balanced arrangements where the upper bound is attained. Before attempting a
global proof with a second correlation, we asked the most dangerous local
question. Could an arbitrarily small second edge tip one of the zero margins
below zero?

The exact answer is no along the entire orthonormal equality line. Almost
everywhere, the margin curves upward quadratically. At one Cayley endpoint a
single direction is quadratically flat, but the exact fourth-order term is

\[
64s^2(2-s^2),
\qquad 0<s\leq1,
\]

which is strictly positive. In geometric language, the apparent flat floor is
not another hidden equality corridor; a finer measurement shows that it rises.

A plausible quadratic shortcut nevertheless fails away from that equality
line. An exact rational example makes its proposed auxiliary expression
negative. This does not disprove the target inequality. It tells us that the
full finite-edge dependence, rather than only its first two derivatives, is
essential.

That full dependence has now been simplified exactly. If \(i\) is the new edge,
set

\[
x=i^2,\qquad y=\sqrt{1-x},\qquad 0\leq y\leq1.
\]

Each pair of signed answers becomes

\[
m_\pm=L(y)\pm\sqrt{1-y^2}\,R(y).
\]

The two answers are both nonnegative exactly when their midpoint is
nonnegative and their half-separation is no larger than that midpoint:

\[
m_+\geq0\ \text{and}\ m_-\geq0
\quad\Longleftrightarrow\quad
L(y)\geq\sqrt{1-y^2}\,\lvert R(y)\rvert.
\]

Because both sides of the right-hand inequality are then nonnegative, squaring
is reversible. Hence the condition is equivalent to

\[
L(y)\geq0,
\qquad
L(y)^2-(1-y^2)R(y)^2\geq0.
\]

The first expression has degree six in \(y\); the second has degree twelve.
Thus a nested-radical, eight-sign problem has become eight ordinary polynomial
sign problems. This conversion is an exact algebraic identity. A historical
GPU search over 851,968 interior and boundary samples found no negative value.
The sign question on that frozen two-edge family has since been settled by a
34-region exact Bernstein atlas, including exact integer checks wherever
floating-point interval bounds were inconclusive.

The last residual correlation has also been incorporated algebraically. The
full problem becomes sixteen ordinary coefficient maps in seven squared
variables. Two completely separate rational grids reconstructed the same maps
from 2.5 million exact determinants, and 32 new rational points checked all
sixteen maps again. This proves that the formulas are the right formulas; it
does not yet prove that every one has the right sign everywhere.

Near the perpendicular configuration, however, the sign is now understood
exactly. The leading change is nonnegative for every Cayley angle. At the one
endpoint where that leading term becomes flat, the next term is

\[
128(p^2+q^2)^2,
\]

which is positive unless there was no movement at all. A more complete zoom
that includes every way of approaching the flat endpoint becomes a sum of
plain squares. In addition, an exact 588,245-coefficient check proves that the
main amplitude is at least as large as the combined root-mean-square size of
all fifteen fluctuating modes throughout the entire seven-dimensional box.
The proof had to isolate four troublesome coefficients onto two small boundary
faces and certify those faces in coordinates adapted to their triangular
shape. Equivalently, the sixteen orientation scores cannot fluctuate too much
on average. One bad orientation can still hide inside an average bound,
however, so this is not yet the unrestricted inequality.

There is one more exact consequence. Put the sixteen orientation scores into
one polynomial whose roots are their negatives. The first four coefficients
of that polynomial are now proved nonnegative. The third coefficient does not
need a four-million-term expansion: a convolution inequality for the sixteen
Walsh sign patterns bounds all 35 cubic interactions at once. Twelve
coefficients remain, so this is a real reduction of the open problem rather
than a claim that the open problem has disappeared.

The most dangerous-looking endpoint now has a further exact safeguard. On one
complete four-variable boundary, only three fluctuation patterns survive.
Those patterns behave like the three nonzero moves in a four-state toggle
system. Instead of checking sixteen square-root formulas separately, the proof
puts the four distinct scores into one symmetric \(4\times4\) matrix. It then
checks every principal minor of that matrix exactly. All are nonnegative, so
the matrix has no negative eigenvalue and none of the sixteen scores can be
negative there. This settles the whole boundary face, but not neighboring
faces or the seven-dimensional interior.

The neighboring boundary is harder but now partly understood. Eight
fluctuation patterns survive there, and their on/off labels form a three-bit
toggle system. Splitting on one bit turns its (8\times8) score matrix into
two coupled copies of the four-state matrix above. An exact change of basis
reduces the question to two smaller matrices. The first smaller matrix is now
proved nonnegative throughout the whole five-variable boundary. The first
scalar test for the second matrix is also proved by rewriting it as the global
average-energy margin plus explicit squares. Its larger determinant tests are
still open. There is one more exact foothold: on a particular corner shared by
all three quadratic tests, each test becomes the same perfect square and is
therefore automatically nonnegative. In ordinary language: the neighboring wall has been reduced to a
much smaller lock, and half of that lock is open, but the door itself has not
yet been proved safe.

## Why the crashes did not count as evidence

A computer stopping does not make a theorem true or false. The new replay tool
splits the calculation into five stages and links them with SHA-256 hashes. A
completed stage can be checked and reused after a restart. The GPU is used only
to search quickly for counterexamples. The final signs came from exact CPU
integer arithmetic.

## What Spin(9) changes

The Spin(8) triality programme uses three different eight-dimensional views.
The Spin(9) extension asks a different question: how much can be learned from
several probes in one faithful sixteen-dimensional spinor view?

Nine symmetric matrices \(P_0,\ldots,P_8\) satisfy the exact rule

\[
P_iP_j+P_jP_i=2\delta_{ij}I.
\]

This is the finite matrix form of a nine-dimensional Clifford algebra. Three
generic spinors are enough to identify a shared Spin(9) transformation; two
are not, because a continuous \(\operatorname{SU}(3)\) ambiguity remains.

The newest reduction says that the information carried by several probes does
not remember the probes one by one. It remembers only their combined
"shadow"

\[
M=\sum_rs_rs_r^{\mathsf T}.
\]

This \(16\times16\) matrix is called the frame operator. For three unit probes
it is positive, has trace three, and has rank at most three. Conversely, every
matrix with exactly those properties can be built from three unit probes.
The global design problem is therefore a precise low-rank matrix problem, not
an unstructured search over 48 probe coordinates.

There is also an exact blind spot. Symmetric \(16\times16\) matrices split into
three Clifford pieces of sizes

\[
1+9+126=136.
\]

The information operator sees the one-dimensional scalar piece and all 126
four-form directions, but it cannot see the nine vector directions at all.
This is a structural gauge, not a numerical failure.

The 126 visible numbers are not an arbitrary list. They fit together as a
single geometric object called a four-form. That four-form acts on the 36
possible rotation planes in nine dimensions, and the full information matrix
is exactly one quarter of

\[
\text{trace of }M\times\text{identity}
\;-\;
\text{four-form action}.
\]

In other words, the determinant problem is really a spectral problem: choose
an admissible four-form so that its 36 plane-action eigenvalues stay as evenly
spread away from the trace threshold as possible. This is the Spin(9)
counterpart of letting the Dirac/Clifford algebra reveal the correct
coordinates before attempting a large polynomial proof.

If the rank-three restriction is removed, the relaxed problem can be solved
completely. The best possible information matrix is perfectly uniform:

\[
I=\frac34I_{36},
\qquad
\det I=\left(\frac34\right)^{36}.
\]

Every relaxed optimizer has rank eight or sixteen. Since a frame made from
three probes has rank at most three, no exact three-probe design can attain
that ideal score. This proves that a genuine exact-design gap exists. It does
not yet identify the best rank-three point; that final global optimization is
still open.

The edge of this problem is unusually sharp. Two generic spinors leave eight
directions invisible. If a genuinely new third probe is turned on with small
strength \(\varepsilon\), each missing observation amplitude grows like
\(\varepsilon\). Information is a squared amplitude, and there are eight
missing directions, so

\[
\det I(\varepsilon)
=C\varepsilon^{16}+\text{higher-order terms},
\qquad C>0.
\]

The power sixteen is therefore forced by rank loss; it is not a curve fitted
to numerical data.

The best symmetric three-probe arrangement is now known to be a genuine
strict local peak, not merely the best point on a specially chosen curve. The
full nearby problem has 44 directions. Thirty-three only rotate the whole
configuration and cannot change the score. The remaining eleven split into
one curve direction and two five-direction packs. The two packs can interact,
so checking each one alone would be unsafe. Exact arithmetic computes their
coupled \(2\times2\) curvature matrix and proves both of its eigenvalues are
negative. Therefore every sufficiently small change that is not just a shared
Spin(9) rotation lowers the information determinant.

This is a local theorem. A distant, differently shaped three-probe design
could still have a larger score; excluding that possibility is the remaining
global problem.

One substantial finite-distance family is now excluded exactly. At the
Cayley-null plane, the five non-curve directions form the space of traceless
symmetric \(3\times3\) matrices. Rotation reduces their shapes to a radius and
one cubic parameter. Exact modular determinants recover the full two-parameter
formula, and six small Bernstein rectangles prove that every pure deformation
in those five directions stays below \(101/100\) of the Cayley-null
determinant. The algebraic symmetric candidate is strictly higher. This closes
the entire pure spin-two family, but not mixtures with the curve direction or
the global nonpolar quotient.

## Honest scoreboard

| Claim | Status |
|---|---|
| Balanced sensor invariants \(81/1024\), \(35\), and \(43\) | Proved on the complete orthonormal balanced information family by exact block identities plus the classical orbit classification |
| Generic mixed-view five-probe tuples have trivial shared stabilizer | Proved from exact witnesses and invariant ranks plus the compact principal-orbit theorem; exceptional tuples are not all classified |
| Displayed five-probe tuple has trivial stabilizer | Proved |
| Displayed four-probe subset has an \(\mathfrak{su}(2)\) stabilizer | Proved |
| All multiview coordinate four/five-probe closures follow \(\mathbb F_2^5\) span | Exhaustively verified over the finite coordinate atlas |
| Every continuous four-probe sensor is insufficient | Proved by invariants and principal orbits |
| Every mixed five-probe allocation is generically globally free | Proved |
| Generic triangular bilinear drive has a finite exact staged scan | Constructively proved |
| Signed star Dirac--Gram inequality | Proved |
| Cayley-null four-correlation edge inequality | Proved |
| Complete \(u_a=u_h=0,c^2=1\) endpoint face | Proved by an exact Klein-four matrix principal-minor certificate |
| Adjacent \(u_a=0,c^2=1\) endpoint face | Exact eight-sector reduction; first Schur block and scalar, quadratic, and cubic second-block minors proved; determinant coordinate boundary complete; all margins strictly positive on the extended core \([1/8,7/8]^5\) by a 2,140-leaf atlas; width-\(1/8\) collars remain open |
| Removing residual correlations is always helpful | Disproved |
| Variable-Cayley one-edge inequality | Proved |
| Repeated-view covariance gauge | Proved by symbolic identity |
| Common two-edge Cayley boundary factor \((1-c^2)^3\) | Proved by symbolic factorization |
| All eight two-edge sector polynomials and endpoint factors | Reconstructed and verified in exact arithmetic |
| Universal two-edge flag law and paired four-block reduction | Proved by symbolic identity |
| Two-edge stability transverse to the orthonormal equality line | Proved |
| Finite two-edge radical-to-polynomial reduction | Proved |
| Degree-six and degree-twelve finite two-edge gates are globally nonnegative | Proved on the complete frozen `h=0` family by a 34-leaf triangular Bernstein atlas with exact integer fallbacks |
| Balanced five-query sensor is a strict local optimum after removing rotations | Proved |
| All 35 one-probe coordinate-circle deformations | Proved |
| Equal balanced sensor is best with fractional measurement weights | Disproved by exact counterexample |
| Eight-probe isotropic approximate design is globally best | Proved |
| 24 coloured sensors are spectrally optimal as an ordinary subspace packing | Disproved by exact comparison |
| 24 coloured sensors attain the chordal simplex bound | Disproved by exact comparison |
| 24 coloured sensors are chordally optimal | Open; missing the bound is not a proof |
| Variable-Cayley two-edge inequality | Proved on the complete frozen `h=0` family by an exact 34-region atlas |
| Full seven-variable margin formulas | Reconstructed exactly in sixteen symmetry sectors from two disjoint grids and checked at 32 new exact points |
| Full tangent cone and singular-endpoint weighted leading form | Proved nonnegative; the endpoint form is positive away from its origin |
| Two dangerous coupled modes for (0\le c^2\le2/3) | Controlled by an exact 588,245-coefficient Bernstein certificate |
| Unrestricted seven-parameter inequality | Open |
| Global best possible five-query design | Open |
| Three generic Spin(9) spinors identify a shared action | Proved by a stabilizer-chain argument with independent exact rank witnesses |
| Spin(9) frame-operator reduction and nine-dimensional information gauge | Proved |
| Spin(9) information as a four-form spectrum on rotation planes | Proved |
| Spin(9) convex approximate-design optimum | Proved; unattainable by three exact probes |
| Spin(9) transverse two-to-three-probe boundary order | Proved to be sixteen |
| Symmetric Spin(9) three-spinor candidate is a strict local optimum on the full rank-three frame space | Proved exactly modulo Spin(9); global optimality remains open |
| Every pure Spin(9) \(V_5\) graph over the Cayley-null plane stays below \(101/100\) of Cayley-null | Proved by deterministic modular lifting and a six-cell strict Bernstein atlas |
| Every graph in the coupled Spin(9) \(V_1\oplus V_5\) normal slice stays at or below \(21/20\) of Cayley-null | Proved by a 312-leaf compact atlas with eight exact local handoffs and a 22-prime characteristic-zero lift over both roots of 2; exact candidate optimality and the unrestricted quotient remain open |
| Global best possible exact three-spinor design | Open |

The central methodological result is a proof strategy rather than a single
determinant value: exploit triality symmetry before expanding signs, use exact
rank loss to predict determinant factors, and replace ordinary box coordinates
with boundary-adapted charts when positivity is concentrated near a singular
face.
