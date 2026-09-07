# Inter-tile communications in KTIR

**Scope:** Seven ops — one production op, `ktdp.inter_tile_produce`, and
six delivery ops: `ktdp.inter_tile_consume`, `ktdp.inter_tile_reduce`,
`ktdp.inter_tile_reduce_scatter`, `ktdp.inter_tile_gather`,
`ktdp.inter_tile_all_to_all`, and `ktdp.inter_tile_scatter`. Together they
cover the six inter-core communication patterns: broadcast, all-reduce,
reduce-scatter, gather, all-to-all, and scatter.

**Organization.** §0 is a plain-language orientation: what a tile is, why
data has to cross tiles, and what the seven ops are for. The normative
specification starts at §1. The delivery ops share almost all of their
machinery. That machinery is stated once, in §3 (operand, consumer set,
local index, dependency attribute, combiner, synchronization, result
semantics), §4 (type rules), and §5 (verification rules). §6 then defines
each op by what is *only* true of it.

**All concrete evidence lives in §7 and later.** §0–§6 are stated without
reference to any artifact and without any measured file. §7 introduces the
artifact `torch-spyre` emits for an LX→LX movement, gives the rule that
derives every attribute of §2 and §3 from it, and then works each of the
seven patterns against a measured example — or says plainly that there is
none. A reader who wants the evidence before the specification can read §7
first; a reader who wants only the specification never needs it.

**Rule numbering.** The verification rules are numbered R1–R14 and
collected in §5, which is their single point of definition. They are cited
as `(Rn)` at the place the attribute they constrain is introduced, so a
citation like `(R1)` in §2.1 means "§5 states this rule; here is the
attribute it applies to."

Sections are normative except §0 (orientation), §7 (measured artifact and
patterns), §8 (implementation status) and §9 (open questions). §7 is the
evidence for *which* ops a lowering must actually emit: its preamble carries
the census that reads the requirement off a measured set of 51 relayouts,
and every census or measurement claim inside §7 is marked descriptive. The
attributes §7.2's rules produce are normative outputs; the procedure that
derives them is not.

**Running example.** One measured relayout, `Stcdp_QC_5`, is worked end to
end in §7.3.1 and used throughout §7.2 to illustrate the derivation rule.
It is an **all-to-all**: eight independent 4-way exchanges among 32 tiles.
It appears nowhere before §7.

---

## 0. Orientation — non-normative

This section assumes no background. It builds up, in order: what a tile is
and why anything has to move between tiles (§0.1), and what the seven ops
are for (§0.2). Neither subsection needs an artifact; §7 supplies the
concrete one.

### 0.1 Tiles, and why data crosses them

A KTIR kernel is **SPMD**: one function body, executed by every compute
tile. A **tile** is one compute core together with its private LX
memory — the unit of SPMD execution, and the thing
`ktdp.get_compute_tile_id` names. The measured hardware has 32 of them.

A tile can address only its own LX. So a program that divides a tensor
one way, computes, and then needs it divided a *different* way has a
problem no amount of local code solves: the bytes are on the wrong cores.
Moving them is what this document specifies.

Three facts shape everything that follows.

**What moves is a rectangular sub-box of one logical tensor.** Not a
message, not an arbitrary index list. Both sides of a movement describe
the *same* tensor, cut up differently; each piece is an offset and an
extent per axis. Movement is the difference between the two cuttings.

**Movement is many independent small exchanges, not one global one.**
When 32 cores redistribute data, it is almost never a 32-way exchange —
far more often it is a handful of independent exchanges among small
subsets of the cores, sharing no bytes. A **group** is one such
independent exchange, identified by an integer index `g`. Groups are the
reason a single op can describe all of them at once.

**Two roles per group, and they are separate sets.** Within group `g`,
some tiles hold data before the movement — the **producers**, written
`P(g)` — and some hold data after it — the **consumers**, `C(g)`. These
are independent sets: measured, they are sometimes equal, sometimes
overlapping, and sometimes wholly disjoint (§3.2). Conflating them is the
most common error in reading this document.

These three facts are asserted here and *demonstrated* in §7, which works
them out against a concrete artifact: §7.2 gives the rule that says how
many groups there are and who is in each, and §7.3–§7.10 apply it to every
pattern. Nothing before §7 depends on that treatment; nothing in it
contradicts this summary.

### 0.2 Why seven ops

One op **produces** — it names, per tile, what that tile contributes.
Six **deliver** — they say how the contributions land on the consumers.
Splitting the two is what keeps each op single-purpose: production is
identical in all six patterns, and only delivery differs.

The six deliveries are not an arbitrary list. They are the populated
cells of a three-property grid — whether contributions are *combined*,
how they are *placed*, and how many tiles play each role — set out in
§1.1. Read that matrix first; §3, §4 and §5 then state the machinery all
six share, and §6 reduces each op to its own row.

A reader who wants the shortest path: §1.1 (the matrix), §4 (result
types), §7.2 (how the attributes are derived, including which op a division
requires), §7.3 (all-to-all worked end to end), and §7's preamble census
(which ops the measured set actually requires).

---

## 1. Motivation and the three-property decomposition

Inter-tile communication involves three separate concerns:

1. **Production** — which tiles contribute data and what they contribute.
2. **Delivery** — how the contributed data is mapped onto the receiving
   tiles' results.
3. **Synchronization granularity** — whether each consumer tile waits
   for *all* producer tiles in its group to complete (full-barrier mode),
   or only for the specific producers whose data it requires (per-tile
   mode). Per-tile mode allows a consumer to begin as soon as its
   individual dependencies are satisfied, reducing stall time when
   producers finish at different times.

Separating production from delivery keeps each op single-purpose and
enables any combination: one production op plus a choice of delivery op.
The pairing is **one-to-one** — a production op is consumed by exactly one
delivery op (R2, §2.3). A pattern needing two deliveries therefore needs
two `ktdp.inter_tile_produce` ops. Allowing several deliveries per future
would let them share one production, but it makes R4 (coverage) and R5
(disjointness) non-local — they would have to union the dependency sets across
every use of the SSA value — so the restriction stands until a use case needs
it.

### 1.1 Semantics matrix

The six delivery ops differ in exactly three independent properties.
"Property" rather than "axis" throughout: in this document *axis* always
means a tensor or tile axis.

- **combine** — `none` | `fold` (combiner region + identity operand).
- **placement** — how producer contributions map onto consumer results:
  `replicate` | `concat` | `permute` | `split`.
- **cardinality** — producer tiles per group × consumer tiles per group.

**Semantics matrix.** One row per delivery op.

| Op | combine | placement | producers/grp | consumers/grp | dim attrs | region | identity |
|---|---|---|---|---|---|---|---|
| `consume` | none | replicate | 1 per consumer ¹ | free | — | — | — |
| `reduce` | fold | replicate | all | free | — | combiner | yes |
| `reduce_scatter` | fold | split | all | free | `scatter_dimensions` | combiner | yes |
| `gather` | none | concat | all | free | `gather_dimensions` | — | — |
| `all_to_all` | none | permute | all ² | all ² | `split_dimensions`, `concat_dimensions` | — | — |
| `scatter` | none | split | 1 per group | free | `scatter_dimensions` | — | — |

**¹ `consume` has two regimes, and the matrix row covers both.** With one
producer per group it is a **broadcast**: one value, delivered unchanged
to every consumer. With `N` producers per group and a dependency
attribute pairing each consumer with exactly one of them (R8), it is
**routing** — `N` independent point-to-point deliveries sharing one
`produce`. The bijective case of routing is a whole-partial
**permutation**, XLA's `CollectivePermute` (§6.1, §7.7.2). Both regimes
are `placement = replicate` because each consumer's result is one
producer's value unchanged; "replicate" describes the *type* relation, not
that every consumer gets the same value.

**² `all_to_all`'s `all` cells mean matching cardinalities, not identical
tile sets.** `all` reads naturally as "the group's tiles, producing and
consuming", and it is sometimes literally true — but not in general, and
not by construction. Two `all_to_all` movements can have the same group
count and the same cardinalities and still have producer and consumer sets
that are **wholly disjoint**; §7.2's `P(g)`/`C(g)` independence part shows
measured instances of both. What the cells require is that every producer in the group
contribute and every consumer in the group receive, so `|P(g)| = K` and
`|C(g)| = M` are each uniform across groups (R6, R7) — not that the two
sets coincide.

`all_to_all` is listed before `scatter` because it shares the
all-producers cardinality cell with `gather` and `reduce_scatter`, and
because its relationship to the two copy-only placements is structural:
**permute = split + concat in one step**, which is why it carries both dim
attributes and no new ones. `all_to_all` names them `split_dimensions` and
`concat_dimensions` — the same two roles `scatter_dimensions` and
`gather_dimensions` play on the single-role ops, renamed because on this
op both are present at once and *scatter* / *gather* would then name
neither the op nor a unique role.

**Every dim attribute is a list of axis indices** into `T_p` — an
`i64` array, not a single `i64` — flattened in list order per §4. The
list-valued form is not reserved for a corner case: a measured pattern
concatenates across three axes at once (§7.4).

Three things this matrix makes visible:

- **`placement` takes only four values.** The per-op type rules are four
  formulas (§4), not six.
- **The empty cells are principled.** `none` × `replicate` with all
  producers is undefined (which producer's value wins?), and `fold` ×
  `concat` / `fold` × `permute` is meaningless (fold what, then shuffle
  what?).
- **`all_to_all` is the fourth placement value, not a special case.**

### 1.2 Pattern coverage

The "all-" prefixed patterns are not separate ops: an op whose
`consumers/grp` cell is `free` already subsumes its all-tiles case by
widening `consumer_tiles_per_group`. The consumer set is therefore a
column here, since it is what distinguishes gather from all-gather and
all-to-all from scatter.

**Coverage table.** One row per named collective pattern.

| Pattern | Producers/grp | Consumers/grp | Delivery op | Result per consumer |
|---------|---------------|---------------|-------------|---------------------|
| Broadcast | 1 | free | `inter_tile_consume` | full copy |
| Reduce-to-one | all | 1 | `inter_tile_reduce` | fully reduced |
| All-reduce | all | all | `inter_tile_reduce` | fully reduced |
| Reduce-scatter | all | free | `inter_tile_reduce_scatter` | 1/C slice of reduced |
| Gather | all | 1 | `inter_tile_gather` | full assembled tensor |
| All-gather | all | all | `inter_tile_gather` | full assembled tensor |
| All-to-all | all | all | `inter_tile_all_to_all` | one slice from every producer |
| Scatter | 1 | free | `inter_tile_scatter` | 1/C slice of full |

`inter_tile_scatter` and `inter_tile_consume` have no natural "all-"
variant: R8 (§5) gives each consumer tile exactly one source, so there is no
all-producers case to widen to. `consume` still admits a group holding several
producers, but only as a **routing** pattern in which the dependency attribute
pairs each consumer tile with one of them (§6.1) — several point-to-point
deliveries sharing one `produce`, not an all-producers delivery.

**This table is division-agnostic.** It says which op expresses which
named pattern and nothing about how a particular data division maps onto a
row. That mapping is a separate question with its own procedure — §7.2's
Step 7 decision table for the classification, §7.2's other steps for the resulting
attributes — and §7.3–§7.10 walk a measured instance of each row that has
one.

Every row above is a standard collective under a standard name, which is
the fastest way for a reader with distributed-computing background to
place the six ops:

| this document | XLA / HLO | MPI |
|---|---|---|
| `inter_tile_consume` | `CollectiveBroadcast` | `MPI_Bcast` |
| `inter_tile_reduce` | `AllReduce` | `MPI_Allreduce` |
| `inter_tile_reduce_scatter` | `ReduceScatter` | `MPI_Reduce_scatter` |
| `inter_tile_gather` | `AllGather` | `MPI_Allgather` |
| `inter_tile_all_to_all` | `AllToAll` | `MPI_Alltoall` |
| `inter_tile_scatter` | — | `MPI_Scatter` |
| `consume` + bijective dep set | `CollectivePermute` | `MPI_Sendrecv` |

`scatter` having no XLA counterpart is expected: XLA's collectives are
all-to-all-shaped, and a one-to-many split is a `DynamicSlice` per
consumer there. The last row is the one pattern this document expresses
without giving it an op — see §6.1 and §7.7.2, which name it.

### 1.3 The future value

`ktdp.inter_tile_produce` returns a
`!ktdp.tile_future<(T_p), groups = #groups>` SSA value. The group set
`#groups` is carried as a parameter of the future type rather than repeated
as a separate `groups` attribute on both the production and delivery ops.
Each delivery op therefore infers the groups from its operand type, and a
group mismatch between production and
delivery is inexpressible — the def-use edge already requires the operand
type to equal the result type, so the type system rejects it structurally
rather than a verifier catching it after the fact.

The def-use edge from production to delivery encodes the happens-before
ordering with no explicit barriers in the IR. The synchronization
granularity — full-barrier or per-tile — is controlled by the
`producer_dependency_per_consumer` attribute on the delivery op (§3.4).
Corresponding production and delivery ops are expected to be adjacent in
a single basic block to avoid deadlocks.

---

## 2. `ktdp.inter_tile_produce` — unified production op

### 2.1 Attributes

**`producer_tiles_per_group`** — parameterized affine integer set `(i)[g]`
selecting which tiles produce per group. The set has one dimension (the tile
id) and one symbol (`g`, the group index). For example,
`affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>` selects tile ids
`4g .. 4g+3` for any group index `g`. The attribute is a single integer set
(`KTDP.td:143`) — there is no alternative surface form, and none is needed:
`IntegerSet` constraints are `AffineExpr`s, so `mod` and `floordiv` by a
constant are available for irregular or strided membership (§7.2, Step 4).
Which cardinality each delivery op requires of this set is
given by the `producers/grp` column of §1.1 and enforced by R8 (§5).

**Disjointness invariant (R1).** For any two distinct group indices
`g_1 != g_2` in `groups`, `producer_tiles_per_group(g_1)` and
`producer_tiles_per_group(g_2)` must be disjoint. Every producing tile is
in exactly one group. The motivation is unambiguous group membership: each
tile contributes to exactly one group's production.

**`groups`** — affine integer set defining the range of valid group indices.
For example, `affine_set<(g) : (g >= 0, -g + 7 >= 0)>` defines 8 groups,
indexed `0..7`. It bounds the range of the `g` symbol used by
`producer_tiles_per_group`. This set is **not** a standalone attribute: it
is carried as the trailing parameter of the result
`!ktdp.tile_future<(...), groups = #groups>` type, and every delivery op
infers it from its operand type (§1.3).

### 2.2 Producer region

The producer region indicates **what partial results each tile contributes**.
It is the per-tile boundary that names the SSA values entering the cross-tile
communication. The block runs once per participating tile, in that tile's
SPMD execution.

**Block argument:** `%gid: index` — the index of the group this tile
belongs to. The runtime binding is direct: tile `t` finds its group by
looking up which entry of `producer_tiles_per_group(g)` contains `t`;
that `g` is bound to `%gid` for tile `t`'s execution of the block.

The body knows its tile id via `ktdp.get_compute_tile_id` (the same way
every SPMD KTIR body does) and its group index via `%gid`.

**Termination:** the block terminates with
`ktdp.yield_partial %val_1, ..., %val_N : T_p_1, ..., T_p_N`, yielding
one value per partial-tensor role. The yielded values may reference SSA
values from the enclosing scope — typical use is a thin contribution
marker:

```mlir
{
  ^bb0(%gid: index):
    ktdp.yield_partial %my_partial_1, ..., %my_partial_N
                       : T_p_1, ..., T_p_N
}
```

with the per-tile compute that produced `%my_partial` living at function
scope (where it is naturally executed by every tile under SPMD).

**Restriction: memory ops only.** Beyond index arithmetic on `%gid` and
the tile id, the region may contain only `ktdp.construct_access_tile` and
`ktdp.load`. No compute op — no `linalg`, no `tensor` reshape, no nested
region — appears inside it.

*What this rules out.* Contribution *preparation* inside the region: a
partial that is a sum, a reduction, a fill or a reshape must be computed
at function scope and referenced from the region, as the marker form
above does. §7.5.1's `scatter` example, which sums two loaded tiles inside
the region, is the shape this restriction forbids; under it the
`linalg.add` moves out and only the two `construct_access_tile`/`load`
pairs stay in.

*Why.* Three reasons, in order of weight.

1. **The region's only irreducible job is scoping loads.** Under SPMD,
   function-scope code runs on every tile. When a group has fewer
   producers than tiles — `consume` and `scatter`, one producer per group
   (R8) — the loads that feed the partial *must not* run on the
   non-producing tiles, and the region is the only construct that can
   confine them. Compute has no such need: it is pure (§3.5's purity
   argument applies a fortiori to a partial), so executing it redundantly
   on a non-producer is wasteful but never wrong.
2. **It keeps the production op analyzable without a compute walk.** A
   lowering must read the region to learn *which bytes* each producer
   contributes, and with only access tiles and loads that is an affine
   read of the anchors and the access-tile shape — which is exactly the
   offset-and-extent pair a movement is described by (§7.1). Admit
   arbitrary compute and the same question needs dataflow analysis through
   it.
3. **It matches the artifact, which has no compute on this path at all.**
   A relayout moves bytes and does nothing to them: the artifact of §7.1
   carries no compute op and no combiner field of any kind. So there is
   nothing for compute inside the producer region to lower *to*.

The restriction is on the *producer* region only. §3.5's combiner region
is a different region on a different op and is unaffected: it is required
to contain compute, and only required to be pure.

**Worked example — anchors as a function of `%gid`.** §7.10's "How a bounded
extent is expressed" shows the `construct_access_tile` → `ktdp.load` →
`T_p` chain with fixed anchors. The region's block argument makes those
anchors group-dependent; this is that same chain with `%gid` in it.

Take a rank-4 tensor `[y, mb, out, x] = [1, 8, 128, 512]` cut 8 ways on
`x` and 4 ways on `mb`, `out` and `y` whole, on 32 tiles — 8 groups of 4.
Producer tile `t = 4g + l` owns `mb[2l : 2l+2] × x[64g : 64g+64]`, so the
two anchors are `mb = 2l` and `x = 64g`; both strides are the axis extent
over its slice count (`8/4 = 2` and `512/8 = 64`).

`%gid` supplies `g` directly. `l` is the tile's position within its group,
recovered from the tile id: `l = t - 4*g`. (§7.3.1 works this same
division as a measured relayout, with the placement derived from the
artifact's core-strides rather than assumed contiguous.)

```mlir
#part_tile_set = affine_set<(d0, d1, d2, d3) :
    (d0 == 0,                        // y
     d1 >= 0, -d1 + 1   >= 0,        // mb: 2 wide
     d2 >= 0, -d2 + 127 >= 0,        // out: whole
     d3 >= 0, -d3 + 63  >= 0)>       // x: 64 wide
#identity_4d = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

%future = ktdp.inter_tile_produce
    producer_tiles_per_group = #group_tiles
    -> !ktdp.tile_future<(tensor<1x2x128x64xf16>), groups = #all_groups>
{
  ^bb0(%gid: index):
    // Index arithmetic: l = t - 4*g, then mb = 2*l and x = 64*g.
    %t  = ktdp.get_compute_tile_id : index
    %4g = arith.muli %gid, %c4 : index
    %l  = arith.subi %t, %4g   : index
    %mb_anchor = arith.muli %l,   %c2  : index      // 0, 2, 4, 6
    %x_anchor  = arith.muli %gid, %c64 : index      // 0, 64, ..., 448

    // Memory ops only, per the restriction above.
    %access = ktdp.construct_access_tile
        %T_view[%c0, %mb_anchor, %c0, %x_anchor] {
        access_tile_set = #part_tile_set, access_tile_order = #identity_4d
    } : memref<1x8x128x512xf16> -> !ktdp.access_tile<1x2x128x64xindex>
    %partial = ktdp.load %access
        : !ktdp.access_tile<1x2x128x64xindex> -> tensor<1x2x128x64xf16>

    ktdp.yield_partial %partial : tensor<1x2x128x64xf16>
}
```

Substituting `g = 0, l = 1` gives anchor `[0, 2, 0, 0]`; `g = 2, l = 2`
gives `[0, 4, 0, 128]`. Both are boxes a measured relayout contains
(§7.3.1).

Two things the example makes visible. The **`%gid` dependence is what a
group-indexed anchor looks like**: without it every group would read the
same `x` block and the eight exchanges would not be independent. And the
region needs `%gid` *and* the tile id, not either alone — `%gid` cannot
name `l`, and the tile id alone cannot be split into `(g, l)` inside an
affine set (§3.4).

A single-producer-per-group op is the case where the region is not merely
convenient but required: with `|P(g)| == 1` the load runs on one tile in
four, and only the region confines it (§7.5.1).

### 2.3 Op signature

```mlir
%future = ktdp.inter_tile_produce
    producer_tiles_per_group = <affine-set>
    -> !ktdp.tile_future<(T_p_1, ..., T_p_N), groups = #groups>
{
  ^bb0(%gid: index):
    ktdp.yield_partial %val_1, ..., %val_N : T_p_1, ..., T_p_N
}
```

`%future` is a workgroup-visible handle carrying per-tile availability
signals. Each producer tile's contribution becomes independently
observable the moment that tile executes `ktdp.yield_partial`.

**Single-use invariant (R2).** `%future` must have exactly one use — the
single delivery op that consumes it. If two delivery ops need to
communicate with the same set of producers, they must each have their own
`ktdp.inter_tile_produce` (see §1).

---

## 3. Shared delivery semantics

Everything in this section holds for **every** delivery op. §6 states
only per-op deltas; where §6 is silent, this section governs.

### 3.1 Notation

**Symbol table.** These names are used unqualified throughout.

| Symbol | Meaning |
|---|---|
| `T_p_i` | the partial type of role `i`, as yielded by `ktdp.yield_partial` |
| `N` | number of partial-tensor roles (variadic arity), `N >= 1` |
| `P` | number of producer tiles a given consumer assembles from / waits on |
| `C` | number of consumer tiles per group, `\|consumer_tiles_per_group(g)\|` |
| `l` | within-group local index (§3.3) |

`P` is `|producer_tiles_per_group(g)|` when
`producer_dependency_per_consumer` is absent, and the (common, by R6)
cardinality of the per-consumer producer set when it is present.

### 3.2 Operand and consumer set

**Operand:** `!ktdp.tile_future<(T_p_1, ..., T_p_N), groups = #groups>` —
the future returned by the corresponding `ktdp.inter_tile_produce`. The
def-use edge is the ordering constraint, and the `#groups` parameter of
this type supplies the group set. There is no separate `groups` attribute; a group
mismatch with production is inexpressible (§1.3).

**`consumer_tiles_per_group`** — affine integer set, of the same
`(i)[g]` form as `producer_tiles_per_group`, selecting the tiles that
receive a result per group. Its permitted cardinality per op is the
`consumers/grp` column of §1.1.

The operations that use a delivery op's result are performed only by the
tiles in `consumer_tiles_per_group`. This ownership constraint is carried
by the def-use chain from the result: any use of the result is reachable
only by consumer tiles. No block is needed on any delivery op for
post-delivery computation — that is ordinary function-scope SPMD code
consuming the SSA value.

**`P(g)` and `C(g)` are unrelated sets — normative.** Write
`P(g) = producer_tiles_per_group(g)` and
`C(g) = consumer_tiles_per_group(g)`. Nothing in this design relates them,
and no relation should be assumed. A group is an **index**, not a set of
tiles: one `g` carries two independent affine sets, so "producer set ≠
consumer set" is already expressible with a single group and needs no
second group parameter.

**All four relations occur in measurement** — `P == C`, partial overlap,
strict containment either way, and wholly disjoint — and they occur among
movements that classify to the *same* op with the *same* cardinalities.
§7.2's `P(g)`/`C(g)` independence part tabulates the measured instances and
explains why: membership is a radix map whose *core-strides* are an input
independent of the division, so nothing about the division alone constrains
the relation. This is the measurement that closes R13 for the copy-only ops
(§5).

**What *is* forced, and is a different property: confinement — normative.**
Every delivery runs from a tile in `P(g)` to a tile in `C(g)` for the
**same** `g`. There are no cross-group edges. Confinement is what makes a
group an independent exchange, and it is what lets one op describe several
independent small exchanges instead of one global one. It follows from R1
(each producing tile is in exactly one group) together with
`producer_dependency_per_consumer` being parameterized by `g`.

**Confinement does not imply intersection.** A group whose two sets are
disjoint is still confined — those producers send only to those consumers.
Conflating the two properties is the failure mode this subsection exists to
prevent, and §7.2's `P(g)`/`C(g)` independence part has the measured case.
Confinement holds in
every group of the measured set.

### 3.3 Within-group local index — normative

**`l` is a tile's position, counting from 0 in ascending tile-id order,
among the relevant set within its group** — the producer set for `concat` placement (and for
`permute`'s `concat_dimensions`), the consumer set for `split` placement
(and for `permute`'s `split_dimensions`).

This definition is what makes ordered placement well-defined. Without it,
concatenation and split orders are pinned down only by contiguous-tile-id
coincidence and break silently under non-monotone tile assignments. Every
"ascending local-index order" in this document means exactly this
position — never a tile id, and never an offset in the textual order of the
set's constraints. ("Position" rather than "rank": in this document *rank*
always means a tensor's number of dimensions.)

### 3.4 `producer_dependency_per_consumer` *(optional)*

Affine integer set `(p)[c, g]` over producer tile IDs `p`, parameterized
by consumer tile `c` and group index `g`. For consumer tile `c` in group
`g`, only the producer tiles satisfying this set are waited on and
received. If absent, the consumer waits on and receives from **all**
producer tiles in the group (full-barrier semantics).

The attribute has two distinct effects, depending on placement:

- For `replicate` placement it selects *which* producer a consumer reads
  and *when* it unblocks — a synchronization refinement only.
- For `concat` and `permute` placements it additionally narrows the set
  of contributions assembled, yielding a partial (segmented) gather over
  the declared subset; `P` and hence the result type follow from it.
- For `fold` placement it makes the result a partial reduction over the
  declared subset: contributions from the remaining producers are treated
  as the identity.

`scatter` is the one op that does not accept the attribute (§6.6).

Its verification obligations are R3–R7 (§5); they are stated there and
not restated here.

Not every symbol needs to appear in a given instantiation:

- **`g` may be omitted** when the mapping is the same relative rule for
  every group (group-independent mapping). Example: a fixed per-tile
  pairing, `(p)[c] : (p - c + 2 == 0)`. Note: `g` must appear explicitly
  whenever the constraint involves a group-relative address such as
  `4*g`. Groups are disjoint, so recovering `g` from `c` would take a
  `floordiv`. MLIR's `IntegerSet` does admit `floordiv` and `mod` by
  constants and this dialect accepts them (§7.2, Step 4), but
  naming `g` is the clearer spelling and every example in this document is
  purely linear.
- **Both `c` and `g` are needed** when the mapping varies by both
  consumer identity and group. Example: a butterfly mirror exchange,
  `(p)[c, g] : (p + c - 8*g - 3 == 0)`, where the sum `p + c` differs
  for each group.

### 3.5 Combiner region and `identity` — `fold` placement only

The two `fold` ops (`reduce`, `reduce_scatter`) carry a combiner region
and an `identity` operand list. The four copy-only ops carry neither:
they place contributions by position, so there is nothing to fold and no
identity element to supply.

**Region.** A single block receiving `2N` arguments —
`%lhs_1, ..., %lhs_N, %rhs_1, ..., %rhs_N` with each `%lhs_i` and
`%rhs_i` of type `T_p_i` — terminated by
`ktdp.yield_reduced %val_1, ..., %val_N : T_p_1, ..., T_p_N`.

**Purity (R10).** The combiner must be pure — no memory effects, no calls
to side-effecting ops. Pure tensor ops (`tensor.empty`, `linalg` on
tensors, `arith.*`) are allowed.

**Combine ordering.** The associative-commutative contract is by user
agreement; the scheduler is free to combine in tree, ring, linear, or any
hardware-native topology. Different groups' reductions are independent
and may be scheduled in parallel. This freedom is what distinguishes
`fold` from the copy-only placements, whose ordered placement by `l`
(§3.3) is deterministic and requires no commutativity.

**`identity` (R11).** `N` variadic SSA operands, one per role. Each
identity tensor's shape and element type must match the corresponding
partial type `T_p_i` — *not* the result type. The identities are hoisted
before the op and shared across all groups and all tiles. Combining any
identity with its corresponding partial yields that partial.

### 3.6 Synchronization model

No explicit barriers appear in the IR. The
`!ktdp.tile_future<(T_p), groups = #groups>` SSA value carries **per-tile
availability signals** rather than a monolithic group barrier:

1. Each producer tile's contribution becomes independently observable as
   soon as that tile executes `ktdp.yield_partial` in the production
   block.
2. A delivery op cannot use a producer tile's contribution until that
   tile's signal is set in `%future`.
3. The producer tiles a given consumer tile waits for are declared by
   `producer_dependency_per_consumer` (§3.4):

   - **Absent (default) — full-barrier mode:** consumer tile `c` in group
     `g` waits for every producer tile in `producer_tiles_per_group(g)`
     before the delivery op executes. This maps directly to a hardware
     group barrier and preserves the simplest safety guarantee.
   - **Present — per-tile mode:** consumer tile `c` waits only for the
     producer tiles `p` satisfying
     `producer_dependency_per_consumer(p)[c, g]`. The consumer unblocks
     as soon as those specific tiles have completed, without waiting for
     unrelated producers. Different consumer tiles may declare different
     dependency sets, enabling fine-grained producer–consumer pipelining.

A multi-producer wait is therefore a **per-consumer AND-join over
existing per-tile signals**, not a new primitive. This is why the
all-producers ops (`reduce`, `reduce_scatter`, `gather`, `all_to_all`)
introduce no synchronization machinery beyond what a single-producer op
already needs: they differ only in how many signals the join covers.

In SPMD KTIR, a tile cannot observe other tiles' partials except through
a dialect-defined boundary. The `ktdp.inter_tile_produce` block is that
boundary — it names the per-tile contribution and exposes it via
`%future`. The delivery op's result tensor is an SSA value that cannot
materialize until the declared dependencies are satisfied; standard MLIR
dataflow ordering applies.

Lowering inserts target-specific hardware synchronization: a group
barrier for full-barrier mode, and point-to-point ready/wait signals for
per-tile mode.

### 3.7 Result semantics

Every delivery op produces `N` variadic SSA values, one per
partial-tensor role. The values are **per-tile-valued**: each consumer
tile holds its own result value when the op completes. Whether tiles in
the same group hold the *same* value is a property of the placement.

**Sharing table.** One row per placement value.

| placement | tiles in one group hold | tiles in different groups hold |
|---|---|---|
| `replicate` | the same value | their own group's value |
| `concat` | the same assembled tensor | their own group's assembly |
| `split` | disjoint ordered slices that tile the whole | slices of their own group's tensor |
| `permute` | different assemblies (one slice per producer) | their own group's exchange |

**Non-participating tiles.** Results are undefined for tiles not in
`consumer_tiles_per_group`.

**Multi-tensor (variadic) delivery.** `N >= 1` roles are supported by
every op, and all roles share the same attributes (`scatter_dimensions`,
`gather_dimensions`, `P`, `C`) — only the types differ. Argmax-style reductions,
where each contribution is a correlated tuple of tensors (values,
indices), use `N = 2`: two identities, two yielded partials, four
combiner arguments yielding two combined values, two op results. Each
role's result type follows the §4 rule independently.

---

## 4. Placement algebra and type rules

Result types are a function of the placement value alone. There are four
formulas, applied per role `i` to `T_p_i`.

**Type-rule table.** One row per placement value.

| placement | result type derived from `T_p` |
|---|---|
| `replicate` | `T_p` unchanged — no rank reduction |
| `concat` | extents along `gather_dimensions` multiplied by `P` in total |
| `split` | extents along `scatter_dimensions` divided by `C` in total |
| `permute` | extents along `scatter_dimensions` divided by `C` in total, **and** extents along `gather_dimensions` multiplied by `P` in total |

`reduce_scatter` is `fold` + `split`: the `split` formula applies directly
to `T_p`, with no collapse first.

**What `P` and `C` are, as numbers — normative.** The two factors are
counts of *contributions* and of *destination shares*, not of cores.

- **`P` is the number of producer contributions a consumer assembles.** It
  is `|producer_tiles_per_group(g)|` when
  `producer_dependency_per_consumer` is absent, and the (uniform, by R6)
  size of the per-consumer producer set when it is present (§3.1).
- **`C` is the number of distinct shares the group's result is cut into.**
  Where the group's result is *replicated* across several consumer tiles,
  `C` is the number of distinct shares, **not** the number of tiles that
  hold one.

The two readings of `C` coincide whenever each share has a single holder,
and they diverge whenever a share is multicast. This is not hypothetical:
in the measured set, 28 of the 51 relayouts have a destination share held
by 4, 28 or 32 cores at once (§7.2, Step 1). Every one of them classifies
as `gather`, whose `concat` formula does not use `C` at all — so the
divergence is measured but never yet load-bearing. It would become
load-bearing the moment a `split` or `permute` movement multicast a share,
and a verifier must therefore take `C` from the result type's share
structure rather than from `|consumer_tiles_per_group(g)|`.

On the producer side the two readings coincide in every measured relayout:
every source region has exactly one holder (§7.2), and §7.2's Step 7 returns
*insufficient information* rather than guessing when one does not (§9.2).

**No rank reduction anywhere.** All four formulas keep `T_p`'s rank. The
same reasoning that settled it for `reduce` (§6.2) applies to `concat` and
`split`: the axis the op concatenates along or splits is an axis `T_p`
already has, so there is nothing to collapse and no rank to restore. A
`concat` result differs from `T_p` only in the extent along the listed axes,
a `split` result likewise — never in rank. This keeps every result type a
per-axis extent rewrite of the partial, which is what lets §9.3 reason
about layout transparency one axis at a time.

**Axis sets and flattening — normative.** Each dim attribute is a *list*
of axis indices into `T_p`, not a single axis. A list of length `n > 1`
denotes the product space of those axes, linearized as a row-major
(mixed-radix odometer) order over the listed extents: **the first entry is
the slowest-varying and the last is the fastest-varying**. Write
`E(D) = prod(T_p[d] for d in D)` for the flattened extent of axis set `D`.
The single-axis case is `n == 1`, where `E(D) = T_p[d]` and every formula
below reduces to its familiar form; `n == 0` is invalid for an op that
carries the attribute.

**The list is in ascending numerical order (R9).** Entries must ascend, so
the slowest-to-fastest flattening above coincides with ascending axis index
and the attribute has exactly one legal spelling per axis set. Two reasons
this is a rule and not a convention. It removes a silent-miscompile class:
`[2, 0]` and `[0, 2]` are both "valid, distinct, non-empty" and would flatten
to *different* data orders, so a reversed list passes every other check while
meaning something else. And it makes attribute equality a list comparison —
which §4's conservation case below depends on, since `all_to_all` decides
whether `T_c == T_p` by testing `split_dimensions == concat_dimensions`.

Entries need not be **adjacent**: `[0, 2]` over a rank-3 partial is legal and
is exactly what physicalization produces (§9.3).

**Split and concat apply to the floordiv axis — normative.** When a listed
axis is a **sticked** axis — one that a stick layout has split into a
`floordiv` (chunk-count) axis and a `mod` (within-stick) axis — the `÷ C` or
`× P` applies to the **floordiv axis only**. The `mod` axis is invariant: its
extent is the stick size, and changing it would redefine what a stick is.

This settles what "`E(D)` divided by `C`" alone leaves open, since a flattened
extent does not say which listed axis absorbs the factor. For a partial
`[2, 16, 32]` (logical `[16, 64]`, stick 32) with `gather_dimensions = [0, 2]`
and `P = 4`, the result is `[8, 16, 32]` — the chunk count goes `2 → 8` and
the stick axis stays `32`, which is exactly the physicalization of the logical
result `[16, 256]`. Absorbing into the `mod` axis instead would give
`[2, 16, 128]`: the same flattened extent, the wrong tensor.

A useful consequence: **R9 applied to the floordiv axis is the stick-multiple
check.** `E(floordiv) % C == 0` holds exactly when the logical result extent
is a whole multiple of the stick, so a split that would drive the result
sub-stick fails R9 rather than needing a rule of its own. On the partial
above, `C = 2` gives `2 % 2 == 0` and a result of `[1, 16, 32]`; `C = 4` gives
`2 % 4 ≠ 0` and is rejected — correctly, since the logical result `[16, 16]`
is half a stick and unrepresentable in that layout.

Fixing this order is a requirement, not a convenience: §7.4 has a measured
three-axis concat, so the flattening must be well-defined over more than
two axes for a *named* pattern rather than only a corner case.

**Which slice a tile gets — normative, and stated over slice indices, not
over elements.** This is the one place where the single-axis intuition
misleads, so the multi-axis case is given first.

Let `D = [d_0, …, d_{n-1}]` be the axis list, ascending (R9). For `concat`,
the assembly is a **mixed-radix odometer over per-axis slice indices**: the
producer local index `l` (§3.3) decomposes as

```
l  =  l_0 · (n_1 · n_2 · … · n_{n-1})  +  …  +  l_{n-2} · n_{n-1}  +  l_{n-1}
```

where `n_k` is the number of producer shares along axis `d_k`, `d_0` is
slowest-varying and `d_{n-1}` fastest, and producer `l` occupies the **box**

```
[ l_k · T_p[d_k] : (l_k + 1) · T_p[d_k] )      on each listed axis d_k
```

with every unlisted axis whole. `split` is the same statement with the
consumer local index and per-axis share counts, and `permute` applies both
simultaneously: consumer `l_c` receives, from each producer `l_p`, that
producer's `split_dimensions` box `l_c`, placed at `concat_dimensions` box
`l_p`.

**A producer's contribution is a box, not an interval.** For `n == 1` the
box *is* the interval `[l·chunk : (l+1)·chunk)` of the flattened space, and
the two statements agree — which is why the single-axis form is the one
usually quoted. For `n > 1` they do **not** agree: a rectangular sub-box of
three axes is not a contiguous run of the row-major element flattening of
those three axes. Stating placement as an interval of `E(D)` would be
wrong for the measured three-axis gather. The odometer form above is
correct for all `n` and reduces to the interval form at `n == 1`.

**Consequence: for `n > 1` the result type is checked, not derived.** The
per-axis share counts `n_k` are extra information that `P` alone does not
carry — `P = 32` is consistent with `(n_k) = (2,8,2)`, `(32,1,1)`, `(4,4,2)`
and more. The result type is written explicitly in the IR, so a verifier
does not need to derive it; what it must check is

```
for each listed axis d_k :   T_r[d_k] % T_p[d_k] == 0            (concat)
                             prod_k  T_r[d_k] / T_p[d_k]  ==  P
```

and the mirror form for `split`, with the division and multiplication
exchanged. That is exactly R12 and R9 read per axis, and it is why R12's
per-axis clause is not redundant with its product clause (§5). The measured
three-axis gather satisfies it: `T_p` shares are `in:64, out:64, x:4` of a
tensor `in:128, out:512, x:8`, so the per-axis factors are `(2, 8, 2)`,
their product is `32 = P`, and the flattened extent multiplies out as
`16384 × 32 = 524288` (§7.4).

For `n == 1` the result type *is* derivable, and the table above is the
derivation.

**Conservation in the square case.** Whenever `P == C`, the `permute`
result has the same element count as `T_p` — one axis is divided and
another multiplied by the same factor — so a square all-to-all is a pure
redistribution of ownership. If additionally `split_dimensions == concat_dimensions`,
the result *type* equals `T_p`: the distributed transpose. That equal-type
case is not what the measured set contains — every measured all-to-all
splits and concats *different* axes (§7.3, §7's census), so `P == C` conserves the
element count while the type still changes. The non-square measured case
(`P = 2`, `C = 4`) does not conserve it at all: `131072 → 65536` elements
per tile, halved because twice as many consumers share the same total
(§7.3.2).

**Why `split` divides an honest data axis.** Every splitting op divides an
extent of an axis the partial already has, so the types stay honest:
`<128x1x64>` → `<32x1x64>`, never `<1x...>`. No op manufactures a unit
dimension for a collapse to consume, and none removes one.

---

## 5. Verification rules

Principle: **each rule has exactly one owner and one statement;
applicability is a column, not a restatement.** "Owner" is the op that
carries the attribute the rule constrains.

**Where the statement lives.** Three rules are stated at the attribute they
constrain rather than below: R1 in §2.1, R2 in §2.3 and R10 in §3.5. Their
matrix rows carry the pointer and this section does not restate them, so the
one-statement invariant holds for all fourteen — but §5 is the single point
of definition only for the other eleven.

**Verification matrix.** One row per rule, one column per delivery op.

| Rule | Owner | consume | reduce | red_scat | gather | all_to_all | scatter |
|---|---|---|---|---|---|---|---|
| R1 group disjointness (§2.1) | produce | y | y | y | y | y | y |
| R2 single-use future (§2.3) | produce | y | y | y | y | y | y |
| R3 dep set subset of producers | delivery | y | y | y | y | y | n/a |
| R4 every producer covered by some consumer | delivery | y | y | y | y | y | n/a |
| R5 dep sets pairwise disjoint | delivery | — | — | — | y | y | n/a |
| R6 uniform dep-set cardinality | delivery | — | — | — | y | y | n/a |
| R7 uniform producer cardinality across groups | delivery | — | — | — | y | y | n/a |
| R8 single-source delivery | delivery | y | — | — | — | — | y |
| R9 flattened split extent divisible by `C` | delivery | — | — | y | — | y | y |
| R10 combiner purity (§3.5) | delivery | — | y | y | — | — | — |
| R11 identity shape matches `T_p` (§3.5) | delivery | — | y | y | — | — | — |
| R12 flattened concat extent × `P` well-defined | delivery | — | — | — | y | y | — |
| R13 consumer set subset of producer set | delivery | — | y | ? | n | n | n |
| R14 reduce mode gate: `C == P` or `\|C\| == 1` | delivery | — | y | ? | — | — | — |

**R3–R7 are vacuous on every measured relayout.** All five constrain
`producer_dependency_per_consumer`, and §7.2's Step 6 establishes by
measurement that **no** relayout in the measured set declares it: the
within-group overlap graph is complete bipartite in all 51, so the default
full-barrier reading is correct throughout. R6 and R7 are separately
*confirmed* rather than assumed, since cardinalities are uniform across
groups in all 51. So these five rules are needed for the routing patterns
of §7.7 and for nothing yet measured.

Statements:

- **R3 — subset.** The declared dependency set must be a subset of
  `producer_tiles_per_group`; referencing a non-producer tile is an
  error.

  ```text
  { p | ∃ c, g : producer_dependency_per_consumer(p)[c, g] }
    ⊆
  { p | ∃ g : p ∈ producer_tiles_per_group(g) }
  ```

- **R4 — coverage.** For every group `g` and every producer `p` in
  `producer_tiles_per_group(g)`, at least one consumer `c` in
  `consumer_tiles_per_group(g)` must satisfy
  `producer_dependency_per_consumer(p)[c, g]`. An uncovered producer
  yields a value no consumer reads, risking deadlock in push-based
  lowerings.

  ```text
  ∀ g, ∀ p ∈ producer_tiles_per_group(g) :
      ∃ c ∈ consumer_tiles_per_group(g) :
          producer_dependency_per_consumer(p)[c, g]
  ```

- **R5 — pairwise disjointness.** For the assembling placements
  (`concat`, `permute`), distinct consumers' declared dependency sets
  must be disjoint. R4 alone requires only that each producer be claimed
  by *at least one* consumer, which combined with R6 admits declared sets
  that double-count producers — and a double-counted producer has no
  well-defined position in the assembly.
- **R6 — uniform dep-set cardinality.** All consumers in a group must
  declare the same number of producers, so `P` is a single number and the
  op has one static result type.
- **R7 — uniform producer cardinality across groups.**
  `producer_tiles_per_group` is a parameterized affine set over `g` and
  nothing otherwise requires equal cardinality per group. Since the op
  result is a single static tensor type, unequal groups yield no
  expressible result type for the assembling placements.
- **R8 — single-source delivery.** For the ops whose `producers/grp` cell is
  `1` — `inter_tile_consume` and `inter_tile_scatter` — every consumer tile
  must receive from exactly one producer tile:

  ```text
  ∀ g, ∀ c ∈ consumer_tiles_per_group(g) : |dep(c, g)| == 1
  ```

  where `dep(c, g)` is the producer set `producer_dependency_per_consumer`
  declares for consumer tile `c` (§3.4). `inter_tile_scatter` takes no such
  attribute (§6.6), so for it the rule reduces to its simplest form: exactly
  one producer tile per group.

  **Why per consumer tile rather than per group.** These two ops deliver into a
  result no larger than one contribution — unchanged for `replicate`, a `1/C`
  slice for `split` — with no combiner and nowhere to put a second value. A
  consumer tile holding two contributions is therefore the undefined cell of
  §1.1, "which producer's value wins?". What the op needs is not that the
  *group* hold one producer, but that each *consumer tile* have a single
  source. The two coincide when `|P(g)| == 1`, the common case (broadcast,
  §7.6), which needs no attribute at all.

  **For `inter_tile_consume` with `|P(g)| > 1` the attribute is required.**
  There is no meaningful default, because receiving from every producer is
  exactly that undefined cell. With the attribute, such a group is a
  **routing** pattern — several independent point-to-point deliveries sharing
  one `produce` op, as in §7.7.1 and §7.7.2 — and no consumer tile ever sees
  two values. A group with `|P(g)| > 1` and no attribute is rejected.

  A producer **may** serve several consumer tiles (multicast within the
  group); it may not serve none, which R4 already requires. So `dep` need not
  be injective — only single-valued per consumer tile, and total over
  producers.

- **R9 — split divisibility.** `E(D_split) % C == 0`, where `D_split` is
  the op's split axis set (`scatter_dimensions`, or `split_dimensions` for
  `all_to_all`) and `E` is the flattened extent of §4. Stating the rule on the
  *flattened* extent is what lets one rule cover all three splitting ops
  and every arity: a multi-axis split need only divide in the product,
  not axis by axis.

  Every axis index in the list must be a valid, distinct axis of `T_p`; the
  list must be non-empty; and the entries must be in **ascending numerical
  order** (§4). Repeated indices would double-count an extent in `E`, and an
  out-of-order list would silently denote a different flattening.

  For a multi-axis list the product form is necessary but **not
  sufficient**: §4's odometer placement also requires each listed axis to
  divide individually — `T_p[d_k] % T_r[d_k] == 0` for every listed axis,
  with the per-axis factors `T_p[d_k] / T_r[d_k]` multiplying to `C`. This is
  the same per-axis clause R12 carries on the concat side, and for the same
  reason. No measured split is multi-axis (§7.5), so this clause is
  unexercised while R12's concat counterpart is measured (§7.4).
- **R11 and the shipped constraint.** R11 pins `identity` to `T_p`, while
  the implemented `reduce` ties it to *results* (`KTDP.td:172-174`). With no
  rank reduction (§4) these coincide for `reduce`, since its result *is*
  `T_p`. They diverge for `reduce_scatter`, whose result is `T_p` split by
  `C`: R11's `T_p` is the correct one there, since the identity is combined
  with partials before the split. A verifier generalizing the shipped
  constraint to `reduce_scatter` must therefore retarget it from results to
  the future's partial types (§9.3).

- **R12 — concat well-definedness.** The result flattened extent over the
  concat axis set `D_concat` (`gather_dimensions`, or `concat_dimensions`
  for `all_to_all`) is `P × E(D_concat)`, which requires every assembled
  producer to contribute the same extent along *each* listed axis — equal
  products alone would not give a well-defined multi-axis assembly, since
  the flattening of §4 depends on the individual extents. The same
  validity conditions as R9 apply to the list. For the square
  `all_to_all` case the divisibility follows from R7 + R9, but it must be
  stated independently for the non-square case, which §7.3.2 shows is
  measured and not hypothetical.

  **The per-axis clause is what makes the rule checkable at all for
  `n > 1`.** §4 shows that `P` alone does not determine the per-axis
  factors, so the result type is declared rather than derived; R12 is then
  the check that the declared type is consistent with `T_p` and `P` —
  per-axis divisibility, and per-axis factors multiplying to `P`. The
  measured three-axis gather is the case that exercises it (§7.4).
- **R13 — consumer set subset of producer set.** Every consumer tile in a
  group must also be a producer in that group, i.e.
  `consumer_tiles_per_group(g) ⊆ producer_tiles_per_group(g)`.

  **It holds for `reduce` and fails for the copy-only ops.** For `reduce`
  it is enforced today (`KTIRCheckLegality.cpp:107–117`). For `gather`,
  `all_to_all` and `scatter` it is **falsified by measurement**, so the
  cells are `n` rather than open.

  **16 of the 51 relayouts have receive-only consumers** — the input lives
  on 16 cores while the output spreads over all 32, so the other 16 consume
  without producing. They divide across the three ops as 3 `gather`, 1
  `all_to_all` (the non-square one) and 12 `scatter`. The mirror case is
  measured too, once: one relayout has 32 producers and 28 consumers, so
  four cores send and never receive (§7.4). Both `C ⊄ P` and `C ⊊ P` occur,
  and the two sets are independent (§3.2, and §7.2's `P(g)`/`C(g)`
  independence part for why).
  `scatter` was already resolved *no* by argument (§6.6) — the other two are
  now resolved *no* by measurement. `reduce_scatter` stays open, since no
  measurement reaches it (§9.1).
- **R14 — reduce mode gate.** For `reduce`, the consumer set must either
  equal the producer set (all-reduce) or be a single tile
  (reduce-to-one); a strict multi-tile subset — reduce-to-subset — is
  rejected. This is a present implementation restriction, not a design
  conclusion (§9.1).

---

## 6. The delivery ops

Each subsection states only what is specific to that op: its cells from
§1.1, its result type from §4, its signature, and any op-specific
argument. Shared machinery is §3; rules are §5.

### 6.1 `ktdp.inter_tile_consume` — broadcast and routing

`combine = none`, `placement = replicate`, one producer per **consumer
tile** (R8 — per consumer tile, not per group), consumer set free, no dim
attribute, no region, no identity.

**Result type.** `T_p_i` unchanged (§4, `replicate` + `none`).

**Two regimes, one op.** The distinction is `|P(g)|`, and it decides
whether the dependency attribute is optional or mandatory.

| regime | `\|P(g)\|` | `producer_dependency_per_consumer` | what it is |
|---|---|---|---|
| **broadcast** | 1 | optional — pure synchronization refinement | one value delivered unchanged to every consumer in the group |
| **routing** | > 1 | **required** (R8) | `\|P(g)\|` independent point-to-point deliveries sharing one `produce` |

**Semantics.** No combining occurs, in either regime. In the broadcast
regime the group's single producer's value is delivered unchanged to every
consumer tile in the group. In the routing regime each consumer tile
receives, unchanged, the value of the one producer the attribute pairs it
with — so no consumer tile ever sees two values, which is what keeps
`placement = replicate` honest (§1.1).

**Neither regime is measured.** Every source region in the measured set has
exactly one holder, so no measured relayout has a group with `|P(g)| > 1`
to route within, and none has a replicated *source* to broadcast from
(§7.6, §7.7).

```mlir
%result_1, ..., %result_N = ktdp.inter_tile_consume(%future)
    consumer_tiles_per_group         = <affine-set>,
    producer_dependency_per_consumer = <affine-set>   // optional; default: all producers
    : !ktdp.tile_future<(T_p_1, ..., T_p_N), groups = #groups> -> T_p_1, ..., T_p_N
```

With one producer per group the attribute is a pure synchronization
refinement (§3.4): there is only one value to receive, so it changes when
each consumer unblocks and nothing else. With **several** producers per
group it also names the sender, and R8 then requires it — each consumer
tile must be paired with exactly one producer. That is the routing regime,
which is what lets `consume` express per-tile pairing (§7.7.1) and
one-to-one permutation exchange (§7.7.2). **R8's "one producer" is per
consumer tile, not per group**, and the routing regime is exactly the case
where the two differ (§5).

**The bijective case has a standard name: `CollectivePermute`.** When
`dep` is a bijection between `P(g)` and `C(g)`, each tile sends its whole
partial to exactly one other tile and receives exactly one — XLA's
`CollectivePermute`, JAX's `lax.ppermute`, MPI's `MPI_Sendrecv`. This
document expresses the pattern without giving it an op, deliberately: the
op would be `consume` with a rule attached. The name is used here and in
§7.7.2 so that a reader looking for a permutation primitive finds where it
lives. The artifact of §7.1 can express it directly — identical piece
geometry on both sides with different `memId` — so the pattern is
*expressible*, though no measured relayout is one (§7.7). It does not
overlap `all_to_all`, which splits partials rather than moving them whole
(§6.5).

Delivering to a consumer tile from more than one
producer is never legal here — with no combiner and a result the size of
one contribution, there would be nowhere to put the second value (§1.1).

### 6.2 `ktdp.inter_tile_reduce` — reduction

`combine = fold`, `placement = replicate`, all tiles produce, consumer
set free, no dim attribute, combiner region and `identity` per §3.5.

**Result type.** `T_r_i == T_p_i` — no rank reduction. An earlier draft
collapsed the within-group tile axes; the implementation deliberately does
not (`KTDP.td:197`), because the partial already carries that axis and
keeping it makes the op simpler: result, partial and `identity` are then one
type, tied declaratively (`KTDP.td:168-174`) rather than by a shape
computation. This is also what makes `reduce` transparent under
physicalization (§9.3).

```mlir
%r_1, ..., %r_N = ktdp.inter_tile_reduce(%future)
    consumer_tiles_per_group         = <affine-set>,
    producer_dependency_per_consumer = <affine-set>,   // optional; default: all producers
    identity(%id_1 : T_p_1, ..., %id_N : T_p_N)
    : !ktdp.tile_future<(T_p_1, ..., T_p_N), groups = #groups> -> T_r_1, ..., T_r_N
{
  ^bb0(%lhs_1: T_p_1, ..., %lhs_N: T_p_N,
       %rhs_1: T_p_1, ..., %rhs_N: T_p_N):
    ktdp.yield_reduced %val_1, ..., %val_N : T_p_1, ..., T_p_N
}
```

Consumer set = producer set is all-reduce; a single consumer per group is
reduce-to-one. Both are supported today; a strict multi-tile subset is
not (R14).

**No instance of this op is measured** (§7.8), so the combiner region stays
general on the strength of §3.5 alone: any pure region the user agrees is
associative, with a matching `identity`. Whether a given combiner has a
lowering path is a lowering concern, not an op-surface one.

### 6.3 `ktdp.inter_tile_reduce_scatter` — reduction then split

`combine = fold`, `placement = split`, all tiles produce, consumer set
free, `scatter_dimensions`, combiner region and `identity` per §3.5.

**`scatter_dimensions`** (`i64` array) — axes of `T_p` along which the
reduced result is split row-major across the consumer tiles (R9).

**Result type.** `T_r_i` is `T_p_i` with the flattened extent over
`scatter_dimensions` divided by `C` — no rank reduction (§4). The same axes
and the same split apply to all roles.

```mlir
%chunk_1, ..., %chunk_N = ktdp.inter_tile_reduce_scatter(%future)
    consumer_tiles_per_group         = <affine-set>,
    scatter_dimensions               = <i64-array>,
    producer_dependency_per_consumer = <affine-set>,   // optional; default: all producers
    identity(%id_1 : T_p_1, ..., %id_N : T_p_N)
    : !ktdp.tile_future<(T_p_1, ..., T_p_N), groups = #groups> -> T_r_1, ..., T_r_N
{
  ^bb0(%lhs_1: T_p_1, ..., %lhs_N: T_p_N,
       %rhs_1: T_p_1, ..., %rhs_N: T_p_N):
    ktdp.yield_reduced %val_1, ..., %val_N : T_p_1, ..., T_p_N
}
```

### 6.4 `ktdp.inter_tile_gather` — ordered assembly

`combine = none`, `placement = concat`, all tiles produce, consumer set
free, `gather_dimensions`, no region, no identity.

**Not an index-vector gather.** The name is the collective's, not
`tensor.gather`'s: this op assembles contributions by *position* (§3.3),
and no operand of it is a set of indices. The artifact agrees — nothing in
it can carry an index operand at all (§7.1), so an index-vector gather is a
different artifact entirely, not a variant of this one.

**`gather_dimensions`** (`i64` array) — axes of `T_p` along which the
producers' partials are concatenated, in ascending producer local-index
order (§3.3). A multi-axis set assembles by the per-axis odometer of §4,
listed axes ordered slowest- to fastest-varying; §7.4 supplies a measured
three-axis case.

**Result type.** `T_g_i` is `T_p_i` with the extents along
`gather_dimensions` multiplied by per-axis factors whose product is `P`
(R12). For a single-axis list that determines the type; for a multi-axis
list the type is declared and R12 checks it (§4).

```mlir
%gathered_1, ..., %gathered_N = ktdp.inter_tile_gather(%future)
    consumer_tiles_per_group         = <affine-set>,
    gather_dimensions                = <i64-array>,
    producer_dependency_per_consumer = <affine-set>   // optional; default: all producers
    : !ktdp.tile_future<(T_p_1, ..., T_p_N), groups = #groups> -> T_g_1, ..., T_g_N
```

One consumer per group is a plain gather; the full group as consumer set
is all-gather — the same op with a wider set (§1.2), not a separate op.
With `producer_dependency_per_consumer` present the assembly is a partial
(segmented) gather over each consumer's declared subset, subject to
R5–R7.

### 6.5 `ktdp.inter_tile_all_to_all` — split and reassemble

`combine = none`, `placement = permute`, all tiles produce, all tiles
consume, both `split_dimensions` and `concat_dimensions`, no region, no
identity.

**Attributes.** `split_dimensions` (`i64` array) — axes each producer
splits into `C` chunks (R9). `concat_dimensions` (`i64` array) — axes
along which each consumer concatenates the chunks it received, in
ascending producer local-index order (R12). Both are flattened in list
order per §4. The two sets may in principle be equal (a pure ownership
transpose along one axis set), but **every measured all-to-all splits and
concatenates *different* axes** (§7.3), so the equal case is unattested.

**Cardinalities.** `|P(g)| = K` and `|C(g)| = M` are each uniform across
groups (R6, R7) and **need not be equal to each other**. The `all` cells of
§1.1 assert matching cardinalities, not identical tile sets and not
`M == K`; §7.2's `P(g)`/`C(g)` independence part shows measured groups whose
producer and consumer
sets are wholly disjoint, and §7.3.2 a measured movement with `K = 2`
against `M = 4`. So neither `P == C` nor `P(g) == C(g)` may be assumed.

**Result type.** `T_c_i` is `T_p_i` with the `split_dimensions` extents
divided by per-axis factors whose product is `C`, and the
`concat_dimensions` extents multiplied by per-axis factors whose product is
`P` (§4). Only when `P == C` **and**
`split_dimensions == concat_dimensions` does `T_c_i == T_p_i`; when `P == C`
with different axis sets the element count is conserved but the type is
not; when `P ≠ C` neither is.

**Semantics.** Consumer with local index `l_c` receives, from each producer
`l_p`, that producer's `split_dimensions` **box** `l_c`, and places those
`P` contributions at `concat_dimensions` boxes `0 … P-1` in ascending `l_p`
order. "Box" rather than "slice" because for a multi-axis list the
contribution is a rectangular sub-box and not a contiguous interval of the
flattened space (§4).

```mlir
%out_1, ..., %out_N = ktdp.inter_tile_all_to_all(%future)
    consumer_tiles_per_group         = <affine-set>,
    split_dimensions                 = <i64-array>,
    concat_dimensions                = <i64-array>,
    producer_dependency_per_consumer = <affine-set>   // optional; default: all producers
    : !ktdp.tile_future<(T_p_1, ..., T_p_N), groups = #groups> -> T_c_1, ..., T_c_N
```

**Why it is a first-class op rather than a composition.** All-to-all
requires every tile to be simultaneously a producer of `C` distinct
slices and a consumer of `P` distinct slices:

```text
tile 0: A[0][0..3]     tile 0: A[0][0] A[1][0] A[2][0] A[3][0]
tile 1: A[1][0..3]     tile 1: A[0][1] A[1][1] A[2][1] A[3][1]
tile 2: A[2][0..3] --> tile 2: A[0][2] A[1][2] A[2][2] A[3][2]
tile 3: A[3][0..3]     tile 3: A[0][3] A[1][3] A[2][3] A[3][3]
```

Neither existing copy-only op admits this. `gather` delivers the *same*
assembled tensor to every consumer (§3.7) and cannot give consumers
different content. `scatter` permits exactly one producer per group (R8)
and cannot have every tile contribute. Composing them materializes the
full concatenation on every tile — wrong data volume and wrong
communication pattern.

The only faithful composition is `C` separate `produce`+`scatter` pairs
(one per source tile, forced by R2) followed by a per-consumer `concat`
in ordinary SPMD code: `C×` the ops, `C×` the produce handles, and
reassembly pushed out of the inter-tile layer. That composition is the
useful *reference lowering* — it is why `all_to_all` needs no new
synchronization (§3.6) and no new verification beyond `scatter` ∪
`gather` (R9 and R5–R7/R12 respectively) — but it is the wrong surface
form. Note also that **one-to-one permutation** of whole partials is
already expressible as `consume` + a bijective dependency set (§7.7.2);
`all_to_all` is only for the split-and-redistribute case, so the two
mechanisms do not overlap.

**Generalizing `scatter` to `P > 1` is not an alternative.** Adding a
concat axis set and lifting R8 on `scatter` *is* `all_to_all` under another
name; it hides the multi-producer wait inside `scatter` and gives that op
two regimes. A separate op keeps one-op-one-pattern and leaves
`scatter`'s `P == 1` contract clean.

**Why not `inter_tile_shuffle`.** The artifact calls the primitive
`shuffle`, but `shuffle` is what *several* of these patterns become rather
than this one alone (§7), so `all_to_all` is used as the established
collective term.

### 6.6 `ktdp.inter_tile_scatter` — ordered split

`combine = none`, `placement = split`, one producer per group (R8),
consumer set free, `scatter_dimensions`, no region, no identity.

**`scatter_dimensions`** (`i64` array) — axes of `T_p` along which the
single producer's tensor is partitioned into `C` equal shares (R9), one
per consumer in ascending consumer local-index order (§3.3), by the
per-axis odometer of §4.

**Result type.** `T_s_i` is `T_p_i` with the `scatter_dimensions` extents
divided by per-axis factors whose product is `C` (§4). All measured
scatters are single-axis, where this is just `÷ C` on the one axis (§7.5).

```mlir
%scattered_1, ..., %scattered_N = ktdp.inter_tile_scatter(%future)
    consumer_tiles_per_group = <affine-set>,
    scatter_dimensions       = <i64-array>
    : !ktdp.tile_future<(T_p_1, ..., T_p_N), groups = #groups> -> T_s_1, ..., T_s_N
```

**No `producer_dependency_per_consumer`.** With a single producer per
group there is exactly one producer to wait for, so full-barrier and
per-tile synchronization collapse to the same thing; the attribute would
be degenerate. R3–R7 are therefore `n/a` for this op (§5).

**Consumers need not be producers.** A consumer tile that does not appear
in `producer_tiles_per_group` simply receives its slice; unlike a partial
gather or a reduce there is nothing for a non-producing consumer to
contribute or miss, so no coverage obligation arises. For a pure split
the consumer set is unconstrained relative to the producer set — which
resolves §9.1 for `scatter`, and only for `scatter`.

---

## 7. The measured artifact and the seven patterns

This is where the concrete evidence lives. §1–§6 are stated without
reference to any artifact. This section introduces the artifact
`torch-spyre` emits for an LX→LX movement (§7.1); gives the complete
derivation that turns a pair of piece tables into `groups`, `P(g)`, `C(g)`,
`producer_dependency_per_consumer` **and the choice of delivery op**
(§7.2); and then works each pattern against it (§7.3–§7.10), measured
patterns first.

Each pattern subsection gives its measured evidence, the derivation of
§7.2 applied explicitly — `Ns`, `Nd`, `#groups`, `|P|`, `|C|`, the
membership computed from the stride vector, and whether `D` is needed — then
an abridged instantiation and full IR where the pattern has one. Full-IR listings are
on synthetic shapes chosen to make the axis roles legible; the measured
shapes are given alongside.

**Three of the seven patterns are measured** — all-to-all (§7.3), gather
(§7.4) and scatter (§7.5), plus the selection case of §7.10 that is not a
delivery at all. **The other four have no measured example among the 51**:
broadcast (§7.6, measured in separate broadcast work but by no relayout),
permute (§7.7), reduce (§7.8) and reduce-scatter (§7.9). Each says so
prominently rather than pointing at an approximation.

**Scope: one artifact.** All of §7 reads a single surface — **relayouts**,
the `STCDPOpLx` shuffle of §7.1, which is what `gather`, `all_to_all`,
`scatter` and `consume` become. Every relayout compiles to one SDSC entry
whose payload is a pair of per-core ownership tables — what each core owns
before the movement and after — and classifying a pattern means deciding,
from those two tables, which delivery op expresses the same movement (§7.2,
Step 7). The two `fold` ops of §6 have no expression on that surface at all,
because a relayout moves ownership without combining (§7.1): they are the
patterns §7 cannot speak to.

**Evidence index and census — descriptive, not normative.** One table. The
left half indexes the subsections; the right half is the measured census of
all 51 relayouts, with the divisions read off the tables and the op that
§7.2's Step 7 assigns. Its purpose is to establish **which ops a lowering
must actually emit**, and with what attribute arity — not to constrain the
ops.

| § | pattern | op | n | src division | dst division | `C` | `R` | measured evidence / attributes |
|---|---|---|---|---|---|---|---|---|
| 7.3 | All-to-all, square | `all_to_all` | 6 | `{mb:8, out:4}` | `{mb:32}` | `out` | `mb` | split `[mb]` / concat `[out]`; 8 groups, `M=K=4`. `Stcdp_QC_5` (running example), `Stcdp_QC_18`, `Stcdp_QC_30` |
| 7.3 | All-to-all, square | `all_to_all` | 3 | `{x:8, mb:4}` | `{x:32}` | `mb` | `x` | split `[x]` / concat `[mb]`; 8 groups, `M=K=4`. `Exx2_QC_1` (membership disjoint — §7.2's `P(g)`/`C(g)` independence part) |
| 7.3 | All-to-all, **non-square** | `all_to_all` | 1 | `{mb:16}` | `{mb:8, out:4}` | `mb` | `out` | split `[out]` / concat `[mb]`; 8 groups, **`M=4, K=2`**. `Add_QC_3` |
| 7.4 | all-gather to every core | `gather` | 3 | `{in:2, out:8, x:2}` | `{}` | `in`,`out`,`x` | — | `gather_dimensions = [in, out, x]` — **3 axes**. `BatchMatMulV2_QC_12` |
| 7.4 | all-gather, 4 cores idle | `gather` | 1 | `{in:32}` | `{}` | `in` | — | `gather_dimensions = [in]`; one region on 28 cores, and the one file with send-only cores. `BatchMatMulV2_QC_27` |
| 7.4 | grouped gather, drop an axis | `gather` | 6 | `{mb:8, in:4}` | `{mb:8}` | `in` | — | `gather_dimensions = [in]` |
| 7.4 | grouped gather, coarsen one axis | `gather` | 15 | `{mb:32}` | `{mb:8}` | `mb` | — | `gather_dimensions = [mb]` |
| 7.4 | grouped gather, coarsen one axis | `gather` | 3 | `{mb:16}` | `{mb:8}` | `mb` | — | `gather_dimensions = [mb]`. `BatchMatMulV2_QC_0` (1 axis, `P ⊊ C`) |
| 7.5 | pure split | `scatter` | 12 | `{y:16}` | `{y:32}` | — | `y` | `scatter_dimensions = [y]`. `Mul_QC_1` (16 groups of 2) |
| 7.6 | Broadcast | `consume` | — | `{h:8}` on 8 cores | `{h:8}` × 4 cores | — | — | **none of the 51.** Consumer set widened to the 4 holders; the row is from separate broadcast work (PR #4061) |
| 7.7 | Per-tile sync / permute | `consume` + `D` | — | — | — | — | — | **none of the 51.** The only patterns that need `D` at all (§7.2, Step 6) |
| 7.8 | Reduce | `reduce` | 0 | — | — | — | — | **none of the 51.** A relayout moves ownership without combining, so no relayout is a reduce |
| 7.9 | Reduce-scatter | `reduce_scatter` | 0 | — | — | — | — | **none of the 51**, and unmeasured elsewhere too — the one op in §6 with no measured path of any kind |
| 7.10 | selection, not a partition | none | 1 | `{mb:8, out:4}` | *selection* | — | — | `Stcdp_QC_38` — coverage 1/512, **not a work-division pair**; §7.2 Step 7's guard row |

The `n` column sums to 51 across the relayout rows and reproduces exactly:
applying Step 7 mechanically to all 51 files yields 28 `gather`, 10
`all_to_all`, 12 `scatter` and 1 guard-row selection. The broadcast row is
the one row **not** from the 51 — no measured relayout is a broadcast,
because every source region has a single holder and replication appears only
on the destination side (§7.6). It is kept because the pattern is measured
elsewhere and is expressible in the artifact.

**Which ops this requires.** Four of the six delivery ops, with these
arities:

| Op | measured files | arity needed |
|---|---|---|
| `inter_tile_gather` | 28 | up to **3 axes** |
| `inter_tile_all_to_all` | 10 | 1 axis each side, but **non-square** `M ≠ K` |
| `inter_tile_scatter` | 12 | 1 axis |
| `inter_tile_consume` | broadcast work | — |

**`inter_tile_reduce` and `inter_tile_reduce_scatter` have no relayout
expression — structurally, not incidentally.** It is not that a relayout
happens not to combine: the artifact contains **no compute of any kind** and
no combiner field (§7.1), so there is nothing for a fold to become.
Consistently, **no output byte in the measured set has more than one
producer** (0/51; input pieces are a disjoint exact tiling in 51/51), so the
question of what to do with two contributions never arises. The consequence
is about evidence, not expressiveness: nothing in §7 can attest either
`fold` op, and §7.8 and §7.9 say so rather than approximating one.

Two consequences for implementation order. `gather` carries the most
measured weight *and* the widest arity, so its multi-axis path cannot be
deferred. And `all_to_all`'s non-square case is measured, not hypothetical,
so `P == C` is not a safe simplifying assumption.

**Two negative results the census also establishes.**
`producer_dependency_per_consumer` is **never needed**: every group of all
51 has a complete bipartite overlap graph, so the default full-barrier
reading is always correct (§7.2, Step 6). This is the strongest negative
result in the measured set — the entire dependency-set mechanism, and R3–R7
with it, is unexercised. And **uniformity holds on all 51 measured files**,
so R6/R7 are so far confirmed rather than assumed.

**Still unmeasured.** Two recorded patterns match no file: one needs a side
with 8 active cores (measured counts are 1, 16, 28, 32), and one is an axis
transpose `{A:4, B:8} → {A:8, B:4}`. The transpose classifies under Step 7's
row 3 — `C = {B}`, `R = {A}`, 16 components, `M = K = 2` — but with nothing
coarsened or refined in the *region* sense it could equally be read as row 1,
and only divisions keyed by physical axis settle which.

### 7.1 The artifact `torch-spyre` emits

A KTIR inter-tile op must eventually become something the target can
execute. For every LX→LX movement in the measured set, that something is a
single SDSC **data DSC** whose `op` names `STCDPOpLx` — the artifact
`torch-spyre` emits
([torch-spyre PR #4300](https://github.com/torch-spyre/torch-spyre/pull/4300)).
Knowing its shape is what makes the design decisions of §1–§6
non-arbitrary. This subsection describes the artifact and nothing else;
how anything downstream of `torch-spyre` consumes it is out of scope and
is not relied on anywhere in this document.

The container:

```json
DataOpDsc {
  coreIdsUsed_ : [0, 1, 2, ..., 31]
  dimPool_     : ["mb",  "out", "x", "y"]
  primaryDs_   : [PdsInfo{name_, dimNames}]
  labeledDs_   : [ LdsInfo inp, LdsInfo out ]   # exactly 2, by index
  op           : STCDPOpLx
}
```

`labeledDs_[0]` is the input side and `labeledDs_[1]` the output side, and
**position is the only discriminator available**: both sides carry the
same `ldsName_` in all 51 measured relayouts, so they cannot be told apart
by name.

An `LdsInfo` is one side's description of the whole tensor plus its
decomposition into pieces. Abridged from the running example of §7.3,
input side:

```json
{
    "layoutDimOrder_" : ["x", "out", "mb", "y"],
    "stickDimOrder_" : ["out"],
    "dimToLayoutSize_" : {"mb" : 8, "out" : 128, "x" : 512, "y" : 1},
    "dimToStickSize_" : {"out" : 64},
    "validGap_" : {"mb" : [ [8,0] ], "out" : [ [128,0] ], "x" : [ [512,0] ], "y" : [ [1,0] ]},
    "PieceInfo": [
        ...,
        {
            "key_" : "p1",
            "dimToStartCordinate" : {"mb" : 2, "out" : 0, "x" : 0, "y" : 0},
            "dimToSize_" : {"mb" : 2, "out" : 128, "x" : 64, "y" : 1},
            "validGap_" : {"mb" : [ [2,0] ], "out" : [ [128,0] ], "x" : [ [64,0] ], "y" : [ [1,0] ]},
            "PlacementInfo" : [
            { "type":"lx", "memId": [1], "startAddr":  {
                "dim_prop_func" : [ { "Map" : {} } ],
                "dim_prop_attr" : [ { "factor_" : 1, "label_" : "time" } ],
                "data_" : { "[0]" :["524288"] }
                } }
            ]
        }, ...
    ]
}
```

Six things to take from this.

**The whole movement is two piece decompositions of one logical tensor.**
There is no edge list, no send/receive pairing, and nothing naming a
route. Both sides describe the *same* tensor, cut up differently; the
movement is the difference between the two cuttings. Routing is therefore
*derived*, and §7.2 is the derivation: a movement exists between an input
piece and an output piece exactly when their boxes intersect on **every**
dim, and it crosses cores exactly when their holder sets differ.

**A `PieceInfo` is a rectangular hyperslab** — `(dimToStartCordinate,
dimToSize_)` per dim, in elements. There is no stride field, no index
vector and no coordinate set anywhere in it, so all the geometry is
`start + size - 1`. An index-vector gather is not expressible in this
artifact at all: nothing in it can carry an index operand. This matters
for §4 — no delivery op needs to express anything richer than an offset
and an extent, which is why none of them carries an offset at all (§7.10).

**`PlacementInfo.memId` is a list of core ids**, and it is the only thing
that makes a relayout a communication. Every `PlacementInfo` across the 51
files has `type: "lx"`; the observed `memId` lengths are 1 (2112 entries),
4 (192), 28 (1) and 32 (3). A length greater than one is how replication
is expressed: one output region named by four core ids is a four-way
multicast of that region. This field is also **the only source of group
membership**, though membership itself turns out to be a compact affine
function of the piece's slice indices rather than a table — §7.2's Step 1.

**There is no compute anywhere in this artifact.** A data DSC names no
operation to apply and carries no combiner field of any kind: a relayout
moves bytes and does nothing to them. That is why the two `fold` ops of §6 have no
relayout expression (§7's preamble) and why the producer region of §2.2
needs no compute either.

**`op` carries nothing an emitter must compute.** It names `STCDPOpLx` and
then a fixed collection of scalar flags and empty lists, and the whole
`op` object is **byte-identical in all 51 measured files**. Nothing in the
measured set exercises any of it. So the only parts of the artifact that
vary — hence the only parts an emitter must actually produce — are **the
two piece tables and their core maps**. That is the target §6's ops must be
able to hit. Whether any individual `op` field is meaningfully authorable
cannot be settled from the artifact and is recorded as unverified (§9.4).

**`validGap_` can be ignored by an emitter.** It is a per-dim ordered run
list of `(valid, gap)` pairs; `Σ(valid + gap)` is the **physical span** of
the piece — the allocation pitch — while `dimToSize_` is what actually
gets transferred. The distinction is `cudaMemcpy2D`'s `dpitch` versus
`width`. In the measured set it is uniformly trivial: **all 9438 entries
across the 51 files are `[[full_extent, 0]]`**, i.e. pitch equals width.
Emitting `[[extent, 0]]` per dim therefore reproduces every measured file
exactly.

**Axis names versus axis indices.** The artifact speaks axis *names*
(`mb`, `in`, `out`, `x`, `y`, …) — the vocabulary the ownership tables and
every axis set in §7 use. The op attributes of §6 are `i64` arrays of axis
*indices* into `T_p`. A lowering resolves each named axis to its position in
the producer tile type, preserving list order, which §4 fixes as slowest- to
fastest-varying. So the axis sets stay artifact symbol names until lowering,
and the ops themselves stay layout-agnostic (§9.3). Each example below states
the `T_p` axis order it uses.

**`layoutDimOrder_` is not an ordering this document relies on, and now
there is evidence why.** It enumerates the axes present, but which axis order
a frontend gives `T_p` is that frontend's choice, so long as the dim
attributes index into it consistently. Concretely, `layoutDimOrder_`
restricted to the cut axes **disagrees with the digit order implied by the
core-strides of §7.2's Step 1** on two of three files where the comparison is
non-trivial: it agrees in `Stcdp_QC_18` (`['x','mb']` against the implied
`(x, mb)`), and is reversed in `Exx2_QC_1` (`['mb','out']` against
`(out, mb)`) and `BatchMatMulV2_QC_15` (`['mb','in']` against `(in, mb)`).
An implementer must therefore not take `layoutDimOrder_` for a stride order.

**The stick-level assumption holds.** 20 of the 51 files coarsen or refine
the sticked axis, and every piece size on it is an exact stick multiple. No
measured relayout cuts a stick, so nothing in §4 or §6 has to express a
sub-stick extent. (Descriptive.)

**The per-core LX byte addresses in `startAddr` have no counterpart on any
op here.** In KTIR an address is named by `ktdp.construct_memory_view`,
upstream of the partial.

### 7.2 From two piece tables to the attributes

The artifact is two piece tables. Everything the delivery ops need —
`groups`, `P`, `C`, `D`, and which op — is derivable from them by a fixed
procedure. The one thing not derivable from the divisions alone is the
stride vector, and that is what must be read.

**Step 1 — read the two grids.** Per side, collect the **distinct boxes**
`(dimToStartCordinate, dimToSize_)`. Per axis `a`, let `Ns(a)`/`Nd(a)` be the
count of distinct slices that side's table contains along `a` — an axis
absent from an entry counts 1. Boxes must be counted **distinct**, not as
`PieceInfo` entries: replication has two encodings — a multi-entry `memId`
on one box, or several entries with identical boxes — and counting entries
would give the wrong count under the second encoding. Also read the
**stride vector and base**, the radix map
`core = base + Σ_a stride_a · sliceIndex_a`, for whichever side has a
single holder per box; `layoutDimOrder_` does not reliably give the digit
order, so the vector must be measured from the pieces (§7.1).

Then the **participation guard**: if any distinct region overlaps nothing
on the other side, stop — this is a selection, not a delivery (§7.10). One
measured entry has 28 of 32 source regions participating in no overlap;
counting those as producers would overcount by 8×.

**Step 2 — group count.** `#groups = ∏_a gcd(Ns(a), Nd(a))`. An axis cut on
only one side contributes `gcd(n, 1) = 1` and separates nothing, so **axes
compose multiplicatively and never conflict**. Running example (§7.3):
`mb` gives `gcd(4,1) = 1`, `x` gives `gcd(8,32) = 8` — 8 groups from `x`
alone, and `mb` cannot conflict with `x`. Verified: this formula equals the
true connected-component count of the overlap graph in all 51 of 51
measured relayouts.

At most one axis is ever group-determining: 47 of the 51 have exactly one
axis with `gcd > 1`; the other 4 have none and are a single group. So the
group index **is** that axis's block index, and no multi-axis decode of `g`
is needed. A case with two group-determining axes would need a mixed-radix
decode of `g`, and is unmeasured.

**Step 3 — invert the radix map.** The strides are odometer weights —
sorted strides multiply up by the radices — so the map inverts digit by
digit:

```
x_a(i) = ((i - base) floordiv stride_a) mod N_a
```

Each `x_a(i)` is an affine expression in `i` using only `floordiv`/`mod` by
constants, so it is legal inside an `IntegerSet`. This is the step that
carries Step 1's radix map from slice-index space into tile-id space, which
is where the attributes live.

**Step 4 — emit one constraint per cut axis.** With `k_a = N_a /
gcd(Ns(a), Nd(a))` the within-group extent along `a`, and `b(g)` the
group-determining axis's block index (`= g`, by Step 2):

```
b(g) · k_a  ≤  x_a(i)  <  b(g) · k_a + k_a
```

Three collapses do the real work:

- `k_a = 1` → the constraint becomes an equality `x_a(i) == b(g)`
- `gcd = 1` → the axis is not group-determining, the constraint is
  vacuous, **drop the axis**
- smallest stride `> 1` → add `(i - base) mod stride_min == 0`

Cardinality falls out here: `|P(g)| = ∏_a k_a`. If the group sits on a
**high-order** digit the set collapses to a contiguous range; if it sits on
a **low-order** digit it is a coset and needs `mod`. `mod` is a valid
affine-set constraint and is accepted by `ktdp.inter_tile_produce` today —
verified by round-trip through the built `ktir-opt` — so a coset needs no
fallback and no design change.

The four measured shapes below have all been verified to parse and
round-trip:

| entry / side | stride vector (radix) | resulting affine set |
|---|---|---|
| `Stcdp_QC_5` in | `mb:1(4), x:4(8)`; group on `x`, high digit | `(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)` |
| `Exx2_QC_1` in | `mb:1(8), out:8(4)`; group on `mb`, low digit | `(i)[g] : ((i - g) mod 8 == 0, i >= 0, -i + 31 >= 0)` |
| `BatchMatMulV2_QC_12` in | `x:1(2), in:2(2), out:4(8)`; group on `out` | `(i)[g] : ((i floordiv 4) mod 8 - g == 0, i >= 0, -i + 31 >= 0)` |
| `Add_QC_3` in | `mb:2(16)`, base 0 | `(i)[g] : (i mod 2 == 0, (i floordiv 2) mod 16 - g == 0, i >= 0, -i + 31 >= 0)` |

**Step 5 — repeat on the output side for `C(g)`.** Same `g`, but the
output side's `Nd(a)`, stride vector and base. Nothing here is shared with
Steps 1–4 beyond `g` itself — the two sides' stride vectors are independent
inputs, which is why `P(g)` and `C(g)` need not relate (below).

**Step 6 — is `D` needed?** `producer_dependency_per_consumer` is required
exactly when the within-group overlap graph is not complete bipartite. It
is complete `K(|P(g)|, |C(g)|)` in every group of all 51 measured
relayouts, so no measured relayout needs `D`, and the default full-barrier
reading of §3.4 always applies to them. Its only uses are the routing and
permutation patterns of §7.7, which have no measured example.

**Step 7 — which delivery op.** An axis is **coarsened** when `Ns(a) >
Nd(a)` (fewer, larger pieces after the move — data must be *assembled*
along it) and **refined** when `Nd(a) > Ns(a)` (data must be *split*).
Write `Coarse = {a : Ns(a) > Nd(a)}`, `Refine = {a : Nd(a) > Ns(a)}`. Two
guard rows run first, then four classification rows of which exactly one
fires:

| # | condition | result |
|---|---|---|
| — | `prod(Ns(a)) != len(src_regions)` or `prod(Nd(a)) != len(dst_regions)`, or either side's distinct regions do not cover the tensor | **not a work-division pair** — enumerate regions instead. Check first |
| — | any axis ragged (non-uniform overlap) | *insufficient information* — no single op has uniform dependency-set cardinality (R6) |
| 1 | `Coarse = ∅` and `Refine = ∅` | regions identical: `no op needed` if every core's region is its own, else `inter_tile_consume` — a **relocation**, or a **broadcast** where destination regions are shared |
| 2 | `Coarse = ∅`, `Refine ≠ ∅` | `inter_tile_scatter`, `scatter_dimensions = Refine` |
| 3 | `Coarse ≠ ∅`, `Refine ≠ ∅`, `prod(Nd(a)) == num_cores` | `inter_tile_all_to_all`, `split_dimensions = Refine`, `concat_dimensions = Coarse`; one group per component |
| 4 | `Coarse ≠ ∅` otherwise | `inter_tile_gather`, `gather_dimensions = Coarse`; consumers per group = the destination region's holders |

The dimension attributes are the axis sets themselves, in the order §4
fixes (§1.1). Rows 2–4 return *insufficient information* when a source
region has several holders, since R8 gives each consumer tile exactly one
source and the tables do not say which holder transmits (§9.2).

The coverage clause of the first guard row sums each **distinct** region's
element count against the element count of the value being delivered — not
per core (an all-gather's replicated destination would overshoot), and
against the delivered value rather than the original tensor, so a
select-then-deliver does not trip the guard it exists to satisfy (§7.10).
Nothing in the artifact requires either side to cover the tensor; the guard
row is this document's modelling choice that a selection is not a delivery,
and `Stcdp_QC_38` (§7.10) is the one measured file that fails it — the
cardinalities of Step 4 are wrong on that file, which is why the guard runs
first.

Row 3's `prod(Nd(a)) == num_cores` is the one irreducibly global test:
holding the division fixed and varying the core count changes the op, so no
per-axis quantity can see it. Verified: applying these rows mechanically to
all 51 measured relayouts yields 28 `gather`, 10 `all_to_all`, 12 `scatter`
and 1 guard-row selection, matching §7's preamble census.

**`P(g)` and `C(g)` are independent.** Each side has its own stride vector
(Step 1), so the two sets carry no required relation; every relation below
is measured, on files with otherwise identical division shapes:

| file | `g` | `P(g)` | `C(g)` | relation |
|---|---|---|---|---|
| `Stcdp_QC_18` | 0 | `{0,1,2,3}` | `{0,1,2,3}` | `P == C` |
| `Exx2_QC_1` | 0 | `{0,8,16,24}` | `{0,1,2,3}` | overlap `{0}` |
| `Exx2_QC_1` | 1 | `{1,9,17,25}` | `{4,5,6,7}` | **disjoint** |
| `Add_QC_3` | 1 | `{20,22}` | `{5,13,21,29}` | **disjoint** |
| `BatchMatMulV2_QC_0` | 0 | `{0,2}` | `{0,1,2,3}` | `P ⊊ C` |

This is why R13 does not hold for the copy-only ops. **Confinement** —
every edge runs from a core in `P(g)` to a core in `C(g)` for the same `g`
— is a different property: it is forced (§3.2), and it does not imply any
intersection between the two sets.

**One end-to-end worked instance.** `Stcdp_QC_5` (§7.3.1), source `mb:4,
x:8`, destination `x:32`. Step 1: the boxes give `Ns = {mb:4, x:8}`,
`Nd = {mb:1, x:32}` (`mb` absent from the destination table); source stride
vector `mb:1, x:4`, base 0, destination stride vector `x:1`, base 0; every
region single-holder, and all regions participate. Step 2:
`#groups = gcd(4,1) · gcd(8,32) = 1 · 8 = 8`; `x` alone is
group-determining, so `b(g) = g`. Step 3: `x_x(i) = (i floordiv 4) mod 8`.
Step 4: `x` is the high-order digit (stride 4 spans `mb`'s full range), so
the equality collapses to the contiguous bound
`producer_tiles_per_group = (i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)`,
and `|P(g)| = 4`. Step 5 on the destination (`x` stride 1, the only cut
axis) gives the same set for `consumer_tiles_per_group`, `|C(g)| = 4`.
Step 6: the group graph is `K(4,4)`, complete, so `D` is absent. Step 7:
`mb` (`4 → 1`) is `Coarse`, `x` (`1 → 4`) is `Refine`, both nonempty, and
the destination is cut 32 ways across 32 cores, so row 3 fires:
`inter_tile_all_to_all`, `split_dimensions = [x]`,
`concat_dimensions = [mb]`, 8 groups, 4 producers and 4 consumers each.

### 7.3 All-to-all  →  `inter_tile_produce` + `inter_tile_all_to_all`

**Measured evidence: 10 of the 51 relayouts**, all with 8 groups. This is
the best-evidenced pattern in the document and the one the running example
belongs to, so it comes first.

#### 7.3.1 The running example — `Stcdp_QC_5`

One measured relayout, worked from the artifact of §7.1 to the KTIR op. The
tensor is `mb:8 × out:128 × x:512 × y:1`, f16 — 524288 elements. `out` is
the sticked axis, stick 64, so 128 elements is 2 sticks. This document
writes the tensor as `[y, mb, out, x] = [1, 8, 128, 512]`; that order is
this document's choice of `T_p` axis order and the dim attributes below
index into it (§7.1).

**Before and after.** Each side cuts the tensor into 32 pieces, one per
core, single-holder throughout. Abridged to the first and last group:

| core | before: `(mb, x)` | after: `(mb, x)` |
|---|---|---|
| 0 | `mb[0:2] × x[0:64]` | `mb[0:8] × x[0:16]` |
| 1 | `mb[2:4] × x[0:64]` | `mb[0:8] × x[16:32]` |
| 2 | `mb[4:6] × x[0:64]` | `mb[0:8] × x[32:48]` |
| 3 | `mb[6:8] × x[0:64]` | `mb[0:8] × x[48:64]` |
| … | … | … |
| 28 | `mb[0:2] × x[448:512]` | `mb[0:8] × x[448:464]` |
| 29 | `mb[2:4] × x[448:512]` | `mb[0:8] × x[464:480]` |
| 30 | `mb[4:6] × x[448:512]` | `mb[0:8] × x[480:496]` |
| 31 | `mb[6:8] × x[448:512]` | `mb[0:8] × x[496:512]` |

`out` is uncut on both sides (`out[0:128]` everywhere) and `y` is size 1.
Every piece is `16384` elements on both sides — the movement conserves
volume per core and changes only *which* elements.

**§7.2 applied.**

| quantity | value | how |
|---|---|---|
| `Ns` | `{mb:4, x:8, out:1, y:1}` | distinct slices in the source table (Step 1) |
| `Nd` | `{mb:1, x:32, out:1, y:1}` | distinct slices in the destination table |
| `#groups` | `gcd(4,1) · gcd(8,32) · 1 · 1 =` **8** | Step 2 — `x` alone; `mb` contributes 1 |
| `\|P(g)\|` | `32 / 8 =` **4** regions | Step 4 |
| `\|C(g)\|` | `32 / 8 =` **4** regions | Step 4 |
| `P(g)` | `{4g, 4g+1, 4g+2, 4g+3}` | Step 3–4 — strides `mb:1, x:4`, so `core = mb + 4x` |
| `C(g)` | `{4g, 4g+1, 4g+2, 4g+3}` | Step 5 — stride `x:1`, 4 holders per share |
| `D` | **not needed** | Step 6 — the group graph is `K(4,4)` |

**This is the direct answer to "does `mb` conflict with `x`?" — no, and it
cannot.** The destination is uncut on `mb`, so `mb`'s gcd factor is 1 and it
separates nothing; the 8 groups are `x`-blocks. Geometrically: core 0's
*before* box (`x[0:64]`) cannot overlap core 4's *after* box (`x[64:80]`),
so no data crosses between them, and within an `x` block every source box
overlaps every destination box because `mb[2l:2l+2] ⊂ mb[0:8]` always.
`mb` decides *what* is assembled, not *who* exchanges.

| `g` | `x` block | `P(g)` | `C(g)` |
|---|---|---|---|
| 0 | `x[0:64]` | `{0, 1, 2, 3}` | `{0, 1, 2, 3}` |
| 1 | `x[64:128]` | `{4, 5, 6, 7}` | `{4, 5, 6, 7}` |
| … | … | … | … |
| 7 | `x[448:512]` | `{28, 29, 30, 31}` | `{28, 29, 30, 31}` |

**`P(g) == C(g)` here is a property of this file's stride vectors, not of
all-to-all.** The source gives `mb` stride 1 inside `x` stride 4, so a
group's producers are the contiguous run `{4g … 4g+3}`, and the destination's
`x` stride 1 puts its four holders on the same run. `Exx2_QC_1` below has the
same divisions and the same cardinalities with `P(g) ∩ C(g) = ∅` for most
`g`, because its strides differ (§7.2's `P(g)`/`C(g)` independence part).

**Which op.** Within a group, `mb` goes from 4 pieces to 1 (data must be
**assembled** along it) while `x` goes from 1 piece to 4 (data must be
**split** along it). Both at once, with the destination cut 32 ways across
32 cores, is §7.2 Step 7's row 3: **`inter_tile_all_to_all`,
`split_dimensions = [x]`, `concat_dimensions = [mb]`**, 8 groups, 4
producers and 4 consumers each.

**The KTIR.** Each producer's partial is its `before` box,
`T_p = tensor<1x2x128x64xf16>` (`[y, mb, out, x]`). Splitting `x` by
`C = 4` and multiplying `mb` by `P = 4` gives `tensor<1x8x128x16xf16>` —
exactly each consumer's `after` box. §4's type rules and the measured
destination table agree element for element, and since `P == C` the element
count is conserved (`2 × 64 = 8 × 16`). The attributes are `i64` arrays of
axis *indices* into `T_p`, so `[x]` becomes `[3]` and `[mb]` becomes `[1]`.

```mlir
// 8 groups of 4 tiles; every tile both produces and consumes.
#group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#all_groups  = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

%future = ktdp.inter_tile_produce
    producer_tiles_per_group = #group_tiles
    -> !ktdp.tile_future<(tensor<1x2x128x64xf16>), groups = #all_groups>
{
  ^bb0(%gid: index):
    // memory ops only (§2.2); anchors depend on %gid — worked in §2.2
    ktdp.yield_partial %partial : tensor<1x2x128x64xf16>
}

// producer_dependency_per_consumer is absent: the group graph is K(4,4),
// so the default full-barrier reading of §3.4 is the correct one (Step 6).
%relaid = ktdp.inter_tile_all_to_all(%future)
    consumer_tiles_per_group = #group_tiles,
    split_dimensions         = [3],   // x
    concat_dimensions        = [1]    // mb
    : !ktdp.tile_future<(tensor<1x2x128x64xf16>), groups = #all_groups>
      -> tensor<1x8x128x16xf16>
```

**The local index is measured, not conventional (§3.3).** Consumer `4g+l`
has local index `l` in ascending tile-id order and holds
`x[64g+16l : +16]`, which is `split_dimensions` chunk `l` — so `l` selects
the slice. Producer `4g+l_p` holds `mb[2 l_p : +2]`, and in each consumer's
assembled `mb[0:8]` that contribution sits at offset `2 l_p`. Assembly is
therefore in ascending *producer* local-index order, exactly as §3.3 and
§4's "Which slice a tile gets" state — and the artifact confirms it rather
than merely permitting it.

#### 7.3.2 The rest of the measured spread

The three remaining division shapes, each with §7.2 applied.

| file | `Ns` | `Nd` | `#groups` | `\|P\|` | `\|C\|` | `P(g)` (Step 3–4) | `C(g)` (Step 5) | `D`? |
|---|---|---|---|---|---|---|---|---|
| `Stcdp_QC_18`, `Stcdp_QC_30` | `{mb:4, x:8}` | `{x:32}` | 8 | 4 | 4 | `{4g..4g+3}` | `{4g..4g+3}` | no |
| `Exx2_QC_1` … `Exx2_QC_6` | `{mb:8, out:4}` | `{mb:32}` | 8 | 4 | 4 | `{g, g+8, g+16, g+24}` | `{4g..4g+3}` | no |
| `Add_QC_3` | `{mb:16}` | `{mb:8, out:4}` | 8 | **2** | 4 | 2 cores, e.g. `{20,22}` | 4 cores, strided, e.g. `{5,13,21,29}` | no |

Working the second row by Step 2: `gcd(8,32) · gcd(4,1) = 8 · 1 = 8`
groups, from `mb` alone. Third row: `gcd(16,8) · gcd(1,4) = 8 · 1 = 8`
groups, from `mb` alone again — and here the destination's *new* cut on
`out` contributes `gcd(1,4) = 1`, the mirror of the running example's `mb`.
Step 4 on the third row: `16/8 = 2` producers against `32/8 = 4`
consumers.

Three things this spread settles.

**`P(g)` and `C(g)` may be disjoint even on a square exchange.**
`Exx2_QC_1` has the running example's cardinalities and disjoint sets for
most `g`, because its source strides are `mb:1, out:8` against a destination
stride of `mb:1` — same division, different stride vectors (§7.2's
`P(g)`/`C(g)` independence part). So §1.1's `all` cells mean matching
cardinalities, not
identical tile sets.

**`M ≠ K` is measured.** `Add_QC_3` has 2 producers and 4 consumers per
group, so `P == C` is not a safe simplifying assumption and R12 must be
stated independently of R7 + R9 (§5). Its type arithmetic checks out
against the artifact: `T_p` is `mb:32 × out:4096`, concat `mb` by `P = 2`
gives 64, split `out` by `C = 4` gives 1024, and the measured destination
box is exactly `mb:64 × out:1024`. Note the element count is **not**
conserved (`131072 → 65536`), precisely because `P ≠ C` (§4).

**No group in any of the ten needs `D`.** Every one is complete bipartite
(Step 6).

#### 7.3.3 The op on synthetic shapes

The IR below is on synthetic shapes chosen to make the sequence/head axis
roles legible; §7.3.1 gives the same op on the measured shapes.

```mlir
// 4 tiles per group, 8 groups (32 tiles total).
#all_group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#all_groups      = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

// Sequence-parallel production: every tile owns a 128-row shard of all 4 heads.
%partial_future = ktdp.inter_tile_produce
    producer_tiles_per_group = #all_group_tiles
    -> !ktdp.tile_future<(tensor<128x1x4x64xf16>), groups = #all_groups>
{
  ^bb0(%gid: index):
    ktdp.yield_partial %partial_4d : tensor<128x1x4x64xf16>
}

// Head-parallel consumption: split dim 2 (heads) across the 4 consumers,
// regather dim 0 (sequence) from the 4 producers.
// split_dimensions = [2] → 4 / 4 = 1;  concat_dimensions = [0] → 128 * 4 = 512.
%relaid = ktdp.inter_tile_all_to_all(%partial_future)
    consumer_tiles_per_group = #all_group_tiles,
    split_dimensions                = [2],
    concat_dimensions               = [0]
    : !ktdp.tile_future<(tensor<128x1x4x64xf16>), groups = #all_groups> -> tensor<512x1x1x64xf16>
// Every tile is both producer and consumer; P == C == 4, so the element count
// is conserved (128*4 = 512*1) even though the type changes.
```

#### 7.3.4 Full IR — sequence-parallel to head-parallel (512×8×4×64)

**Layout and partitioning.** `A`, `B`, and `E` are `tensor<512x8x4x64xf16>`
in global memory. The four axes have distinct roles:

- Dim 0 (size 512): the **gather axis** — sequence. Sharded 4 ways before
  the op, whole after it.
- Dim 1 (size 8): the **group axis** — 8 groups.
- Dim 2 (size 4): the **scatter axis** — heads. Whole before the op,
  sharded 4 ways after it.
- Dim 3 (size 64): vector / stick axis, preserved.

32 tiles, 8 groups of 4. `g = t / 4`, `l = t % 4`. Before the op, tile
`(g, l)` owns sequence shard `l`: it reads `[l*128 : l*128+128, g, *, *]`,
shape `<128x1x4x64>`, and its partial is `A + B` over those rows. After the
op, tile `(g, l)` owns head `l` for the whole sequence, shape
`<512x1x1x64>`, and writes it back to `E[*, g, l, *]`.

This is the pattern a sequence-parallel prefill hands to a head-parallel
attention: the ownership axis moves from dim 0 to dim 2 in one collective,
with no tile ever holding more than its `1/4` share.

```mlir
#A_view_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 511 >= 0,
     d1 >= 0, -d1 + 7   >= 0,
     d2 >= 0, -d2 + 3   >= 0,
     d3 >= 0, -d3 + 63  >= 0)>

// A/B access tile for the producer: 128x1x4x64 anchored at [l*128, g, 0, 0].
#AB_tile_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 == 0,
     d2 >= 0, -d2 + 3   >= 0,
     d3 >= 0, -d3 + 63  >= 0)>

// E access tile for the consumer: 512x1x1x64 anchored at [0, g, l, 0].
#E_tile_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 511 >= 0,
     d1 == 0,
     d2 == 0,
     d3 >= 0, -d3 + 63  >= 0)>

#identity_4d = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

#group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#all_groups  = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

module {
  func.func @inter_tile_all_to_all_relayout() {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %row_shard = arith.constant 128 : index   // 512 / 4

    %A_start = arith.constant 1024    : index
    %B_start = arith.constant 2098176 : index
    %E_start = arith.constant 4195328 : index

    %A_view = ktdp.construct_memory_view %A_start, sizes: [512, 8, 4, 64],
        strides: [2048, 256, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<512x8x4x64xf16>
    %B_view = ktdp.construct_memory_view %B_start, sizes: [512, 8, 4, 64],
        strides: [2048, 256, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<512x8x4x64xf16>

    // Per-tile compute (function-scope SPMD).
    %t = ktdp.get_compute_tile_id : index
    %g = arith.divui %t, %c4 : index
    %l = arith.remui %t, %c4 : index
    %row_anchor = arith.muli %l, %row_shard : index

    %A_access = ktdp.construct_access_tile %A_view[%row_anchor, %g, %c0, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_4d
    } : memref<512x8x4x64xf16> -> !ktdp.access_tile<128x1x4x64xindex>
    %B_access = ktdp.construct_access_tile %B_view[%row_anchor, %g, %c0, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_4d
    } : memref<512x8x4x64xf16> -> !ktdp.access_tile<128x1x4x64xindex>

    %A_tile = ktdp.load %A_access
                : !ktdp.access_tile<128x1x4x64xindex> -> tensor<128x1x4x64xf16>
    %B_tile = ktdp.load %B_access
                : !ktdp.access_tile<128x1x4x64xindex> -> tensor<128x1x4x64xf16>

    // No reduction — the summed sequence shard is this tile's partial; the
    // all-to-all redistributes it from sequence-sharded to head-sharded.
    %AB_init = tensor.empty() : tensor<128x1x4x64xf16>
    %partial_4d = linalg.add ins(%A_tile, %B_tile
                                 : tensor<128x1x4x64xf16>, tensor<128x1x4x64xf16>)
                             outs(%AB_init : tensor<128x1x4x64xf16>)
                             -> tensor<128x1x4x64xf16>

    // Produce: every tile contributes its sequence shard to the future.
    %partial_future = ktdp.inter_tile_produce
        producer_tiles_per_group = #group_tiles
        -> !ktdp.tile_future<(tensor<128x1x4x64xf16>), groups = #all_groups>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %partial_4d : tensor<128x1x4x64xf16>
    }

    // All-to-all: consumer l takes head slice l (split_dimensions = [2], 4 / 4 = 1)
    // from each of the 4 producers, and concatenates them along the sequence
    // axis (concat_dimensions = [0], 128 * 4 = 512) in ascending producer local-index
    // order. The producer's local index picks the destination row block;
    // the consumer's local index picks the head. No combiner, no identity.
    %relaid = ktdp.inter_tile_all_to_all(%partial_future)
        consumer_tiles_per_group = #group_tiles,
        split_dimensions                = [2],
        concat_dimensions               = [0]
        : !ktdp.tile_future<(tensor<128x1x4x64xf16>), groups = #all_groups>
          -> tensor<512x1x1x64xf16>

    // Post-exchange: tile (g, l) now owns head l for the whole sequence and
    // writes it to E[*, g, l, *]. Every tile is a consumer, so unlike the
    // gather example there is no idle tile after the collective.
    %E_view = ktdp.construct_memory_view %E_start, sizes: [512, 8, 4, 64],
        strides: [2048, 256, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<512x8x4x64xf16>

    %E_access = ktdp.construct_access_tile %E_view[%c0, %g, %l, %c0] {
        access_tile_set = #E_tile_set, access_tile_order = #identity_4d
    } : memref<512x8x4x64xf16> -> !ktdp.access_tile<512x1x1x64xindex>

    ktdp.store %relaid, %E_access
              : tensor<512x1x1x64xf16>, !ktdp.access_tile<512x1x1x64xindex>

    return
  }
}
```

### 7.4 Gather  →  `inter_tile_produce` + `inter_tile_gather`

**Measured evidence: 28 of the 51 relayouts** — the largest group, and the
one with the widest attribute arity. Three of them bracket the range, with
§7.2 applied to each.

| file | `Ns` | `Nd` | `#groups` (Step 2) | `\|P\|` | `\|C\|` (Step 4) | `P(g)` / `C(g)` (Step 3–5) | `D`? (Step 6) | `gather_dimensions` |
|---|---|---|---|---|---|---|---|---|
| `BatchMatMulV2_QC_0` | `{mb:16}` | `{mb:8}` | `gcd(16,8) =` 8 | 2 | 1 share | `P = {0,2}`, `C = {0,1,2,3}` — **4 holders of the one share** | no, `K(2,1)` | `[mb]` — 1 axis |
| `BatchMatMulV2_QC_12` | `{in:2, out:8, x:2}` | `{}` | `1·1·1 =` **1** | 32 | 1 share | `P =` all 32, `C =` all 32 holding the single share | no, `K(32,1)` | `[in, out, x]` — **3 axes** |
| `BatchMatMulV2_QC_27` | `{in:32}` | `{}` | `gcd(32,1) =` **1** | 32 | 1 share | `P =` all 32, `C =` 28 cores; 4 idle | no, `K(32,1)` | `[in]` — 1 axis |

Note Step 2 on the last two rows: with the destination uncut on every dim,
every gcd is 1 and there is exactly **one group** — a single 32-way
all-gather, not several small exchanges. That is the opposite extreme from
the running example and it falls out of the same formula.

Four facts to carry into an implementation.

**`|C|` is a share count, not a core count, and this op is where they
diverge.** All three rows have `|C(g)| = 1` — one destination share — while
the *holders* of that share number 4, 32 and 28 respectively. `gather`'s
`concat` type rule does not use `C` at all (§4), which is why the
divergence is harmless here and would not be elsewhere.

**Row 1 is `P ⊊ C`** (§3.2): half the consumers never produce, which is
what closes R13 for this op (§5).

**Row 2 is the measured three-axis concat.** Its per-axis factors are
`(2, 8, 2)`, product `32 = P`; the tensor is `in:128 × out:512 × x:8` and
each producer's share is `in:64 × out:64 × x:4`. This is the case that
forces §4's placement to be stated as a per-axis odometer rather than an
interval of the flattened extent, and R12's per-axis clause to be a real
check rather than a formality. It is also the reason `gather_dimensions` is
list-valued rather than a single `i64`.

**Row 3 is the mirror relation and the one send-only case in the measured
set.** One destination share held by 28 of 32 cores, so four cores produce
and never consume: `C ⊊ P`. Both containments therefore occur within this
one op.

Two census facts belong here rather than as free-standing findings, both
descriptive.

**A three-axis concat exists in measurement**, so §4's placement rule must
be well-defined over three axes: list-valued attributes are a requirement of
a *named, measured* pattern, not a corner case. That is row 2 above.

**Idleness is the norm, and replicated sources do not occur.** Every source
region in all 51 files has exactly one holder — which is why §7.6 has no
measured broadcast, and why the producer-election escape hatch of §9.2 is
unforced. 16 files have fewer source regions than cores and all resolve to
single holders plus idle cores. Replication appears only on the destination
side, and every destination side that carries it classifies as `gather` —
including row 3's single region held by 28 cores with 4 idle. This is the
same 28-side set that §7.2's Step 1 identifies as the only sides where the
core map is set-valued rather than a function.

```mlir
// 4 tiles per group, 8 groups (32 tiles total).
#all_group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#group_consumer  = affine_set<(i)[g] : (i - 4*g == 0)>
#all_groups      = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

// All tiles contribute a partial slab.
%partial_future = ktdp.inter_tile_produce
    producer_tiles_per_group = #all_group_tiles
    -> !ktdp.tile_future<(tensor<128x1x3x64xf16>), groups = #all_groups>
{
  ^bb0(%gid: index):
    ktdp.yield_partial %partial_4d : tensor<128x1x3x64xf16>
}

// Gather along dim 2; one consumer per group (tile 4g) assembles the four
// 3-wide slabs. No combiner, no identity — placement is by within-group
// local index. gather_dimensions = [2] → 3 * 4 = 12; consumer gets <128x1x12x64>.
%assembled = ktdp.inter_tile_gather(%partial_future)
    consumer_tiles_per_group = #group_consumer,
    gather_dimensions               = [2]
    : !ktdp.tile_future<(tensor<128x1x3x64xf16>), groups = #all_groups> -> tensor<128x1x12x64xf16>
// The consumer holds the full assembled tensor — ownership via SSA result.
```

#### 7.4.1 Full IR — multi-group gather (128×8×12×64)

**Layout and partitioning.** `A` and `B` are `tensor<128x8x12x64xf16>` in global
memory. The four axes have distinct roles:

- Dim 0 (size 128): preserved through this op.
- Dim 1 (size 8): the **group axis** — 8 groups.
- Dim 2 (size 12): the **gather axis** — within each group, 4 tiles each own
  a 3-wide slab that gather concatenates back into the full 12.
- Dim 3 (size 64): vector / stick axis, preserved.

32 tiles, 8 groups of 4. `g = t / 4`, `l = t % 4`. Tile `(g, l)` reads
slice `[*, g, l*3 : l*3+3, *]` — shape `<128x1x3x64>`. Each tile's partial
is the summed slab `A + B` over its own columns (no reduction across tiles).
Gather along dim 2 places tile `(g, l)`'s slab at columns `[l*3 : l*3+3]` of
the assembled `<128x1x12x64>`, which one consumer per group (tile `4g`)
writes back to `E[*, g, *, *]`.

```mlir
#A_view_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 >= 0, -d1 + 7   >= 0,
     d2 >= 0, -d2 + 11  >= 0,
     d3 >= 0, -d3 + 63  >= 0)>

#AB_tile_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 == 0,
     d2 >= 0, -d2 + 2   >= 0,
     d3 >= 0, -d3 + 63  >= 0)>

// E access tile for the consumer: 128x1x12x64 anchored at [0, g, 0, 0].
#E_tile_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 == 0,
     d2 >= 0, -d2 + 11  >= 0,
     d3 >= 0, -d3 + 63  >= 0)>

#identity_4d = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

#group_tiles    = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#group_consumer = affine_set<(i)[g] : (i - 4*g == 0)>
#all_groups     = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

module {
  func.func @inter_tile_gather_multi_group() {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %col_slab = arith.constant 3 : index   // 12 / 4

    %A_start = arith.constant 1024     : index
    %B_start = arith.constant 12583936 : index
    %E_start = arith.constant 25166848 : index

    %A_view = ktdp.construct_memory_view %A_start, sizes: [128, 8, 12, 64],
        strides: [6144, 768, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<128x8x12x64xf16>
    %B_view = ktdp.construct_memory_view %B_start, sizes: [128, 8, 12, 64],
        strides: [6144, 768, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<128x8x12x64xf16>

    // Per-tile compute (function-scope SPMD).
    %t = ktdp.get_compute_tile_id : index
    %g = arith.divui %t, %c4 : index
    %l = arith.remui %t, %c4 : index
    %col_anchor = arith.muli %l, %col_slab : index

    %A_access = ktdp.construct_access_tile %A_view[%c0, %g, %col_anchor, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_4d
    } : memref<128x8x12x64xf16> -> !ktdp.access_tile<128x1x3x64xindex>
    %B_access = ktdp.construct_access_tile %B_view[%c0, %g, %col_anchor, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_4d
    } : memref<128x8x12x64xf16> -> !ktdp.access_tile<128x1x3x64xindex>

    %A_tile = ktdp.load %A_access
                : !ktdp.access_tile<128x1x3x64xindex> -> tensor<128x1x3x64xf16>
    %B_tile = ktdp.load %B_access
                : !ktdp.access_tile<128x1x3x64xindex> -> tensor<128x1x3x64xf16>

    // No reduction — the summed slab is this tile's partial; gather will
    // concatenate the four slabs along dim 2.
    %AB_init = tensor.empty() : tensor<128x1x3x64xf16>
    %partial_4d = linalg.add ins(%A_tile, %B_tile
                                 : tensor<128x1x3x64xf16>, tensor<128x1x3x64xf16>)
                             outs(%AB_init : tensor<128x1x3x64xf16>)
                             -> tensor<128x1x3x64xf16>

    // Produce: every tile contributes its 3-wide slab to the future.
    %partial_future = ktdp.inter_tile_produce
        producer_tiles_per_group = #group_tiles
        -> !ktdp.tile_future<(tensor<128x1x3x64xf16>), groups = #all_groups>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %partial_4d : tensor<128x1x3x64xf16>
    }

    // Gather dim 2: 4 producers x 3 = 12. One consumer (tile 4g) per group
    // assembles the full <128x1x12x64>. No combiner region, no identity.
    %assembled = ktdp.inter_tile_gather(%partial_future)
        consumer_tiles_per_group = #group_consumer,
        gather_dimensions               = [2]
        : !ktdp.tile_future<(tensor<128x1x3x64xf16>), groups = #all_groups>
          -> tensor<128x1x12x64xf16>

    // Post-gather: the consumer tile 4g writes its group's full slab to
    // E[*, g, *, *]. Ownership is explicit via the def-use chain of %assembled.
    %E_view = ktdp.construct_memory_view %E_start, sizes: [128, 8, 12, 64],
        strides: [6144, 768, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<128x8x12x64xf16>

    %E_access = ktdp.construct_access_tile %E_view[%c0, %g, %c0, %c0] {
        access_tile_set = #E_tile_set, access_tile_order = #identity_4d
    } : memref<128x8x12x64xf16> -> !ktdp.access_tile<128x1x12x64xindex>

    ktdp.store %assembled, %E_access
              : tensor<128x1x12x64xf16>, !ktdp.access_tile<128x1x12x64xindex>

    return
  }
}
```

### 7.5 Scatter  →  `inter_tile_produce` + `inter_tile_scatter`

**Measured evidence: 12 of the 51 relayouts**, all the same division.
`Mul_QC_1` is representative: tensor `mb:2 × out:64 × x:1 × y:512` (plus
two size-1 dims), `out` sticked at 64.

**§7.2 applied.**

| quantity | value | how |
|---|---|---|
| `Ns` | `{y:16}`, all other dims 1 | Step 1 |
| `Nd` | `{y:32}`, all other dims 1 | Step 1 |
| `#groups` | `gcd(16, 32) =` **16** | Step 2 — `y` alone; every other dim contributes `gcd(1,1) = 1` |
| `\|P(g)\|` | `16 / 16 =` **1** | Step 4 |
| `\|C(g)\|` | `32 / 16 =` **2** | Step 4 |
| `P(g)` | one core — the even cores `{0}`, `{2}`, … `{30}` | Step 3–4 — stride `y:2`, so `core = 2y` |
| `C(g)` | that core plus its odd neighbour: `{0,1}`, `{2,3}`, … | Step 5 — stride `y:1` |
| `D` | **not needed** | Step 6 — the group graph is `K(1,2)` |

Three facts to carry into an implementation.

**`|P(g)| == 1` falls out of Step 4**, not out of an assumption. It is R8
in its simplest form and it is why this op takes no dependency attribute at
all (§6.6) — with one producer, full-barrier and per-tile synchronization
are the same thing.

**Half the consumers never produce**, so `C ⊄ P`: R13 is `n` here by
measurement as well as by argument (§5, §6.6). Note this is a genuine
core-set fact and not a share-count artefact — every destination share in
these 12 files has exactly one holder, so `|C(g)| = 2` shares *and* 2
consumer tiles.

**`scatter_dimensions = [y]`, one axis**: `y` goes from a 32-wide piece per
core to a 16-wide one, `C = 2`, and `32 % 2 == 0` satisfies R9. All 12 are
single-axis, so no measured scatter exercises §4's multi-axis odometer.

```mlir
// 4 tiles per group, 8 groups (32 tiles total).
#group_producer  = affine_set<(i)[g] : (i - 4*g == 0)>
#all_group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#all_groups      = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

// One producer per group (tile 4g) holds the whole 128-row tensor.
%whole_future = ktdp.inter_tile_produce
    producer_tiles_per_group = #group_producer
    -> !ktdp.tile_future<(tensor<128x1x64xf16>), groups = #all_groups>
{
  ^bb0(%gid: index):
    ktdp.yield_partial %whole : tensor<128x1x64xf16>
}

// Scatter along dim 0; the four tiles per group each receive one 32-row
// chunk. No combiner, no identity — placement is by within-group local
// index. scatter_dimensions = [0] → 128 / 4 = 32; each consumer gets <32x1x64>.
%chunk = ktdp.inter_tile_scatter(%whole_future)
    consumer_tiles_per_group = #all_group_tiles,
    scatter_dimensions              = [0]
    : !ktdp.tile_future<(tensor<128x1x64xf16>), groups = #all_groups> -> tensor<32x1x64xf16>
// Each consumer holds its own 32-row slice — ownership via SSA result.
```

#### 7.5.1 Full IR — multi-group scatter (128×8×64)

**Layout and partitioning.** `A` and `B` are `tensor<128x8x64xf16>` in global
memory. The three axes have distinct roles:

- Dim 0 (size 128): the **scatter axis** — the producer's 128 rows are
  split into 4 chunks of 32, one per consumer tile.
- Dim 1 (size 8): the **group axis** — 8 groups.
- Dim 2 (size 64): vector / stick axis, preserved.

32 tiles, 8 groups of 4. `g = t / 4`, `l = t % 4`. Per group, the single
producer tile `4g` reads its group's whole slab `A[*, g, *]` and
`B[*, g, *]` — shape `<128x1x64>` each — and produces both as `N = 2`
roles. Scatter along dim 0 delivers chunk `[l*32 : l*32+32, *, *]` of each
role to the consumer with within-group local index `l`, which sums its two
chunks and writes the `<32x1x64>` result back to `E[l*32 : l*32+32, g, *]`.

**Why the loads live inside the produce region.** Unlike the other full-IR
examples, which hoist their `ktdp.load`s to function scope, this one keeps
them inside `ktdp.inter_tile_produce`. That is deliberate, and it follows
from single-producer cardinality (§2.2): only tile `4g` may read the
group's whole slab, so hoisting the loads would make every tile in the
group execute them. The other examples have every tile produce, so
function-scope loads are correct there.

**And why the sum does not.** §2.2 restricts the producer region to memory
ops, so the `linalg.add` cannot sit inside it beside the loads. It does not
need to: the op is variadic (§3.7), so the region yields the two slabs as
`N = 2` roles and each consumer adds the two `<32x1x64>` chunks it
receives. That is the general shape of the restriction — the loads stay
scoped because only the region can scope them, and the arithmetic moves to
where its operands are, doing `1/C` of the work per tile instead of all of
it on the producer.

```mlir
#A_view_set = affine_set<(d0, d1, d2) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 >= 0, -d1 + 7   >= 0,
     d2 >= 0, -d2 + 63  >= 0)>

// Producer partial: the whole 128-row slab of one group, anchored at [0, g, 0].
#whole_tile_set = affine_set<(d0, d1, d2) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 == 0,
     d2 >= 0, -d2 + 63  >= 0)>

// Consumer chunk: 32 rows, anchored at [l*32, g, 0].
#chunk_tile_set = affine_set<(d0, d1, d2) :
    (d0 >= 0, -d0 + 31 >= 0,
     d1 == 0,
     d2 >= 0, -d2 + 63 >= 0)>

#identity_3d = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

#group_producer  = affine_set<(i)[g] : (i - 4*g == 0)>
#all_group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#all_groups      = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

module {
  func.func @inter_tile_scatter_multi_group() {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %row_chunk = arith.constant 32 : index   // 128 / 4

    %A_start = arith.constant 1024    : index
    %B_start = arith.constant 1049600 : index
    %E_start = arith.constant 2098176 : index

    %A_view = ktdp.construct_memory_view %A_start, sizes: [128, 8, 64],
        strides: [512, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<128x8x64xf16>
    %B_view = ktdp.construct_memory_view %B_start, sizes: [128, 8, 64],
        strides: [512, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<128x8x64xf16>

    %t = ktdp.get_compute_tile_id : index
    %g = arith.divui %t, %c4 : index
    %l = arith.remui %t, %c4 : index

    // Produce: only the group's producer tile (4g) runs this region; it
    // reads its group's whole 128-row slab of A and of B. Memory ops only
    // (§2.2), so the two slabs are yielded as N = 2 roles rather than summed
    // here.
    %whole_future = ktdp.inter_tile_produce
        producer_tiles_per_group = #group_producer
        -> !ktdp.tile_future<(tensor<128x1x64xf16>, tensor<128x1x64xf16>), groups = #all_groups>
    {
      ^bb0(%gid: index):
        %A_access = ktdp.construct_access_tile %A_view[%c0, %gid, %c0] {
            access_tile_set = #whole_tile_set, access_tile_order = #identity_3d
        } : memref<128x8x64xf16> -> !ktdp.access_tile<128x1x64xindex>
        %B_access = ktdp.construct_access_tile %B_view[%c0, %gid, %c0] {
            access_tile_set = #whole_tile_set, access_tile_order = #identity_3d
        } : memref<128x8x64xf16> -> !ktdp.access_tile<128x1x64xindex>

        %A_tile = ktdp.load %A_access
                    : !ktdp.access_tile<128x1x64xindex> -> tensor<128x1x64xf16>
        %B_tile = ktdp.load %B_access
                    : !ktdp.access_tile<128x1x64xindex> -> tensor<128x1x64xf16>

        ktdp.yield_partial %A_tile, %B_tile
                           : tensor<128x1x64xf16>, tensor<128x1x64xf16>
    }

    // Scatter dim 0: 128 / 4 = 32. Each of the four consumer tiles per group
    // receives one 32-row chunk of each role. Both roles share the one
    // scatter_dimensions attribute (§3.7). No combiner region, no identity.
    %A_chunk, %B_chunk = ktdp.inter_tile_scatter(%whole_future)
        consumer_tiles_per_group = #all_group_tiles,
        scatter_dimensions              = [0]
        : !ktdp.tile_future<(tensor<128x1x64xf16>, tensor<128x1x64xf16>), groups = #all_groups>
          -> tensor<32x1x64xf16>, tensor<32x1x64xf16>

    // Post-scatter: consumer (g, l) sums its two chunks and writes them to
    // E[l*32 : l*32+32, g, *]. Ownership is explicit via the def-use chain,
    // and the add is now 32 rows per tile rather than 128 on the producer.
    %chunk_init = tensor.empty() : tensor<32x1x64xf16>
    %chunk = linalg.add ins(%A_chunk, %B_chunk
                            : tensor<32x1x64xf16>, tensor<32x1x64xf16>)
                        outs(%chunk_init : tensor<32x1x64xf16>)
                        -> tensor<32x1x64xf16>

    %row_anchor = arith.muli %l, %row_chunk : index

    %E_view = ktdp.construct_memory_view %E_start, sizes: [128, 8, 64],
        strides: [512, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<128x8x64xf16>

    %E_access = ktdp.construct_access_tile %E_view[%row_anchor, %g, %c0] {
        access_tile_set = #chunk_tile_set, access_tile_order = #identity_3d
    } : memref<128x8x64xf16> -> !ktdp.access_tile<32x1x64xindex>

    ktdp.store %chunk, %E_access
              : tensor<32x1x64xf16>, !ktdp.access_tile<32x1x64xindex>

    return
  }
}
```

### 7.6 Broadcast  →  `inter_tile_produce` + `inter_tile_consume`

**Measured evidence: NONE of the 51 relayouts.** There is no measured
example of this pattern, and the IR below is a construction rather than a
transcription.

**Why there is none, precisely.** Broadcast needs a *source* share held by
one core and delivered to several. In the measured set **every source
region has exactly one holder and every destination region that has several
holders is the destination of a `gather`** — replication only ever appears
on the destination side, never as a fan-out from one producer's own share
(§7.2, Step 1). So `|P(g)| == 1` with `|C(g)| == 1` and several holders is
never the shape a measured relayout takes.

**The pattern is nonetheless expressible in the artifact** — one input piece
on one core, one output region whose `PlacementInfo.memId` lists many cores,
which is exactly the length-4/28/32 `memId` shape the measured set does
contain on its destination side (§7.1). And it *is* measured elsewhere: the
broadcast row of §7's census comes from separate broadcast work (PR #4061),
not from these 51. That row is the one row in the census that is not one of
the 51, for exactly the reason above: **replicated sources do not occur in
the measured set** (§7.4), so a broadcast has nothing to be read off.

**§7.2 applied — to the construction below, not to a measurement.** One
group; `Ns = {}`, `Nd = {}` so Step 2 gives `#groups = 1`; `|P| = |C| = 1`
share; `P = {0}` and `C = {0,1,2,3}` are the construction's choice, there
being no cut axis for Step 3 to give a stride to; the group
graph is `K(1,1)` so `D` is not needed.

```mlir
// 4 tiles, 1 group: tile 0 loads W; all 4 tiles compute.
#tile_0          = affine_set<(i)[g] : (i - 4*g == 0)>
#all_group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#single_group    = affine_set<(g) : (g == 0)>

%W_future = ktdp.inter_tile_produce
    producer_tiles_per_group = #tile_0
    -> !ktdp.tile_future<(tensor<64x128xf16>), groups = #single_group>
{
  ^bb0(%gid: index):
    %W = ktdp.load ...
    ktdp.yield_partial %W : tensor<64x128xf16>
}

// Every consumer tile extracts its copy; no combiner → value passes through.
// Groups are inferred from the future's #single_group parameter.
%W_tile = ktdp.inter_tile_consume(%W_future)
    consumer_tiles_per_group = #all_group_tiles
    : !ktdp.tile_future<(tensor<64x128xf16>), groups = #single_group> -> tensor<64x128xf16>

// Post-delivery SPMD compute — owned by consumer_tiles_per_group.
// Ownership verified by traversing the def-use chain from %W_tile.
%A = ktdp.load ...
%C = linalg.matmul ins(%A, %W_tile ...) ...
ktdp.store %C, ...
```

### 7.7 Per-tile synchronization  →  `inter_tile_consume` with `producer_dependency_per_consumer`

**Measured evidence: NONE of the 51 relayouts.** Both examples below are
constructions — the dedicated-pair of §7.7.1 and the butterfly of §7.7.2 —
not transcriptions.

**These are the only patterns that need `D` at all.** §7.2's Step 6 gives
the reason: `producer_dependency_per_consumer` is required exactly when the
within-group overlap graph is not complete bipartite, and it is complete in
**every group of all 51** measured relayouts. So the entire `D` mechanism —
and with it R3–R7 (§5) — is exercised by nothing measured. That is worth
stating plainly rather than burying: `D` is not on the critical path for any
measured work, and an implementation may defer it without blocking any
measured pattern.

Two further facts about why no measurement reaches here. Routing needs a
group with several producers whose pairing to consumers is declared, and
every source region in the measured set has a single holder, so there is
never a transmitter to choose (§9.2). And the *mechanism* for whole-partial
permutation is expressible in the artifact — identical piece geometry on
both sides with different `memId` (§6.1) — so the gap is in the measurements,
not in the artifact.

#### 7.7.1 Per-tile pairing within a single group

Four tiles per group: tiles `4g` and `4g+1` are producers, tiles `4g+2`
and `4g+3` are consumers. Each consumer depends on its dedicated producer
(`4g+2` ← `4g`, `4g+3` ← `4g+1`), so the pairing is `p = c - 2` — a
constant relative offset that does not depend on the group index `g`.

**Dependency table** for group 0:

| group | producer | consumer |
|-------|----------|----------|
| 0     | 0        | 2        |
| 0     | 1        | 3        |

```mlir
// Producers: tiles 4g, 4g+1.  Consumers: tiles 4g+2, 4g+3.
#producer_tiles  = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 1 >= 0)>
#consumer_tiles  = affine_set<(i)[g] : (i - 4*g - 2 >= 0, -i + 4*g + 3 >= 0)>
#single_group    = affine_set<(g) : (g == 0)>

// The pairing p = c - 2 is group-independent, so g is not needed as a symbol.
#dep_per_consumer = affine_set<(p)[c] : (p - c + 2 == 0)>

%data_future = ktdp.inter_tile_produce
    producer_tiles_per_group = #producer_tiles
    -> !ktdp.tile_future<(tensor<64xf16>), groups = #single_group>
{
  ^bb0(%gid: index):
    %data = ktdp.load ...
    ktdp.yield_partial %data : tensor<64xf16>
}

// Each consumer unblocks independently as its assigned producer finishes.
%my_data = ktdp.inter_tile_consume(%data_future)
    consumer_tiles_per_group         = #consumer_tiles,
    producer_dependency_per_consumer = #dep_per_consumer
    : !ktdp.tile_future<(tensor<64xf16>), groups = #single_group> -> tensor<64xf16>
```

Without `producer_dependency_per_consumer`, both consumers stall until
both producers finish. With it, each consumer stalls only for its own
producer, halving the worst-case wait when the two producers finish at
different times.

#### 7.7.2 Butterfly mirror exchange across multiple groups — `CollectivePermute`

Eight groups of 4 tiles; all 4 tiles in each group both produce and
consume. Tile `c = 4g + l` waits only for its mirror partner
`p = 4g + (3 - l)`, equivalent to `p + c = 8g + 3`. This models a
butterfly-style partner exchange.

Both `c` and `g` are required: `c` identifies which specific consumer is
asking (different consumers within the group have different mirrors), and
`g` anchors the equation to the group (the target sum `8g + 3` is `3`,
`11`, `19`, ... for groups `0`, `1`, `2`, ..., so `g` cannot be
eliminated).

**Dependency table**, first two groups:

| group | producer | consumer |
|-------|----------|----------|
| 0     | 0        | 3        |
| 0     | 1        | 2        |
| 0     | 2        | 1        |
| 0     | 3        | 0        |
| 1     | 4        | 7        |
| 1     | 5        | 6        |
| 1     | 6        | 5        |
| 1     | 7        | 4        |
| …     | …        | …        |

```mlir
// 8 groups of 4 tiles; every tile is both producer and consumer.
#all_group_tiles  = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#all_groups       = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

// Tile c = 4g+l waits for mirror tile p = 4g+(3-l), i.e. p + c = 8g + 3.
// c is needed: different consumers have different mirrors within a group.
// g is needed: the sum p + c = 8g + 3 is a different value for each group.
#dep_per_consumer = affine_set<(p)[c, g] : (p + c - 8*g - 3 == 0)>

%data_future = ktdp.inter_tile_produce
    producer_tiles_per_group = #all_group_tiles
    -> !ktdp.tile_future<(tensor<64xf16>), groups = #all_groups>
{
  ^bb0(%gid: index):
    %data = ktdp.load ...
    ktdp.yield_partial %data : tensor<64xf16>
}

// Each tile unblocks as soon as its single mirror partner has yielded,
// without waiting for the other two tiles in the group.
%partner_data = ktdp.inter_tile_consume(%data_future)
    consumer_tiles_per_group         = #all_group_tiles,
    producer_dependency_per_consumer = #dep_per_consumer
    : !ktdp.tile_future<(tensor<64xf16>), groups = #all_groups> -> tensor<64xf16>
```

### 7.8 Reduce  →  `inter_tile_produce` + `inter_tile_reduce`

**Measured evidence: NONE of the 51 relayouts.** There is no measured
example of this pattern, and the IR below is a construction rather than a
transcription.

**Why there is none, precisely.** A relayout moves ownership of bytes and
does nothing to them: the artifact of §7.1 names no operation to apply and
carries no combiner field of any kind, and no output byte in the measured
set has more than one producer (§7's preamble). A `fold` therefore has
nothing to become on this surface, and the absence is structural rather
than a gap in the sample — §7.2's Rules 1–5 classify a pair of ownership
tables, and `reduce` is not a statement about ownership.

The consequence is about evidence, not expressiveness. Everything §6.2 and
§5 say about `reduce` — R11's identity typing, R13's `C ⊆ P`, R14's mode
gate — rests on argument and on what the verifier implements today (§8),
not on a measurement.

**§7.2 applied — to the construction below, not to a measurement.** Eight
groups of four tiles; `producer_tiles_per_group` is §2.1's worked example at
group size 4, the group graph is complete bipartite so `D` is not needed
(Step 6), and the group is confined in §3.2's sense — only the four tiles of
a group exchange partials.

```mlir
// 4 tiles per group, 8 groups (32 tiles total).
#all_group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#all_groups      = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

// All tiles contribute a partial; future carries all partials.
%partial_future = ktdp.inter_tile_produce
    producer_tiles_per_group = #all_group_tiles
    -> !ktdp.tile_future<(tensor<1x64xf16>), groups = #all_groups>
{
  ^bb0(%gid: index):
    ktdp.yield_partial %partial_2d : tensor<1x64xf16>
}

// Reduce all partials; every consumer tile receives the same reduced value.
%reduced = ktdp.inter_tile_reduce(%partial_future)
    consumer_tiles_per_group = #all_group_tiles,
    identity(%add_id : tensor<1x64xf16>)
    : !ktdp.tile_future<(tensor<1x64xf16>), groups = #all_groups> -> tensor<1x64xf16>
{
  ^bb0(%lhs: tensor<1x64xf16>, %rhs: tensor<1x64xf16>):
    %sum = linalg.add ins(%lhs, %rhs ...) ...
    ktdp.yield_reduced %sum : tensor<1x64xf16>
}
```

#### 7.8.1 Full IR — single-group reduce (96×64)

**Layout and partitioning.** `A` and `B` are `tensor<96x64xf16>` in global memory.
The kernel computes the column-wise sum of `A + B`, producing a
`tensor<1x64xf16>` (the leading unit dim is the within-group tile axis,
preserved by the op per §4).

The 32 compute tiles form a single group. Tile `t` owns rows
`t*3 .. t*3+2` of `A` and `B` — a 3×64 slab each. The per-tile
contribution is the row-reduced partial expanded to `tensor<1x64xf16>`,
whose leading unit dimension is the within-group tile axis. The op preserves
it (§4), so every tile holds the same `%reduced : tensor<1x64xf16>`
(all-reduce case: consumer set = producer set).

```mlir
#A_view_set  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#AB_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 +  2 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#E_view_set  = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
#E_tile_set  = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
#identity_2d = affine_map<(d0, d1) -> (d0, d1)>

// One group containing all 32 tiles.
#group_tiles = affine_set<(i)[g] : (i - 32*g >= 0, -i + 32*(g+1) - 1 >= 0)>
#all_groups  = affine_set<(g) : (g == 0)>

module {
  func.func @inter_tile_reduce_single_group() {
    %c0 = arith.constant 0 : index
    %tile_size = arith.constant 3 : index
    %A_start = arith.constant 1024  : index
    %B_start = arith.constant 12288 : index
    %E_start = arith.constant 22528 : index

    %A_view = ktdp.construct_memory_view %A_start, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<96x64xf16>
    %B_view = ktdp.construct_memory_view %B_start, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<96x64xf16>

    // Identity: tensor<1x64xf16> of zeros — matches partial type T_p.
    %c_zero   = arith.constant 0.0 : f16
    %add_init = tensor.empty() : tensor<1x64xf16>
    %add_id   = linalg.fill ins(%c_zero : f16) outs(%add_init : tensor<1x64xf16>)
                  -> tensor<1x64xf16>

    // Per-tile compute (function-scope SPMD).
    %t = ktdp.get_compute_tile_id : index
    %start_row = arith.muli %t, %tile_size : index

    %A_access = ktdp.construct_access_tile %A_view[%start_row, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_2d
    } : memref<96x64xf16> -> !ktdp.access_tile<3x64xindex>
    %B_access = ktdp.construct_access_tile %B_view[%start_row, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_2d
    } : memref<96x64xf16> -> !ktdp.access_tile<3x64xindex>

    %A_tile = ktdp.load %A_access : !ktdp.access_tile<3x64xindex> -> tensor<3x64xf16>
    %B_tile = ktdp.load %B_access : !ktdp.access_tile<3x64xindex> -> tensor<3x64xf16>

    %AB_init = tensor.empty() : tensor<3x64xf16>
    %AB_sum  = linalg.add ins(%A_tile, %B_tile : tensor<3x64xf16>, tensor<3x64xf16>)
                          outs(%AB_init : tensor<3x64xf16>) -> tensor<3x64xf16>

    %red_init   = tensor.empty() : tensor<64xf16>
    %red_filled = linalg.fill ins(%c_zero : f16) outs(%red_init : tensor<64xf16>)
                    -> tensor<64xf16>
    %partial_1d = linalg.reduce { arith.addf }
                    ins(%AB_sum : tensor<3x64xf16>)
                    outs(%red_filled : tensor<64xf16>)
                    dimensions = [0]
    %partial_2d = tensor.expand_shape %partial_1d [[0, 1]] output_shape [1, 64]
                    : tensor<64xf16> into tensor<1x64xf16>

    // Produce: every tile contributes its partial_2d to the future.
    %partial_future = ktdp.inter_tile_produce
        producer_tiles_per_group = #group_tiles
        -> !ktdp.tile_future<(tensor<1x64xf16>), groups = #all_groups>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %partial_2d : tensor<1x64xf16>
    }

    // Reduce: unit dim 0 is the within-group tile axis; the op preserves it.
    // Every tile holds the same %reduced : tensor<1x64xf16> (all-reduce case).
    %reduced = ktdp.inter_tile_reduce(%partial_future)
        consumer_tiles_per_group = #group_tiles,
        identity(%add_id : tensor<1x64xf16>)
        : !ktdp.tile_future<(tensor<1x64xf16>), groups = #all_groups> -> tensor<1x64xf16>
    {
      ^bb0(%lhs: tensor<1x64xf16>, %rhs: tensor<1x64xf16>):
        %init = tensor.empty() : tensor<1x64xf16>
        %sum  = linalg.add ins(%lhs, %rhs : tensor<1x64xf16>, tensor<1x64xf16>)
                           outs(%init : tensor<1x64xf16>) -> tensor<1x64xf16>
        ktdp.yield_reduced %sum : tensor<1x64xf16>
    }

    // Post-reduction: every tile redundantly writes the same value.
    // No expand_shape needed — the result already carries the unit dim.

    %E_view = ktdp.construct_memory_view %E_start, sizes: [1, 64], strides: [64, 1] {
        coordinate_set = #E_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<1x64xf16>
    %E_access = ktdp.construct_access_tile %E_view[%c0, %c0] {
        access_tile_set = #E_tile_set, access_tile_order = #identity_2d
    } : memref<1x64xf16> -> !ktdp.access_tile<1x64xindex>

    ktdp.store %reduced, %E_access
              : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>

    return
  }
}
```

#### 7.8.2 Full IR — multi-group reduce (128×8×12×64)

**Layout and partitioning.** `A` and `B` are `tensor<128x8x12x64xf16>` in
global memory. The four axes have distinct roles:

- Dim 0 (size 128): preserved through this op.
- Dim 1 (size 8): the **group axis** — 8 groups.
- Dim 2 (size 12): the **reduction axis** — within each group, 4 tiles
  cooperate over this axis.
- Dim 3 (size 64): vector / stick axis, preserved.

There are 32 compute tiles forming 8 groups of 4. For tile `t`,
`g = t / 4` and `l = t % 4`. Tile `(g, l)` reads slice
`[*, g, l*3 : l*3+3, *]` of `A` and `B` — shape `<128x1x3x64>` each.

The partial is `<128x1x1x64>`: dim 1 is the group axis and dim 2 the
within-group tile axis, both preserved, so the result is `<128x1x1x64>`
too (§4). All four tiles in a group hold identical values; different groups
hold different values.

```mlir
#A_view_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 >= 0, -d1 + 7   >= 0,
     d2 >= 0, -d2 + 11  >= 0,
     d3 >= 0, -d3 + 63  >= 0)>

#AB_tile_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 == 0,
     d2 >= 0, -d2 + 2   >= 0,
     d3 >= 0, -d3 + 63  >= 0)>

#E_view_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 >= 0, -d1 + 7   >= 0,
     d2 >= 0, -d2 + 3   >= 0,
     d3 >= 0, -d3 + 63  >= 0)>

#E_tile_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 == 0,
     d2 == 0,
     d3 >= 0, -d3 + 63 >= 0)>

#identity_4d = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

#group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#all_groups  = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

module {
  func.func @inter_tile_reduce_multi_group() {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %red_slab = arith.constant 3 : index   // 12 / 4

    %A_start = arith.constant 1024     : index
    %B_start = arith.constant 12583936 : index
    %E_start = arith.constant 25166848 : index

    %A_view = ktdp.construct_memory_view %A_start, sizes: [128, 8, 12, 64],
        strides: [6144, 768, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<128x8x12x64xf16>
    %B_view = ktdp.construct_memory_view %B_start, sizes: [128, 8, 12, 64],
        strides: [6144, 768, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<128x8x12x64xf16>

    // Identity: tensor<128x1x1x64xf16> of zeros — matches partial type T_p.
    %c_zero  = arith.constant 0.0 : f16
    %id_init = tensor.empty() : tensor<128x1x1x64xf16>
    %add_id  = linalg.fill ins(%c_zero : f16) outs(%id_init : tensor<128x1x1x64xf16>)
                 -> tensor<128x1x1x64xf16>

    // Per-tile compute (function-scope SPMD).
    %t = ktdp.get_compute_tile_id : index
    %g = arith.divui %t, %c4 : index
    %l = arith.remui %t, %c4 : index
    %red_anchor = arith.muli %l, %red_slab : index

    %A_access = ktdp.construct_access_tile %A_view[%c0, %g, %red_anchor, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_4d
    } : memref<128x8x12x64xf16> -> !ktdp.access_tile<128x1x3x64xindex>
    %B_access = ktdp.construct_access_tile %B_view[%c0, %g, %red_anchor, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_4d
    } : memref<128x8x12x64xf16> -> !ktdp.access_tile<128x1x3x64xindex>

    %A_tile = ktdp.load %A_access
                : !ktdp.access_tile<128x1x3x64xindex> -> tensor<128x1x3x64xf16>
    %B_tile = ktdp.load %B_access
                : !ktdp.access_tile<128x1x3x64xindex> -> tensor<128x1x3x64xf16>

    %AB_init = tensor.empty() : tensor<128x1x3x64xf16>
    %AB_sum  = linalg.add ins(%A_tile, %B_tile
                              : tensor<128x1x3x64xf16>, tensor<128x1x3x64xf16>)
                          outs(%AB_init : tensor<128x1x3x64xf16>)
                          -> tensor<128x1x3x64xf16>

    %red_init   = tensor.empty() : tensor<128x1x64xf16>
    %red_filled = linalg.fill ins(%c_zero : f16)
                              outs(%red_init : tensor<128x1x64xf16>)
                              -> tensor<128x1x64xf16>
    %partial_3d = linalg.reduce { arith.addf }
                    ins(%AB_sum : tensor<128x1x3x64xf16>)
                    outs(%red_filled : tensor<128x1x64xf16>)
                    dimensions = [2]

    %partial_4d = tensor.expand_shape %partial_3d [[0], [1], [2, 3]]
                    output_shape [128, 1, 1, 64]
                    : tensor<128x1x64xf16> into tensor<128x1x1x64xf16>

    // Produce: every tile contributes its partial_4d to the future.
    %partial_future = ktdp.inter_tile_produce
        producer_tiles_per_group = #group_tiles
        -> !ktdp.tile_future<(tensor<128x1x1x64xf16>), groups = #all_groups>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %partial_4d : tensor<128x1x1x64xf16>
    }

    // Multi-group reduce: no rank reduction — dims 1 and 2 both preserved.
    // Each tile gets its group's <128x1x1x64>.
    %my_group_result = ktdp.inter_tile_reduce(%partial_future)
        consumer_tiles_per_group = #group_tiles,
        identity(%add_id : tensor<128x1x1x64xf16>)
        : !ktdp.tile_future<(tensor<128x1x1x64xf16>), groups = #all_groups>
          -> tensor<128x1x1x64xf16>
    {
      ^bb0(%lhs: tensor<128x1x1x64xf16>, %rhs: tensor<128x1x1x64xf16>):
        %init = tensor.empty() : tensor<128x1x1x64xf16>
        %sum  = linalg.add ins(%lhs, %rhs
                               : tensor<128x1x1x64xf16>, tensor<128x1x1x64xf16>)
                           outs(%init : tensor<128x1x1x64xf16>)
                           -> tensor<128x1x1x64xf16>
        ktdp.yield_reduced %sum : tensor<128x1x1x64xf16>
    }

    // Post-reduction: each tile writes its group's result to slice [*, g, l, *].
    // No expand_shape needed — the result already carries both unit dims.

    %E_view = ktdp.construct_memory_view %E_start, sizes: [128, 8, 4, 64],
        strides: [2048, 256, 64, 1] {
        coordinate_set = #E_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<128x8x4x64xf16>

    %E_access = ktdp.construct_access_tile %E_view[%c0, %g, %l, %c0] {
        access_tile_set = #E_tile_set, access_tile_order = #identity_4d
    } : memref<128x8x4x64xf16> -> !ktdp.access_tile<128x1x1x64xindex>

    ktdp.store %my_group_result, %E_access
              : tensor<128x1x1x64xf16>, !ktdp.access_tile<128x1x1x64xindex>

    return
  }
}
```

### 7.9 Reduce-scatter  →  `inter_tile_produce` + `inter_tile_reduce_scatter`

**Measured evidence: NONE. This is the one op in §6 with no measured path
at all** — unmeasured here, and unmeasured anywhere else this document can
appeal to. No relayout reaches it for the reason §7.8 gives: a relayout does
not combine, and the artifact has no compute in it (§7.1, §7's preamble). So
the IR below is a construction, and §7.2's rules have nothing to say about
it.

Every consequence of that gap is flagged where it arises: R13's cell stays
open (§5), §9.1 keeps the question, and R11's identity retargeting (§9.3)
is untested. Of the five unbuilt ops, this is the only one no measurement
argues for.

```mlir
// 4 tiles per group, 8 groups (32 tiles total).
#all_group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#all_groups      = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

// All tiles contribute a partial.
%partial_future = ktdp.inter_tile_produce
    producer_tiles_per_group = #all_group_tiles
    -> !ktdp.tile_future<(tensor<128x1x1x64xf16>), groups = #all_groups>
{
  ^bb0(%gid: index):
    ktdp.yield_partial %partial_4d : tensor<128x1x1x64xf16>
}

// Reduce and scatter; each tile receives its own slice along dim 0.
// scatter_dimensions = [0] → 128-row axis split across 4 tiles; each gets
// <32x1x1x64> (rank preserved, §4).
%my_chunk = ktdp.inter_tile_reduce_scatter(%partial_future)
    consumer_tiles_per_group = #all_group_tiles,
    scatter_dimensions              = [0],
    identity(%add_id : tensor<128x1x1x64xf16>)
    : !ktdp.tile_future<(tensor<128x1x1x64xf16>), groups = #all_groups>
      -> tensor<32x1x1x64xf16>
{
  ^bb0(%lhs: tensor<128x1x1x64xf16>, %rhs: tensor<128x1x1x64xf16>):
    %sum = linalg.add ins(%lhs, %rhs ...) ...
    ktdp.yield_reduced %sum : tensor<128x1x1x64xf16>
}
// Each tile holds a different slice — ownership explicit via SSA result.
```

#### 7.9.1 Full IR — multi-group reduce-scatter (128×8×12×64)

**Layout and partitioning.** `A` and `B` are `tensor<128x8x12x64xf16>`
in global memory. The four axes have distinct roles:

- Dim 0 (size 128): the **scatter axis** — within each group, this axis
  is split across that group's 4 tiles.
- Dim 1 (size 8): the **group axis** — 8 groups.
- Dim 2 (size 12): the **reduction axis** — within each group, 4 tiles
  cooperate over this axis.
- Dim 3 (size 64): vector / stick axis, preserved.

32 tiles, 8 groups of 4. `g = t / 4`, `l = t % 4`. Tile `(g, l)` reads
slice `[*, g, l*3 : l*3+3, *]` — shape `<128x1x3x64>`. The per-tile
pipeline through to `%partial_4d` (shape `<128x1x1x64>`) is identical
to §7.8.2.

The op reduces across the group and scatters dim 0 (128 / 4 = 32 rows per
tile), preserving rank (§4). Tile `(g, l)` ends up with rows
`[l*32 : (l+1)*32]` of group `g`'s reduced `<128x1x1x64>`.

```mlir
#A_view_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 >= 0, -d1 + 7   >= 0,
     d2 >= 0, -d2 + 11  >= 0,
     d3 >= 0, -d3 + 63  >= 0)>

#AB_tile_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 == 0,
     d2 >= 0, -d2 + 2   >= 0,
     d3 >= 0, -d3 + 63  >= 0)>

// E view (post-scatter output): 128x8x64.
#E_view_set = affine_set<(d0, d1, d2) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 >= 0, -d1 + 7   >= 0,
     d2 >= 0, -d2 + 63  >= 0)>

// E access tile per writer: 32x1x64 in E's 3-D memref, anchored at [l*32, g, 0].
#E_tile_set = affine_set<(d0, d1, d2) :
    (d0 >= 0, -d0 + 31 >= 0,
     d1 == 0,
     d2 >= 0, -d2 + 63 >= 0)>

#identity_4d = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#identity_3d = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

#group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#all_groups  = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

module {
  func.func @inter_tile_reduce_scatter_multi_group() {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %red_slab      = arith.constant 3  : index   // 12 / 4
    %scatter_chunk = arith.constant 32 : index   // 128 / 4

    %A_start = arith.constant 1024     : index
    %B_start = arith.constant 12583936 : index
    %E_start = arith.constant 25166848 : index

    %A_view = ktdp.construct_memory_view %A_start, sizes: [128, 8, 12, 64],
        strides: [6144, 768, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<128x8x12x64xf16>
    %B_view = ktdp.construct_memory_view %B_start, sizes: [128, 8, 12, 64],
        strides: [6144, 768, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<128x8x12x64xf16>

    // Identity: tensor<128x1x1x64xf16> of zeros — matches partial type T_p.
    %c_zero  = arith.constant 0.0 : f16
    %id_init = tensor.empty() : tensor<128x1x1x64xf16>
    %add_id  = linalg.fill ins(%c_zero : f16) outs(%id_init : tensor<128x1x1x64xf16>)
                 -> tensor<128x1x1x64xf16>

    // Per-tile compute (function-scope SPMD).
    %t = ktdp.get_compute_tile_id : index
    %g = arith.divui %t, %c4 : index
    %l = arith.remui %t, %c4 : index
    %red_anchor = arith.muli %l, %red_slab : index

    %A_access = ktdp.construct_access_tile %A_view[%c0, %g, %red_anchor, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_4d
    } : memref<128x8x12x64xf16> -> !ktdp.access_tile<128x1x3x64xindex>
    %B_access = ktdp.construct_access_tile %B_view[%c0, %g, %red_anchor, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_4d
    } : memref<128x8x12x64xf16> -> !ktdp.access_tile<128x1x3x64xindex>

    %A_tile = ktdp.load %A_access
                : !ktdp.access_tile<128x1x3x64xindex> -> tensor<128x1x3x64xf16>
    %B_tile = ktdp.load %B_access
                : !ktdp.access_tile<128x1x3x64xindex> -> tensor<128x1x3x64xf16>

    %AB_init = tensor.empty() : tensor<128x1x3x64xf16>
    %AB_sum  = linalg.add ins(%A_tile, %B_tile
                              : tensor<128x1x3x64xf16>, tensor<128x1x3x64xf16>)
                          outs(%AB_init : tensor<128x1x3x64xf16>)
                          -> tensor<128x1x3x64xf16>

    %red_init   = tensor.empty() : tensor<128x1x64xf16>
    %red_filled = linalg.fill ins(%c_zero : f16)
                              outs(%red_init : tensor<128x1x64xf16>)
                              -> tensor<128x1x64xf16>
    %partial_3d = linalg.reduce { arith.addf }
                    ins(%AB_sum : tensor<128x1x3x64xf16>)
                    outs(%red_filled : tensor<128x1x64xf16>)
                    dimensions = [2]

    %partial_4d = tensor.expand_shape %partial_3d [[0], [1], [2, 3]]
                    output_shape [128, 1, 1, 64]
                    : tensor<128x1x64xf16> into tensor<128x1x1x64xf16>

    // Produce: every tile contributes its partial_4d to the future.
    %partial_future = ktdp.inter_tile_produce
        producer_tiles_per_group = #group_tiles
        -> !ktdp.tile_future<(tensor<128x1x1x64xf16>), groups = #all_groups>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %partial_4d : tensor<128x1x1x64xf16>
    }

    // Reduce across the group, then scatter dim 0 (chunk = 32).
    // No rank reduction: dims 1 and 2 preserved. Each tile receives <32x1x1x64>.
    %my_chunk = ktdp.inter_tile_reduce_scatter(%partial_future)
        consumer_tiles_per_group = #group_tiles,
        scatter_dimensions              = [0],
        identity(%add_id : tensor<128x1x1x64xf16>)
        : !ktdp.tile_future<(tensor<128x1x1x64xf16>), groups = #all_groups>
          -> tensor<32x1x1x64xf16>
    {
      ^bb0(%lhs: tensor<128x1x1x64xf16>, %rhs: tensor<128x1x1x64xf16>):
        %init = tensor.empty() : tensor<128x1x1x64xf16>
        %sum  = linalg.add ins(%lhs, %rhs
                               : tensor<128x1x1x64xf16>, tensor<128x1x1x64xf16>)
                           outs(%init : tensor<128x1x1x64xf16>)
                           -> tensor<128x1x1x64xf16>
        ktdp.yield_reduced %sum : tensor<128x1x1x64xf16>
    }

    // Post-scatter: tile (g, l) writes rows [l*32 : l*32+32] of group g's result.
    %my_row_anchor = arith.muli %l, %scatter_chunk : index

    %E_view = ktdp.construct_memory_view %E_start, sizes: [128, 8, 64],
        strides: [512, 64, 1] {
        coordinate_set = #E_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<128x8x64xf16>

    %E_access = ktdp.construct_access_tile %E_view[%my_row_anchor, %g, %c0] {
        access_tile_set = #E_tile_set, access_tile_order = #identity_3d
    } : memref<128x8x64xf16> -> !ktdp.access_tile<32x1x64xindex>

    // Rank reduction now lives in ordinary code, not the op: collapse the
    // within-group tile axis to match E's 3-D layout.
    %my_chunk_3d = tensor.collapse_shape %my_chunk [[0], [1, 2], [3]]
                     : tensor<32x1x1x64xf16> into tensor<32x1x64xf16>

    ktdp.store %my_chunk_3d, %E_access
              : tensor<32x1x64xf16>, !ktdp.access_tile<32x1x64xindex>

    return
  }
}
```

### 7.10 Selection, not a delivery — `Stcdp_QC_38`

**Measured evidence: 1 of the 51**, and it is the guard case rather than a
pattern. It is worth its own subsection because it is the one file on which
§7.2's Step 4 gives the wrong answer, and knowing *why* is what keeps an
implementer from generalizing them past their precondition.

The tensor is `mb:512 × out:4096 (× y:1)`, `out` sticked at 64. The source
divides it 8 ways on `mb` and 4 ways on `out`. The destination keeps **one**
`mb` index — the last, `mb[511]` — and spreads that single row over all 32
cores, one 128-element run of `out` each.

| | slice counts | core 0 | core 28 | per-core box |
|---|---|---|---|---|
| src | `{mb:8, out:4}` | `mb[0:64] × out[0:1024]` | `mb[256:320] × out[3072:4096]` | `64 × 1024` |
| dst | `{mb:1, out:32}` | `mb[511:512] × out[0:128]` | `mb[511:512] × out[3584:3712]` | `1 × 128` |

**Where the counts come from matters here.** Read off the destination
table, `mb` carries one distinct slice — every core names `mb[511:512]` —
so `Nd(mb) = 1` against `Ns(mb) = 8`, and `mb` is coarsened; `out` goes the
other way, 4 against 32, so it is refined. Divide instead, extent 512 by
the slice's extent 1, and `Nd(mb) = 512`: `mb` would come out *refined*, on
the strength of 511 pieces the table never mentions. Only the first reading
is a fact about the tables (§7.2, Step 1).

**Why Step 7's guard fires.** Both of Step 7's *count* clauses pass:
`prod(Ns) = 8 × 4 = 32` equals the source region count and
`prod(Nd) = 1 × 32 = 32` equals the destination's. **Coverage** is what
fails — the 32 distinct destination regions hold `32 × 128 = 4096` elements
against the tensor's `512 × 4096 = 2097152`, a 512th of it. The coverage
clause is the only clause of Step 7 that reads slice *sizes* and the tensor
shape rather than slice counts, which is why the count clauses cannot
replace it. This is the one file of the 51 that is a **selection, not a
partition**, and it is the whole reason the validity guard exists.

**And why §7.2's steps must not be run on it.** Step 2 gives
`gcd(8,1) · gcd(4,32) = 1 · 4 = 4`, which does match the number of
components the overlap graph actually has. Step 4 gives
`|P(g)| = 32/4 = 8` and `|C(g)| = 32/4 = 8` — and that is **wrong**: the
measured components are `K(1, 8)`, one source region feeding eight
destination regions. The reason is that **28 of the 32 source regions
participate in no overlap at all**; they are isolated in the graph, so
`prod(Ns)` overcounts the producers by a factor of 8. Steps 2 and 4 divide
region totals by the group count, which is only sound when every region
participates — exactly what the guard row checks. This is the sole file in
the measured set with isolated regions, and the sole file on which Step 4
fails.

**The guard row is a KTIR modelling choice, not a property of the
artifact.** Nothing in the artifact of §7.1 requires either side's pieces
to cover the tensor, and this file is the proof: it is a legal, measured
`STCDPOpLx`. What the guard row encodes is this document's position that
**a selection is not a delivery**. It needs a select *before* one,
expressed as the access tile the partial is loaded through — the chain
below. The distinction matters because a reader who takes the row for a fact
about the artifact, then observes a selection being emitted, will "fix" the
rule and lose the modelling.

#### How a bounded extent is expressed

This file's destination owns `mb[511:512]`, and **no delivery op can say
that.** The tile sets name which *tiles* participate, the dimension
attributes name which *axes* split or concatenate, and §4's type rules are
extent arithmetic with no base coordinate: none of the six ops carries an
offset.

A bounded extent is therefore a property of `T_p`, and `T_p` gets it from the
access tile the partial was loaded through. The chain is the same whatever
the bound:

```mlir
// The extent is the access tile's shape and the offset is its anchor. Both are
// arguments here, and neither appears again downstream.
%access  = ktdp.construct_access_tile %view[<anchor-indices>] {
    access_tile_set = <affine-set>, access_tile_order = <affine-map>
} : memref<...> -> !ktdp.access_tile<...xindex>
%partial = ktdp.load %access : !ktdp.access_tile<...xindex> -> T_p

// From here the bound is invisible: the delivery op sees a T_p and an axis
// set, never a coordinate. Selecting a different sub-tensor changes only T_p.
%future  = ktdp.inter_tile_produce producer_tiles_per_group = <affine-set>
    -> !ktdp.tile_future<(T_p), groups = #groups>
{ ^bb0(%gid: index): ktdp.yield_partial %partial : T_p }
%result  = ktdp.inter_tile_consume(%future)
    consumer_tiles_per_group = <affine-set>
    : !ktdp.tile_future<(T_p), groups = #groups> -> T_p
```

`ktdp.construct_access_tile` fixes the coordinates, `ktdp.load` yields the
value, `ktdp.yield_partial` only names it, and the delivery carries whatever
the partial turned out to be. **That is why a selection is not a delivery**,
and why the coverage clause measures the value *delivered*: read as a
delivery of the whole tensor the pair above covers 1/512 and trips the guard,
while the sub-tensor it should have been is covered exactly. §2.2 works the
same chain with group-dependent anchors.

Whether a delivery is then needed at all is the ordinary classification
question, and here it is not: each core owns a different stick-run of the
selected row, so nothing crosses cores and Step 7's row 1 gives
`no op needed`. Had they all needed the same run it would be a broadcast, and
the load would move inside the producer region, since one producer per group
(R8) means the non-producing tiles must not run it (§2.2, and §7.5.1 for the
same reason on `scatter`).

---

## 8. Implementation status

Where the rules of §5 stand in the verifier today. Non-normative: this
section records the current state, not an obligation.

The legality pass (`lib/Conversion/ConvertToKTIR/KTIRCheckLegality.cpp`,
182 lines) currently walks only `InterTileProduceOp` and
`InterTileReduceOp`:

**Implemented-rule table.** One row per check that exists today.

| Rule | Op | Check | Location |
|---|---|---|---|
| R2 | `inter_tile_produce` | `future.hasOneUse()` | `KTIRCheckLegality.cpp:80–85` |
| R13 | `inter_tile_reduce` | `C ⊆ P` per group | `KTIRCheckLegality.cpp:107–117` |
| R14 | `inter_tile_reduce` | `C == P` or `\|C\| == 1` | `KTIRCheckLegality.cpp:119–128` |
| R3 | `inter_tile_reduce` | declared dep `p ∈ P(g)` | `KTIRCheckLegality.cpp:151–160` |
| R4 | `inter_tile_reduce` | every `p` covered by some dep | `KTIRCheckLegality.cpp:163–174` |

**Not yet implemented:** R1, R5, R6, R7, R8, R9, R10, R11, R12, and
R3/R4/R13/R14 for every op other than `reduce`. Of the ops §7's census shows the
measured set requires — `gather`, `all_to_all`, `scatter`, `consume` — none
has a verifier today, and R9/R12 in particular are stated over multi-axis
axis *lists* (§4), so implementing them means validating a list and its
per-axis factors, not a single index. R5 and R7 are enforced in the
`torch-spyre` SDSC planner but are absent from the KTIR verifier entirely —
the gap exists at both the spec and the implementation level.

**Priority follows §7.2's Step 6.** R3–R7 all constrain
`producer_dependency_per_consumer`, which **no measured relayout declares**
(§7.2, §5). So the five unimplemented dependency-set rules block nothing
measured, while R9 and R12 — which every measured `gather`, `all_to_all`
and `scatter` needs — block everything.

**Dependency-set arity.** §3.4's one-symbol spelling `(p)[c]` is accepted as
of `KTDPInterTileHelpers.cpp:69–100` and `KTIRCheckLegality.cpp:135–142`.
Before that, `depTilesOf` always bound two symbols and the pass rejected any
set whose symbol count was not exactly 2, so the group-independent form this
document documents — and uses in §7.7.1 — was unusable in practice. The symbol
count now selects how many values are bound, and 3-or-more is diagnosed.

**Two asymmetries the verification matrix (§5) forces into the open.**

1. R8 is a verifier obligation for both `consume` and `scatter`, but it
   bites differently: `scatter` takes no dependency attribute, so one
   producer per group is the whole rule, whereas `consume` admits a
   multi-producer group whenever the attribute pairs each consumer tile with
   exactly one producer (§5). Neither is implemented yet.
2. R13 and R14 are implemented for `reduce` only, and R13 is the
   implementation of open question §9.1 (must a consumer also be a
   producer?) for that one op. The rule's answers now differ by op: **yes**
   for `reduce` (enforced today), **no** for `scatter` (§6.6), **no** for
   `gather` and `all_to_all` (falsified by measurement, §5), and still
   undecided for `reduce_scatter` — the single remaining `?` cell. So an
   implementer extending the check must gate it per op rather than
   generalizing the `reduce` path. R14's mode gate is likewise a current
   implementation restriction, not a design conclusion.

---

## 9. Open questions

### 9.1 Must a consumer also be a producer? — now only for the two `fold` ops

**What turns on it:** whether the verifier rejects a delivery op whose consumer
set is not contained in its producer set. That check exists and runs today — R13
for `reduce` (`KTIRCheckLegality.cpp:107–117`) — and the answer changes which
programs are legal.

**Closed for the copy-only ops, by measurement.** `gather`, `all_to_all` and
`scatter` are resolved **no**: 16 of the 51 relayouts have receive-only
consumers and one has send-only producers, so both `C ⊄ P` and `C ⊊ P` occur
in shipping patterns (R13, §5; §3.2). `scatter`'s *no* was already argued
(§6.6); the other two are now measurement rather than judgement. What remains
is a `?` cell for `reduce_scatter` alone.

**Why `reduce_scatter` stays open.** It is the one op with no measured path
at all (§7.9): relayouts do not combine (§7's preamble), so nothing in the
evidence reaches either `fold` op. `reduce`'s *yes* is one op's
implementation choice, argued from its own semantics, and whether a
reduce-*scatter*'s consumers must also have contributed is genuinely
undecided. Related and equally open on the `fold` side: R14's mode gate
(all-reduce or reduce-to-one, no strict multi-tile subset) is a present
restriction on `reduce` awaiting the same call.

### 9.2 Two things a work division cannot settle

Both are escape hatches in §7.2's Step 7, and both need the per-region core
sets rather than the division.

**Producer election.** Rows 2–4 return *insufficient information* when a
source region has several holders: R8 requires each consumer tile to have
exactly one source, and the tables record who *holds* a region, not who
*transmits* it. Unforced by measurement — every source region across the 51
relayouts has exactly one holder (§7.2, Step 1) — so the choice between
electing a canonical producer (lowest tile id), requiring the frontend to
pick, and rejecting replicated sources can wait.

**Replication versus idleness.** Row 3's `prod(Nd(a)) == num_cores` is weaker
than asking whether a destination region is shared, and the two part company on
every row-4 output: a region held by several cores is either genuine
replication or one consumer plus idle cores. Both occur in measurement — one
region held by 28 cores with 4 idle, and the broadcast work genuinely
replicating — so the distinction is real. What is open is whether the op
surface should mark it, or whether `consumer_tiles_per_group` naming the actual
holders suffices. This is why §7.2's membership steps (Steps 1, 3–5) run before the
core-count test.

### 9.3 Physicalization: which ops are layout-transparent

Raised by Triton issue #92. **Physicalization** rewrites a tensor to a stick
layout, splitting one axis by the stick size with the chunk count at the front
and the within-stick extent at the back:

```
logical [16, 64], stick on the 64 axis, stick = 32
     →  physical [64/32, 16, 32] = [2, 16, 32]
```

Rank grows by one and the logical stick axis becomes **two non-adjacent
physical axes**. Nothing in this repository represents a stick layout today, so
what follows is a design obligation, not current behaviour.

**Why today's `reduce` is transparent.** It carries no axis-index attribute and
pins results to partials (`KTDP.td:168-171`), so physicalizing the input carries
the result along with no op knowledge — the "elementwise" property. Issue #92's
failure is adjacent: the `identity` operand is tied to results
(`KTDP.td:172-174`) but materialized at logical rank before any layout pass
runs. That is a *propagation* bug, and since the identity is a splat,
re-materializing it at the right type is shape-agnostic by construction.

**The split follows §1.1 exactly**, because §4 makes result type a function of
`placement` alone and `replicate` is the only placement naming no axis set:

| Op | placement | Axis attrs | Result vs partial | Transparent? |
|---|---|---|---|---|
| `consume` | replicate | — | identical | **yes** |
| `reduce` | replicate | — | identical | **yes** |
| `reduce_scatter` | split | `scatter_dimensions` | ÷ `C` | no — attrs, shape, identity |
| `gather` | concat | `gather_dimensions` | × `P` | no — attrs, shape |
| `all_to_all` | permute | `split_`/`concat_dimensions` | ÷ `C` and × `P` | no — attrs, shape |
| `scatter` | split | `scatter_dimensions` | ÷ `C` | no — attrs, shape |

`consume` joins `reduce`. `scatter` does **not**, despite being copy-only: it
divides an extent and names the axis it divides. No rank reduction (§4) is
load-bearing here — a collapse is an axis-*position* operation, so a `reduce`
that collapsed would not be transparent either.

**What §4's rules already settle.** A dim attribute naming a sticked axis
becomes *two* indices (`[1]` → `[0, 2]`), which only the list-valued form can
express, and §4's slowest-to-fastest order is exactly what the stick layout
produces — physical `(c, m, s)` holds logical `n = c*32 + s`. The floordiv rule
fixes which axis absorbs the ×`P` or ÷`C`, and R9 on the floordiv axis is then
precisely the stick-multiple check: `scatter` with `C = 4` on a 2-chunk axis
fails `2 % 4`, correctly rejecting a logical result of `[16,16]` that is half a
stick. `E(D)` itself is invariant (`2 × 32 = 64`), so R9 and R12 cannot change
verdict on the flattened extent — provided a rewrite lists *both* halves of a
split axis; listing one half is simply the wrong rewrite, and R9 catches it.

**Axis indices shift, and physicalization is where that is handled.** The
chunk-count axis is inserted at the *front*, so logical axis 0 of `[16,64]`
becomes physical axis 1: no dim attribute survives untouched, including one
naming an axis physicalization never split. Left unshifted,
`gather_dimensions = [0]` names the chunk axis instead — a valid, distinct
index, so R9/R12 pass and the op is silently wrong.

The remedy needs no new mechanism. Physicalization **is** the logical-to-physical
mapping, so the pass that applies it already knows which logical axis was split,
the stick size, and where every logical axis landed — exactly the information a
dim attribute needs. The attributes name logical axes as authored, and the pass
rewrites them in the same step it retypes the tensors: `[1]` → `[0, 2]` for the
split axis, `[0]` → `[1]` for the shifted one. Nothing downstream re-derives it,
and the ops stay layout-agnostic, which matches §7.1's framing where axes are
artifact symbol names until lowering.

This is not the shape of issue #92. There the `identity` was missed because
`retypeChain` walks forward along operand 0 and never reaches a sibling
operand — an incompleteness in *which values* the pass visits. Attributes sit on
the op the pass is already rewriting, so they are in reach by construction; what
is required is that the mapping be applied to them, not that it be discovered
somewhere else.

**What is still open.**

1. **R12's per-axis clause gains teeth.** Single-axis lists make it trivial;
   `[0,2]` makes it two checks. Stick size depends on element type (32 for f32,
   64 for f16), so variadic roles with mixed types can have equal products and
   unequal per-axis extents — reachable, since §3.7 requires all roles to share
   one axis set.
2. **`reduce_scatter`'s identity.** Its identity must match `T_p` while its
   result is `T_p` split by `C`, so issue #92's fix is needed there in a harder
   form — and the shipped constraint must be retargeted from results to partials
   (R11, §5).
3. **The floordiv rule against a sticked multi-axis pattern**, once one is
   measured. §7.4's three-axis concat has no sticked axis among its listed
   axes, so it does not test it.

### 9.4 One thing unverified

Not a design question; it is a fact this document's claims lean on and
cannot establish.

**Is any field on the artifact's `op` meaningfully authorable?** §7.1
observes that the whole `op` object is byte-identical across all 51 measured
files, which establishes that an emitter reproducing the measured set need
not compute any of it. It does **not** establish that no field matters for a
pattern outside the measured set. Nothing in the artifact can settle that,
so it is recorded here rather than asserted there.

### 9.5 Fused relayout is deferred

Relayout stays a separate preceding op, and fusing it into the consuming
computation is treated as a lowering concern rather than an op-surface one.
Nothing in the measured set fuses one, so there is no evidence either way
about what the surface would have to express; the question is recorded here
so that the absence is deliberate rather than an oversight.

---

## Appendix A. Relationship to what exists today

**Two of the seven ops exist.** `include/ktir/Dialect/KTDP/KTDP.td` defines
`ktdp.inter_tile_produce` and `ktdp.inter_tile_reduce`, plus the
`ktdp.yield_partial` and `ktdp.yield_reduced` terminators. That is all —
the other five delivery ops are new work, not revisions of existing ops.

**Migration table.** One row per op of this design, with its state in
`KTDP.td` today.

| Op | Status today | This design |
|---|---|---|
| `inter_tile_produce` | exists (`KTDP.td:107`) | already matches: carries `producer_tiles_per_group` and no consumer set, returns a future |
| `inter_tile_reduce` | exists (`KTDP.td:165`) | already matches: consumes the future, carries `consumer_tiles_per_group` and a reducer region only |
| `inter_tile_consume` | **not implemented** | new (§6.1) |
| `inter_tile_reduce_scatter` | **not implemented** | new (§6.3) |
| `inter_tile_gather` | **not implemented** | new (§6.4) |
| `inter_tile_all_to_all` | **not implemented** | new (§6.5) |
| `inter_tile_scatter` | **not implemented** | new (§6.6) |

`inter_tile_consume` and `inter_tile_reduce_scatter` appear in the current
tree only as prose: `KTDP.td:70` names them as unbuilt future work, and
`KTDPTypes.td:235` lists them among the delivery ops the future type is
*intended* to serve. Neither has an op definition, so this document is a
specification for five new ops rather than a restructuring of existing
ones.

Cross-checking against §7's census: of the five unbuilt ops, `gather`,
`all_to_all`, `scatter` and `consume` are required by measured relayouts,
while `reduce_scatter` is required by no measurement at all (§7.9).

**The `!ktdp.tile_future<(T), groups = #groups>` type** already exists
(`KTDPTypes.td:231`) and is shared across all ops; its `#groups` parameter
carries the group set (§1.3).

**The earlier single-op draft.** A `ktdp.inter_tile` op carrying producer
and optional combiner regions in one op, with `consumer_tiles_per_group`
determining the delivery mode, was drafted but never landed. Splitting
production from delivery makes the mode a choice of op rather than an
inference over attribute combinations — which is what lets §3 state the
shared machinery once and §6 reduce each op to its own cells.
