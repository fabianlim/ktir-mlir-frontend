# Inter-tile communications in KTIR

**Scope:** Seven ops — one production op, `ktdp.inter_tile_produce`, and
six delivery ops: `ktdp.inter_tile_consume`, `ktdp.inter_tile_reduce`,
`ktdp.inter_tile_reduce_scatter`, `ktdp.inter_tile_gather`,
`ktdp.inter_tile_all_to_all`, and `ktdp.inter_tile_scatter`. Together they
cover the six inter-core communication patterns: broadcast, all-reduce,
reduce-scatter, gather, all-to-all, and scatter.

**Organization.** The delivery ops share almost all of their machinery.
That machinery is stated once, in §3 (operand, consumer set, local index,
dependency attribute, combiner, synchronization, result semantics), §4
(type rules), and §5 (verification rules). §6 then defines each op by
what is *only* true of it.

**Rule numbering.** The verification rules are numbered R1–R14 and
collected in §5, which is their single point of definition. They are cited
as `(Rn)` at the place the attribute they constrain is introduced, so a
citation like `(R1)` in §2.1 means "§5 states this rule; here is the
attribute it applies to."

Sections are normative except §8 (implementation status) and §9 (observed
backend patterns).

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
two `ktdp.inter_tile_produce` ops; §10.3 discusses relaxing that.

### 1.1 Semantics matrix

The six delivery ops differ in exactly three independent properties.
"Property" rather than "axis" throughout: in this document *axis* always
means a tensor or tile axis.

- **combine** — `none` | `fold` (combiner region + identity operand).
- **placement** — how producer contributions map onto consumer results:
  `replicate` | `concat` | `permute` | `split`.
- **cardinality** — producer tiles per group × consumer tiles per group.

| Op | combine | placement | producers/grp | consumers/grp | dim attrs | region | identity |
|---|---|---|---|---|---|---|---|
| `consume` | none | replicate | 1 | free | — | — | — |
| `reduce` | fold | replicate | all | free | — | combiner | yes |
| `reduce_scatter` | fold | split | all | free | `scatter_dim` | combiner | yes |
| `gather` | none | concat | all | free | `gather_dim` | — | — |
| `all_to_all` | none | permute | all | all | `scatter_dim`, `gather_dim` | — | — |
| `scatter` | none | split | 1 | free | `scatter_dim` | — | — |

`all_to_all` is listed before `scatter` because it shares the
all-producers cardinality cell with `gather` and `reduce_scatter`, and
because its relationship to the two copy-only placements is structural:
**permute = split + concat in one step**, which is why it carries both dim
attributes and no new ones.

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
variant: R8 (§5) limits them to one producer per group.

### 1.3 The future value

`ktdp.inter_tile_produce` returns a `!ktdp.tile_future<T_p, #groups>` SSA
value. The group set `#groups` is carried as a parameter of the future
type rather than repeated as a separate `groups` attribute on both the
production and delivery ops. Each delivery op therefore infers the groups
from its operand type, and a group mismatch between production and
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
`4g .. 4g+3` for any group index `g`. An enumerated form (a list of tile-id
lists) is supported as a fallback when per-group membership is irregular.
Which cardinality each delivery op requires of this set is given by the
`producers/grp` column of §1.1 and enforced by R8 (§5).

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
`!ktdp.tile_future<..., #groups>` type, and every delivery op infers it
from its operand type (§1.3).

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
scope (where it is naturally executed by every tile under SPMD). Richer
bodies are allowed when the contribution is awkward to hoist — e.g.,
compute that depends only on `%gid`, or visibly local "contribution
preparation" the author wants to keep adjacent to the op. A
single-producer-per-group op (`consume`, `scatter`) is the case where a
richer body is normally *required*: the loads that feed the partial must
not run on the group's non-producing tiles, so they belong inside the
region rather than at function scope (§7.7.1).

### 2.3 Op signature

```mlir
%future = ktdp.inter_tile_produce
    producer_tiles_per_group = <affine-set>
    : T_p_1, ..., T_p_N -> !ktdp.tile_future<T_p_1, ..., T_p_N, #groups>
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
`ktdp.inter_tile_produce`. §10.2 discusses relaxing this.

---

## 3. Shared delivery semantics

Everything in this section holds for **every** delivery op. §6 states
only per-op deltas; where §6 is silent, this section governs.

### 3.1 Notation

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

**Operand:** `!ktdp.tile_future<T_p_1, ..., T_p_N, #groups>` — the future
returned by the corresponding `ktdp.inter_tile_produce`. The def-use edge
is the ordering constraint, and the `#groups` parameter of this type
supplies the group set. There is no separate `groups` attribute; a group
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

### 3.3 Within-group local index — normative

**`l` is a tile's position, counting from 0 in ascending tile-id order,
among the relevant set within its group** — the producer set for `concat` placement (and for
`permute`'s `gather_dim`), the consumer set for `split` placement (and
for `permute`'s `scatter_dim`).

This definition is what makes ordered placement well-defined. Without it,
concatenation and split orders are pinned down only by contiguous-tile-id
coincidence and break silently under non-monotone tile assignments. Every
"ascending local-index order" in this document means exactly this
position — never a tile id, and never an offset in the textual order of an
enumerated set. ("Position" rather than "rank": in this document *rank*
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
  pairing, `(p)[c] : (p - c + 2 == 0)`. Note: because groups are
  disjoint and integer division is not affine, `g` cannot be derived
  from `c` alone; it must appear explicitly whenever the constraint
  involves group-relative addresses such as `4*g`.
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
`!ktdp.tile_future<T_p, #groups>` SSA value carries **per-tile
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
the same group hold the *same* value is a property of the placement:

| placement | tiles in one group hold | tiles in different groups hold |
|---|---|---|
| `replicate` | the same value | their own group's value |
| `concat` | the same assembled tensor | their own group's assembly |
| `split` | disjoint ordered slices that tile the whole | slices of their own group's tensor |
| `permute` | different assemblies (one slice per producer) | their own group's exchange |

**Non-participating tiles.** Results are undefined for tiles not in
`consumer_tiles_per_group`.

**Multi-tensor (variadic) delivery.** `N >= 1` roles are supported by
every op, and all roles share the same attributes (`scatter_dim`,
`gather_dim`, `P`, `C`) — only the types differ. Argmax-style reductions,
where each contribution is a correlated tuple of tensors (values,
indices), use `N = 2`: two identities, two yielded partials, four
combiner arguments yielding two combined values, two op results. Each
role's result type follows the §4 rule independently.

---

## 4. Placement algebra and type rules

Result types are a function of the placement value alone. There are four
formulas, applied per role `i` to `T_p_i`:

| placement | result type derived from `T_p` |
|---|---|
| `replicate` | within-group tile axes collapsed (`fold`) / `T_p` unchanged (`none`) |
| `concat` | extent along `gather_dim` multiplied by `P` |
| `split` | extent along `scatter_dim` divided by `C` |
| `permute` | extent along `scatter_dim` divided by `C`, **and** extent along `gather_dim` multiplied by `P` |

`reduce_scatter` is `fold` + `split`: the within-group tile axes are
collapsed first, then the `split` formula applies to the collapsed type.

**Which slice a tile gets.** For `split`, the consumer with local index
`l` (§3.3) receives `[l*chunk : (l+1)*chunk]` along `scatter_dim`, where
`chunk = T_p[scatter_dim] / C`. For `concat`, the producer with local
index `l` occupies `[l*chunk : (l+1)*chunk]` along `gather_dim`, where
`chunk = T_p[gather_dim]` is that producer's own extent along the axis.
For `permute`, both hold simultaneously: consumer `l_c` receives, from
each producer `l_p`, that producer's `scatter_dim` slice `l_c`, placed at
`gather_dim` position `l_p`.

**Conservation in the square case.** Whenever `P == C`, the `permute`
result has the same element count as `T_p` — one axis is divided and
another multiplied by the same factor — so a square all-to-all is a pure
redistribution of ownership. If additionally `scatter_dim == gather_dim`,
the result *type* equals `T_p`: the distributed transpose, which is the
uniform one-to-one shuffle the SDSC backend emits today (§9).

**Why `split` divides an honest data axis.** `reduce_scatter` collapses a
*within-group tile axis* and then splits, so its partial must carry an
artificial unit dimension for the collapse to consume. `scatter`,
`gather`, and `all_to_all` split or grow a natural data axis directly, so
their types stay honest: `<128x1x64>` → `<32x1x64>` rather than
`<1x...>`.

---

## 5. Verification rules

Principle: **each rule has exactly one owner and one statement;
applicability is a column, not a restatement.** "Owner" is the op that
carries the attribute the rule constrains.

| Rule | Owner | consume | reduce | red_scat | gather | all_to_all | scatter |
|---|---|---|---|---|---|---|---|
| R1 group disjointness (§2.1) | produce | y | y | y | y | y | y |
| R2 single-use future (§2.3) | produce | y | y | y | y | y | y |
| R3 dep set subset of producers | delivery | y | y | y | y | y | n/a |
| R4 every producer covered by some consumer | delivery | y | y | y | y | y | n/a |
| R5 dep sets pairwise disjoint | delivery | — | — | — | y | y | n/a |
| R6 uniform dep-set cardinality | delivery | — | — | — | y | y | n/a |
| R7 uniform producer cardinality across groups | delivery | — | — | — | y | y | n/a |
| R8 producers per group = 1 | delivery | y | — | — | — | — | y |
| R9 `scatter_dim` extent divisible by `C` | delivery | — | — | y | — | y | y |
| R10 combiner purity (§3.5) | delivery | — | y | y | — | — | — |
| R11 identity shape matches `T_p` (§3.5) | delivery | — | y | y | — | — | — |
| R12 `gather_dim` extent × `P` well-defined | delivery | — | — | — | y | y | — |
| R13 consumer set subset of producer set | delivery | — | y | ? | ? | ? | n |
| R14 reduce mode gate: `C == P` or `\|C\| == 1` | delivery | — | y | ? | — | — | — |

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
- **R8 — single producer.** Exactly one producer tile per group, for the
  ops whose `producers/grp` cell is `1`.
- **R9 — split divisibility.** `T_p[scatter_dim] % C == 0` (for
  `reduce_scatter`, the post-collapse extent). One rule covers all three
  splitting ops because they share the `scatter_dim` attribute; a separate
  divisibility rule for `all_to_all` would only be needed if its split
  axis had its own attribute name.
- **R12 — concat well-definedness.** The result extent along `gather_dim`
  is `P × T_p[gather_dim]`, which requires every assembled producer to
  contribute the same extent along that axis. For the square
  `all_to_all` case this follows from R7 + R9, but it must be stated
  independently for the non-square case.
- **R13 — consumer set subset of producer set.** Every consumer tile in a
  group must also be a producer in that group, i.e.
  `consumer_tiles_per_group(g) ⊆ producer_tiles_per_group(g)`. Whether
  this should hold is §10.1; it is currently enforced for `reduce` only.
- **R14 — reduce mode gate.** For `reduce`, the consumer set must either
  equal the producer set (all-reduce) or be a single tile
  (reduce-to-one); a strict multi-tile subset — reduce-to-subset — is
  rejected. This is a present implementation restriction, not a design
  conclusion (§10.1).

---

## 6. The delivery ops

Each subsection states only what is specific to that op: its cells from
§1.1, its result type from §4, its signature, and any op-specific
argument. Shared machinery is §3; rules are §5.

### 6.1 `ktdp.inter_tile_consume` — broadcast

`combine = none`, `placement = replicate`, one producer per group,
consumer set free, no dim attribute, no region, no identity.

**Result type.** `T_p_i` unchanged (§4, `replicate` + `none`).

**Semantics.** No combining occurs. The value produced by the group's
producer tile is delivered unchanged to every consumer tile in that
group — broadcast.

```mlir
%result_1, ..., %result_N = ktdp.inter_tile_consume(%future)
    consumer_tiles_per_group         = <affine-set>,
    producer_dependency_per_consumer = <affine-set>   // optional; default: all producers
    : !ktdp.tile_future<T_p_1, ..., T_p_N, #groups> -> T_p_1, ..., T_p_N
```

With one producer per group, `producer_dependency_per_consumer` is a pure
synchronization refinement (§3.4): it changes when each consumer
unblocks, never what it receives. That is what makes `consume` also the
op for one-to-one permutation exchange — a bijective dependency set over
a multi-producer group (§7.4.2).

### 6.2 `ktdp.inter_tile_reduce` — reduction

`combine = fold`, `placement = replicate`, all tiles produce, consumer
set free, no dim attribute, combiner region and `identity` per §3.5.

**Result type.** `T_r_i` is `T_p_i` with the within-group tile axes
collapsed; the same axes are removed for all roles.

```mlir
%r_1, ..., %r_N = ktdp.inter_tile_reduce(%future)
    consumer_tiles_per_group         = <affine-set>,
    producer_dependency_per_consumer = <affine-set>,   // optional; default: all producers
    identity(%id_1 : T_p_1, ..., %id_N : T_p_N)
    : !ktdp.tile_future<T_p_1, ..., T_p_N, #groups> -> T_r_1, ..., T_r_N
{
  ^bb0(%lhs_1: T_p_1, ..., %lhs_N: T_p_N,
       %rhs_1: T_p_1, ..., %rhs_N: T_p_N):
    ktdp.yield_reduced %val_1, ..., %val_N : T_p_1, ..., T_p_N
}
```

Consumer set = producer set is all-reduce; a single consumer per group is
reduce-to-one. Both are supported today; a strict multi-tile subset is
not (R14).

### 6.3 `ktdp.inter_tile_reduce_scatter` — reduction then split

`combine = fold`, `placement = split`, all tiles produce, consumer set
free, `scatter_dim`, combiner region and `identity` per §3.5.

**`scatter_dim`** (i64) — axis of the *post-collapse* type along which the
reduced result is split row-major across the consumer tiles (R9).

**Result type.** `T_r_i` is `T_p_i` with the within-group tile axes
collapsed and then divided by `C` along `scatter_dim`. The same axes and
the same split apply to all roles.

```mlir
%chunk_1, ..., %chunk_N = ktdp.inter_tile_reduce_scatter(%future)
    consumer_tiles_per_group         = <affine-set>,
    scatter_dim                      = <i64>,
    producer_dependency_per_consumer = <affine-set>,   // optional; default: all producers
    identity(%id_1 : T_p_1, ..., %id_N : T_p_N)
    : !ktdp.tile_future<T_p_1, ..., T_p_N, #groups> -> T_r_1, ..., T_r_N
{
  ^bb0(%lhs_1: T_p_1, ..., %lhs_N: T_p_N,
       %rhs_1: T_p_1, ..., %rhs_N: T_p_N):
    ktdp.yield_reduced %val_1, ..., %val_N : T_p_1, ..., T_p_N
}
```

### 6.4 `ktdp.inter_tile_gather` — ordered assembly

`combine = none`, `placement = concat`, all tiles produce, consumer set
free, `gather_dim`, no region, no identity.

**`gather_dim`** (i64) — axis of `T_p` along which the producers'
partials are concatenated, in ascending producer local-index order
(§3.3).

**Result type.** `T_g_i` is `T_p_i` with the extent along `gather_dim`
multiplied by `P` (R12).

```mlir
%gathered_1, ..., %gathered_N = ktdp.inter_tile_gather(%future)
    consumer_tiles_per_group         = <affine-set>,
    gather_dim                       = <i64>,
    producer_dependency_per_consumer = <affine-set>   // optional; default: all producers
    : !ktdp.tile_future<T_p_1, ..., T_p_N, #groups> -> T_g_1, ..., T_g_N
```

One consumer per group is a plain gather; the full group as consumer set
is all-gather — the same op with a wider set (§1.2), not a separate op.
With `producer_dependency_per_consumer` present the assembly is a partial
(segmented) gather over each consumer's declared subset, subject to
R5–R7.

### 6.5 `ktdp.inter_tile_all_to_all` — split and reassemble

`combine = none`, `placement = permute`, all tiles produce, all tiles
consume, both `scatter_dim` and `gather_dim`, no region, no identity.

**Attributes.** `scatter_dim` (i64) — axis each producer splits into `C`
chunks (R9). `gather_dim` (i64) — axis along which each consumer
concatenates the chunks it received, in ascending producer local-index
order (R12). The two may be equal (pure ownership transpose along one
axis) or different (reshape-transpose, e.g. split heads and regather
sequence).

**Result type.** `T_c_i` is `T_p_i` with the `scatter_dim` extent divided
by `C` and the `gather_dim` extent multiplied by `P`. In the square case
(`P == C` and `scatter_dim == gather_dim`) `T_c_i == T_p_i` (§4).

**Semantics.** Consumer with local index `l_c` receives, from each
producer `l_p`, the slice `[l_c*chunk : (l_c+1)*chunk]` of that
producer's tensor along `scatter_dim` (`chunk = T_p[scatter_dim] / C`),
and concatenates those `P` slices along `gather_dim` in ascending `l_p`
order.

```mlir
%out_1, ..., %out_N = ktdp.inter_tile_all_to_all(%future)
    consumer_tiles_per_group         = <affine-set>,
    scatter_dim                      = <i64>,
    gather_dim                       = <i64>,
    producer_dependency_per_consumer = <affine-set>   // optional; default: all producers
    : !ktdp.tile_future<T_p_1, ..., T_p_N, #groups> -> T_c_1, ..., T_c_N
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
already expressible as `consume` + a bijective dependency set (§7.4.2);
`all_to_all` is only for the split-and-redistribute case, so the two
mechanisms do not overlap.

**Generalizing `scatter` to `P > 1` is not an alternative.** Adding a
`gather_dim` and lifting R8 on `scatter` *is* `all_to_all` under another
name; it hides the multi-producer wait inside `scatter` and gives that op
two regimes. A separate op keeps one-op-one-pattern and leaves
`scatter`'s `P == 1` contract clean.

### 6.6 `ktdp.inter_tile_scatter` — ordered split

`combine = none`, `placement = split`, one producer per group (R8),
consumer set free, `scatter_dim`, no region, no identity.

**`scatter_dim`** (i64) — axis of `T_p` along which the single producer's
tensor is partitioned into `C` equal chunks (R9), one per consumer in
ascending consumer local-index order (§3.3).

**Result type.** `T_s_i` is `T_p_i` with the extent along `scatter_dim`
divided by `C`.

```mlir
%scattered_1, ..., %scattered_N = ktdp.inter_tile_scatter(%future)
    consumer_tiles_per_group = <affine-set>,
    scatter_dim              = <i64>
    : !ktdp.tile_future<T_p_1, ..., T_p_N, #groups> -> T_s_1, ..., T_s_N
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
resolves §10.1 for `scatter`, and only for `scatter`.

---

## 7. Pattern instantiation

### 7.1 Broadcast  →  `inter_tile_produce` + `inter_tile_consume`

```mlir
// 4 tiles, 1 group: tile 0 loads W; all 4 tiles compute.
#tile_0          = affine_set<(i)[g] : (i - 4*g == 0)>
#all_group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#single_group    = affine_set<(g) : (g == 0)>

%W_future = ktdp.inter_tile_produce
    producer_tiles_per_group = #tile_0
    : tensor<64x128xf16> -> !ktdp.tile_future<tensor<64x128xf16>, #single_group>
{
  ^bb0(%gid: index):
    %W = ktdp.load ...
    ktdp.yield_partial %W : tensor<64x128xf16>
}

// Every consumer tile extracts its copy; no combiner → value passes through.
// Groups are inferred from the future's #single_group parameter.
%W_tile = ktdp.inter_tile_consume(%W_future)
    consumer_tiles_per_group = #all_group_tiles
    : !ktdp.tile_future<tensor<64x128xf16>, #single_group> -> tensor<64x128xf16>

// Post-delivery SPMD compute — owned by consumer_tiles_per_group.
// Ownership verified by traversing the def-use chain from %W_tile.
%A = ktdp.load ...
%C = linalg.matmul ins(%A, %W_tile ...) ...
ktdp.store %C, ...
```

### 7.2 Reduce  →  `inter_tile_produce` + `inter_tile_reduce`

```mlir
// 4 tiles per group, 8 groups (32 tiles total).
#all_group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#all_groups      = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

// All tiles contribute a partial; future carries all partials.
%partial_future = ktdp.inter_tile_produce
    producer_tiles_per_group = #all_group_tiles
    : tensor<1x64xf16> -> !ktdp.tile_future<tensor<1x64xf16>, #all_groups>
{
  ^bb0(%gid: index):
    ktdp.yield_partial %partial_2d : tensor<1x64xf16>
}

// Reduce all partials; every consumer tile receives the same reduced value.
%reduced = ktdp.inter_tile_reduce(%partial_future)
    consumer_tiles_per_group = #all_group_tiles,
    identity(%add_id : tensor<1x64xf16>)
    : !ktdp.tile_future<tensor<1x64xf16>, #all_groups> -> tensor<1x64xf16>
{
  ^bb0(%lhs: tensor<1x64xf16>, %rhs: tensor<1x64xf16>):
    %sum = linalg.add ins(%lhs, %rhs ...) ...
    ktdp.yield_reduced %sum : tensor<1x64xf16>
}
```

#### 7.2.1 Full IR — single-group reduce (96×64)

**Layout and partitioning.** `A` and `B` are `tensor<96x64xf16>` in global memory.
The kernel computes the column-wise sum of `A + B`, producing a
1-D `tensor<64xf16>`.

The 32 compute tiles form a single group. Tile `t` owns rows
`t*3 .. t*3+2` of `A` and `B` — a 3×64 slab each. The per-tile
contribution is the row-reduced partial expanded to `tensor<1x64xf16>`,
where the leading unit dimension is the within-group tile axis the op
collapses. Every tile holds the same `%reduced : tensor<64xf16>` (all-reduce case: consumer set = producer set).

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
        : tensor<1x64xf16> -> !ktdp.tile_future<tensor<1x64xf16>, #all_groups>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %partial_2d : tensor<1x64xf16>
    }

    // Reduce: unit dim 0 is the within-group tile axis; the op collapses it.
    // Every tile holds the same %reduced : tensor<64xf16> (all-reduce case).
    %reduced = ktdp.inter_tile_reduce(%partial_future)
        consumer_tiles_per_group = #group_tiles,
        identity(%add_id : tensor<1x64xf16>)
        : !ktdp.tile_future<tensor<1x64xf16>, #all_groups> -> tensor<64xf16>
    {
      ^bb0(%lhs: tensor<1x64xf16>, %rhs: tensor<1x64xf16>):
        %init = tensor.empty() : tensor<1x64xf16>
        %sum  = linalg.add ins(%lhs, %rhs : tensor<1x64xf16>, tensor<1x64xf16>)
                           outs(%init : tensor<1x64xf16>) -> tensor<1x64xf16>
        ktdp.yield_reduced %sum : tensor<1x64xf16>
    }

    // Post-reduction: every tile redundantly writes the same value.
    %reduced_2d = tensor.expand_shape %reduced [[0, 1]] output_shape [1, 64]
                    : tensor<64xf16> into tensor<1x64xf16>

    %E_view = ktdp.construct_memory_view %E_start, sizes: [1, 64], strides: [64, 1] {
        coordinate_set = #E_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<1x64xf16>
    %E_access = ktdp.construct_access_tile %E_view[%c0, %c0] {
        access_tile_set = #E_tile_set, access_tile_order = #identity_2d
    } : memref<1x64xf16> -> !ktdp.access_tile<1x64xindex>

    ktdp.store %reduced_2d, %E_access
              : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>

    return
  }
}
```

#### 7.2.2 Full IR — multi-group reduce (128×8×12×64)

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

The partial is `<128x1x1x64>`: dim 1 is the group axis (preserved), dim 2
is the within-group tile axis (collapsed by the op to `<128x1x64>`). All
four tiles in a group hold identical values; different groups hold
different values.

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
        : tensor<128x1x1x64xf16>
          -> !ktdp.tile_future<tensor<128x1x1x64xf16>, #all_groups>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %partial_4d : tensor<128x1x1x64xf16>
    }

    // Multi-group reduce: dim 2 (within-group tile axis) collapsed.
    // Dim 1 (group axis) preserved. Each tile gets its group's <128x1x64>.
    %my_group_result = ktdp.inter_tile_reduce(%partial_future)
        consumer_tiles_per_group = #group_tiles,
        identity(%add_id : tensor<128x1x1x64xf16>)
        : !ktdp.tile_future<tensor<128x1x1x64xf16>, #all_groups>
          -> tensor<128x1x64xf16>
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
    %my_result_4d = tensor.expand_shape %my_group_result [[0], [1, 2], [3]]
                      output_shape [128, 1, 1, 64]
                      : tensor<128x1x64xf16> into tensor<128x1x1x64xf16>

    %E_view = ktdp.construct_memory_view %E_start, sizes: [128, 8, 4, 64],
        strides: [2048, 256, 64, 1] {
        coordinate_set = #E_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<128x8x4x64xf16>

    %E_access = ktdp.construct_access_tile %E_view[%c0, %g, %l, %c0] {
        access_tile_set = #E_tile_set, access_tile_order = #identity_4d
    } : memref<128x8x4x64xf16> -> !ktdp.access_tile<128x1x1x64xindex>

    ktdp.store %my_result_4d, %E_access
              : tensor<128x1x1x64xf16>, !ktdp.access_tile<128x1x1x64xindex>

    return
  }
}
```

### 7.3 Reduce-scatter  →  `inter_tile_produce` + `inter_tile_reduce_scatter`

```mlir
// 4 tiles per group, 8 groups (32 tiles total).
#all_group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#all_groups      = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

// All tiles contribute a partial.
%partial_future = ktdp.inter_tile_produce
    producer_tiles_per_group = #all_group_tiles
    : tensor<128x1x1x64xf16> -> !ktdp.tile_future<tensor<128x1x1x64xf16>, #all_groups>
{
  ^bb0(%gid: index):
    ktdp.yield_partial %partial_4d : tensor<128x1x1x64xf16>
}

// Reduce and scatter; each tile receives its own slice along dim 0.
// scatter_dim = 0 → 128-row axis split across 4 tiles; each gets <32x1x64>.
%my_chunk = ktdp.inter_tile_reduce_scatter(%partial_future)
    consumer_tiles_per_group = #all_group_tiles,
    scatter_dim              = 0,
    identity(%add_id : tensor<128x1x1x64xf16>)
    : !ktdp.tile_future<tensor<128x1x1x64xf16>, #all_groups> -> tensor<32x1x64xf16>
{
  ^bb0(%lhs: tensor<128x1x1x64xf16>, %rhs: tensor<128x1x1x64xf16>):
    %sum = linalg.add ins(%lhs, %rhs ...) ...
    ktdp.yield_reduced %sum : tensor<128x1x1x64xf16>
}
// Each tile holds a different slice — ownership explicit via SSA result.
```

#### 7.3.1 Full IR — multi-group reduce-scatter (128×8×12×64)

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
to §7.2.2.

The op reduces dim 2 (within-group tile axis, size 1) and scatters dim 0
(128 / 4 = 32 rows per tile). Tile `(g, l)` ends up with rows
`[l*32 : (l+1)*32]` of group `g`'s reduced `<128x1x64>`.

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

// E access tile per writer: 32x1x64 anchored at [l*32, g, 0].
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
        : tensor<128x1x1x64xf16>
          -> !ktdp.tile_future<tensor<128x1x1x64xf16>, #all_groups>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %partial_4d : tensor<128x1x1x64xf16>
    }

    // Reduce dim 2 (within-group tile axis). Scatter dim 0 (chunk = 32).
    // Group axis (dim 1) preserved. Each tile receives <32x1x64>.
    %my_chunk = ktdp.inter_tile_reduce_scatter(%partial_future)
        consumer_tiles_per_group = #group_tiles,
        scatter_dim              = 0,
        identity(%add_id : tensor<128x1x1x64xf16>)
        : !ktdp.tile_future<tensor<128x1x1x64xf16>, #all_groups>
          -> tensor<32x1x64xf16>
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

    ktdp.store %my_chunk, %E_access
              : tensor<32x1x64xf16>, !ktdp.access_tile<32x1x64xindex>

    return
  }
}
```

### 7.4 Per-tile synchronization  →  `inter_tile_consume` with `producer_dependency_per_consumer`

#### 7.4.1 Per-tile pairing within a single group

Four tiles per group: tiles `4g` and `4g+1` are producers, tiles `4g+2`
and `4g+3` are consumers. Each consumer depends on its dedicated producer
(`4g+2` ← `4g`, `4g+3` ← `4g+1`), so the pairing is `p = c - 2` — a
constant relative offset that does not depend on the group index `g`.

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
    : tensor<64xf16> -> !ktdp.tile_future<tensor<64xf16>, #single_group>
{
  ^bb0(%gid: index):
    %data = ktdp.load ...
    ktdp.yield_partial %data : tensor<64xf16>
}

// Each consumer unblocks independently as its assigned producer finishes.
%my_data = ktdp.inter_tile_consume(%data_future)
    consumer_tiles_per_group         = #consumer_tiles,
    producer_dependency_per_consumer = #dep_per_consumer
    : !ktdp.tile_future<tensor<64xf16>, #single_group> -> tensor<64xf16>
```

Without `producer_dependency_per_consumer`, both consumers stall until
both producers finish. With it, each consumer stalls only for its own
producer, halving the worst-case wait when the two producers finish at
different times.

#### 7.4.2 Butterfly mirror exchange across multiple groups

Eight groups of 4 tiles; all 4 tiles in each group both produce and
consume. Tile `c = 4g + l` waits only for its mirror partner
`p = 4g + (3 - l)`, equivalent to `p + c = 8g + 3`. This models a
butterfly-style partner exchange.

Both `c` and `g` are required: `c` identifies which specific consumer is
asking (different consumers within the group have different mirrors), and
`g` anchors the equation to the group (the target sum `8g + 3` is `3`,
`11`, `19`, ... for groups `0`, `1`, `2`, ..., so `g` cannot be
eliminated).

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
    : tensor<64xf16> -> !ktdp.tile_future<tensor<64xf16>, #all_groups>
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
    : !ktdp.tile_future<tensor<64xf16>, #all_groups> -> tensor<64xf16>
```

### 7.5 Gather  →  `inter_tile_produce` + `inter_tile_gather`

```mlir
// 4 tiles per group, 8 groups (32 tiles total).
#all_group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#group_consumer  = affine_set<(i)[g] : (i - 4*g == 0)>
#all_groups      = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

// All tiles contribute a partial slab.
%partial_future = ktdp.inter_tile_produce
    producer_tiles_per_group = #all_group_tiles
    : tensor<128x1x3x64xf16> -> !ktdp.tile_future<tensor<128x1x3x64xf16>, #all_groups>
{
  ^bb0(%gid: index):
    ktdp.yield_partial %partial_4d : tensor<128x1x3x64xf16>
}

// Gather along dim 2; one consumer per group (tile 4g) assembles the four
// 3-wide slabs. No combiner, no identity — placement is by within-group
// local index. gather_dim = 2 → 3 * 4 = 12; consumer gets <128x1x12x64>.
%assembled = ktdp.inter_tile_gather(%partial_future)
    consumer_tiles_per_group = #group_consumer,
    gather_dim               = 2
    : !ktdp.tile_future<tensor<128x1x3x64xf16>, #all_groups> -> tensor<128x1x12x64xf16>
// The consumer holds the full assembled tensor — ownership via SSA result.
```

#### 7.5.1 Full IR — multi-group gather (128×8×12×64)

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
        : tensor<128x1x3x64xf16>
          -> !ktdp.tile_future<tensor<128x1x3x64xf16>, #all_groups>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %partial_4d : tensor<128x1x3x64xf16>
    }

    // Gather dim 2: 4 producers x 3 = 12. One consumer (tile 4g) per group
    // assembles the full <128x1x12x64>. No combiner region, no identity.
    %assembled = ktdp.inter_tile_gather(%partial_future)
        consumer_tiles_per_group = #group_consumer,
        gather_dim               = 2
        : !ktdp.tile_future<tensor<128x1x3x64xf16>, #all_groups>
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

### 7.6 All-to-all  →  `inter_tile_produce` + `inter_tile_all_to_all`

```mlir
// 4 tiles per group, 8 groups (32 tiles total).
#all_group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#all_groups      = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

// Sequence-parallel production: every tile owns a 128-row shard of all 4 heads.
%partial_future = ktdp.inter_tile_produce
    producer_tiles_per_group = #all_group_tiles
    : tensor<128x1x4x64xf16> -> !ktdp.tile_future<tensor<128x1x4x64xf16>, #all_groups>
{
  ^bb0(%gid: index):
    ktdp.yield_partial %partial_4d : tensor<128x1x4x64xf16>
}

// Head-parallel consumption: split dim 2 (heads) across the 4 consumers,
// regather dim 0 (sequence) from the 4 producers.
// scatter_dim = 2 → 4 / 4 = 1;  gather_dim = 0 → 128 * 4 = 512.
%relaid = ktdp.inter_tile_all_to_all(%partial_future)
    consumer_tiles_per_group = #all_group_tiles,
    scatter_dim              = 2,
    gather_dim               = 0
    : !ktdp.tile_future<tensor<128x1x4x64xf16>, #all_groups> -> tensor<512x1x1x64xf16>
// Every tile is both producer and consumer; P == C == 4, so the element count
// is conserved (128*4 = 512*1) even though the type changes.
```

#### 7.6.1 Full IR — sequence-parallel to head-parallel (512×8×4×64)

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
        : tensor<128x1x4x64xf16>
          -> !ktdp.tile_future<tensor<128x1x4x64xf16>, #all_groups>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %partial_4d : tensor<128x1x4x64xf16>
    }

    // All-to-all: consumer l takes head slice l (scatter_dim = 2, 4 / 4 = 1)
    // from each of the 4 producers, and concatenates them along the sequence
    // axis (gather_dim = 0, 128 * 4 = 512) in ascending producer local-index
    // order. The producer's local index picks the destination row block;
    // the consumer's local index picks the head. No combiner, no identity.
    %relaid = ktdp.inter_tile_all_to_all(%partial_future)
        consumer_tiles_per_group = #group_tiles,
        scatter_dim              = 2,
        gather_dim               = 0
        : !ktdp.tile_future<tensor<128x1x4x64xf16>, #all_groups>
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

### 7.7 Scatter  →  `inter_tile_produce` + `inter_tile_scatter`

```mlir
// 4 tiles per group, 8 groups (32 tiles total).
#group_producer  = affine_set<(i)[g] : (i - 4*g == 0)>
#all_group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#all_groups      = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

// One producer per group (tile 4g) holds the whole 128-row tensor.
%whole_future = ktdp.inter_tile_produce
    producer_tiles_per_group = #group_producer
    : tensor<128x1x64xf16> -> !ktdp.tile_future<tensor<128x1x64xf16>, #all_groups>
{
  ^bb0(%gid: index):
    ktdp.yield_partial %whole : tensor<128x1x64xf16>
}

// Scatter along dim 0; the four tiles per group each receive one 32-row
// chunk. No combiner, no identity — placement is by within-group local
// index. scatter_dim = 0 → 128 / 4 = 32; each consumer gets <32x1x64>.
%chunk = ktdp.inter_tile_scatter(%whole_future)
    consumer_tiles_per_group = #all_group_tiles,
    scatter_dim              = 0
    : !ktdp.tile_future<tensor<128x1x64xf16>, #all_groups> -> tensor<32x1x64xf16>
// Each consumer holds its own 32-row slice — ownership via SSA result.
```

#### 7.7.1 Full IR — multi-group scatter (128×8×64)

**Layout and partitioning.** `A` and `B` are `tensor<128x8x64xf16>` in global
memory. The three axes have distinct roles:

- Dim 0 (size 128): the **scatter axis** — the producer's 128 rows are
  split into 4 chunks of 32, one per consumer tile.
- Dim 1 (size 8): the **group axis** — 8 groups.
- Dim 2 (size 64): vector / stick axis, preserved.

32 tiles, 8 groups of 4. `g = t / 4`, `l = t % 4`. Per group, the single
producer tile `4g` reads its group's whole slab `A[*, g, *]` /
`B[*, g, *]` — shape `<128x1x64>` — sums them, and produces the summed
tensor. Scatter along dim 0 delivers chunk `[l*32 : l*32+32, *, *]` to the
consumer with within-group local index `l`, which writes its `<32x1x64>`
slice back to `E[l*32 : l*32+32, g, *]`.

**Why the loads live inside the produce region.** Unlike the other full-IR
examples, which hoist their `ktdp.load`s to function scope, this one keeps
them inside `ktdp.inter_tile_produce`. That is deliberate, and it follows
from single-producer cardinality (§2.2): only tile `4g` may read the
group's whole slab, so hoisting the loads would make every tile in the
group execute them. The other examples have every tile produce, so
function-scope loads are correct there.

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
    // reads and sums its group's whole 128-row slab.
    %whole_future = ktdp.inter_tile_produce
        producer_tiles_per_group = #group_producer
        : tensor<128x1x64xf16>
          -> !ktdp.tile_future<tensor<128x1x64xf16>, #all_groups>
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

        %AB_init = tensor.empty() : tensor<128x1x64xf16>
        %whole = linalg.add ins(%A_tile, %B_tile
                                : tensor<128x1x64xf16>, tensor<128x1x64xf16>)
                            outs(%AB_init : tensor<128x1x64xf16>)
                            -> tensor<128x1x64xf16>
        ktdp.yield_partial %whole : tensor<128x1x64xf16>
    }

    // Scatter dim 0: 128 / 4 = 32. Each of the four consumer tiles per group
    // receives one 32-row chunk. No combiner region, no identity.
    %chunk = ktdp.inter_tile_scatter(%whole_future)
        consumer_tiles_per_group = #all_group_tiles,
        scatter_dim              = 0
        : !ktdp.tile_future<tensor<128x1x64xf16>, #all_groups>
          -> tensor<32x1x64xf16>

    // Post-scatter: consumer (g, l) writes its 32-row chunk to
    // E[l*32 : l*32+32, g, *]. Ownership is explicit via the def-use chain.
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

---

## 8. Implementation status

Where the rules of §5 stand in the verifier today. Non-normative: this
section records the current state, not an obligation.

The legality pass (`lib/Conversion/ConvertToKTIR/KTIRCheckLegality.cpp`,
182 lines) currently walks only `InterTileProduceOp` and
`InterTileReduceOp`:

| Rule | Op | Check | Location |
|---|---|---|---|
| R2 | `inter_tile_produce` | `future.hasOneUse()` | `KTIRCheckLegality.cpp:80–85` |
| R13 | `inter_tile_reduce` | `C ⊆ P` per group | `KTIRCheckLegality.cpp:107–117` |
| R14 | `inter_tile_reduce` | `C == P` or `\|C\| == 1` | `KTIRCheckLegality.cpp:119–128` |
| R3 | `inter_tile_reduce` | declared dep `p ∈ P(g)` | `KTIRCheckLegality.cpp:151–160` |
| R4 | `inter_tile_reduce` | every `p` covered by some dep | `KTIRCheckLegality.cpp:163–174` |

**Not yet implemented:** R1, R5, R6, R7, R8, R9, R10, R11, R12, and
R3/R4/R13/R14 for every op other than `reduce`. R5 and R7 are enforced in
the Torch-Spyre SDSC planner (`_compatible_partitions`) but are absent
from the KTIR verifier entirely — the gap exists at both the spec and the
implementation level.

**Two asymmetries the §5 matrix forces into the open.**

1. R8 is stated as a verifier obligation for `scatter` but is merely
   conventional for `consume`, even though both ops have the same
   single-producer cardinality. Neither is implemented yet, so this is a
   spec asymmetry to decide rather than inherit.
2. R13 and R14 are implemented for `reduce` only, and R13 is the
   implementation of open question §10.1 (must a consumer also be a
   producer?) for that one op. The `?` cells in the §5 matrix are exactly
   that question, unresolved: for `scatter` the answer is **no** (§6.6), for
   `reduce` the current answer is **yes** (enforced), and for
   `reduce_scatter` / `gather` / `all_to_all` it is undecided. R14's
   mode gate is likewise a current implementation restriction, not a
   design conclusion.

---

## 9. Backend pattern catalogue — non-normative

This section records relayout patterns observed in the Torch-Spyre backend
(`scratchpad/lx_relayout.py`) and how they map onto the ops above. It is
descriptive, not normative: nothing here constrains the op definitions,
and the mapping column is a proposal for lowering rather than a
guarantee. Rows whose split axes are unknown are omitted.

All implemented patterns emit a **single** SDSC `opfunc = "shuffle"` whose
entire payload is two `coreIdToWkSlice_` tables — one per tensor in
`coordinates_` — describing per-core ownership before and after the
movement. Classification uses
`gathered_dims = src_syms − dst_syms`,
`scattered_dims = dst_syms − src_syms`,
`factor = num_cores // prod(dst splits)`.

| ID | SenDNN op | Src `work_div` | Dst view | gathered | scattered | factor | KTIR op | Backend status |
|---|---|---|---|---|---|---|---|---|
| P01 | all-gather | `{H:8, Lk:4}` | replicated (all cores) | H, Lk | — | N/A | `inter_tile_gather` (all consumers) | missing — replication not supported |
| P03 | grouped all-gather | `{H:8, Lk:4}` | `{H:8}` | Lk | — | 4 | `inter_tile_gather` | #3440 (open PR) |
| P04 | grouped all-gather | `{Lk:32}` | `{H:8}` | Lk | H | 4 | `inter_tile_gather` | #3440 (open PR) |
| P06 | all-to-all | `{H:8, Lk:4}` | `{Lk:32}` | H | — | 1 | `inter_tile_all_to_all` | main (#3439) |
| P08 | all-to-all (axis transpose) | `{A:4, B:8}` | `{A:8, B:4}` | — | — | 1 | `inter_tile_all_to_all` + explicit coord map | missing — axis swap inexpressible |
| P14 | all-to-all | `{H:8, Lq:4}` | `{Lq:32}` | H | — | 1 | `inter_tile_all_to_all` | main (#3439) |

- **P03 / P04** show that `gathered_dims ≠ ∅` with `factor > 1` maps to
  `inter_tile_gather`. For P04, `scattered_dims = {H}` (H is introduced at
  the destination) — a combined gather+scatter in one shuffle step, so the
  KTIR op must express both axes.
- **P06 / P14** show `inter_tile_all_to_all` at `factor = 1`:
  `gathered_dims = {H}`, `scattered_dims = ∅`, with H contracted into the
  1-D destination axis.
- **P08 is the case `all_to_all` with dim attributes cannot express.** Both
  sides split the same two dims with swapped counts (`[4,8] → [8,4]`). The
  `coreIdToWkSlice_` tables differ because the mixed-radix odometer
  ordering changes, but no single `scatter_dim`/`gather_dim` pair captures
  the transformation. Marked out of scope for `all_to_all`; a verifier
  should reject non-decomposable transpositions rather than silently
  mis-lower them. §10.4 (Option B) is the eventual fix.
- **P01** (full replication) is blocked because `_compatible_partitions`
  requires distinct slices per destination core. It maps to
  `inter_tile_gather` with `consumer_tiles_per_group = all`, but needs new
  backend support for non-bijective shuffle.

**Classification decision rule** (from `TensorArg.work_division` on a
relayout identity OpSpec):

| `gathered_dims` | `scattered_dims` | `factor` | `slot_exprs_differ` | KTIR op |
|---|---|---|---|---|
| ∅ | ∅ | 1 | false | no-op |
| ∅ | ∅ | 1 | true | `inter_tile_all_to_all` + explicit coord map |
| non-∅ | ∅ or non-∅ | > 1 | — | `inter_tile_gather` |
| non-∅ | ∅ | 1 | — | `inter_tile_all_to_all` |
| ∅ | non-∅ | 1 | — | `inter_tile_scatter` |
| ∅ | ∅ | 1 | — (replication) | `inter_tile_gather` (all consumers) |

**Fused relayout is deferred.** Relayout stays a separate preceding op and
fusion is a lowering concern. The backend structurally cannot fuse them
today: restickify is a separate pass that runs before LX planning, and
restickified weights are explicitly barred as shuffle sources.

---

## 10. Open questions and extensions

### 10.1 Must a consumer also be a producer?

Open for `consume`, `reduce`, `reduce_scatter`, `gather`, and
`all_to_all`; **resolved for `scatter`** — no (§6.6).

The current implementation answers *yes* for `reduce` and enforces it
(R13, `KTIRCheckLegality.cpp:107–117`), whose error text names this
question explicitly. That is one op's implementation choice, not a design
conclusion for the family. The related R14 mode gate — `reduce` supports
all-reduce (`C == P`) and reduce-to-one (`|C| == 1`) but rejects a strict
multi-tile consumer subset — is likewise a present restriction awaiting a
decision.

Deciding this per op is what the `?` cells in §5 record.

### 10.2 Delivery-op placement

Whether the verifier should enforce that a delivery op appears only inside
a guard matching `consumer_tiles_per_group`, or whether that is left to
lowering. If the union of consumer sets equals the set of all executing
tiles, no guard is needed; otherwise a tile outside the consumer set that
reaches the delivery op would be a verifier error.

### 10.3 Multiple delivery ops per future

R2 restricts a `!ktdp.tile_future<...>` value to exactly one delivery use.
A natural extension would allow several delivery ops to consume the same
future, each declaring its own `producer_dependency_per_consumer` — one
`ktdp.inter_tile_produce` serving two independent deliveries, e.g. one
waiting on the first half of the producers and another on the second half.

**Expressiveness gain.** Patterns that today need two separate
`ktdp.inter_tile_produce` ops with identical producer regions collapse to
one produce plus two delivery ops, removing redundant producer-side code
and making the shared production explicit in the IR.

**Verification cost.** Single-use keeps R4 (coverage) local: the verifier
inspects one delivery op to confirm every producer tile is covered. With
multiple uses, coverage becomes global — for every group `g` and producer
`p ∈ producer_tiles_per_group(g)`, at least one consumer `c` across *any*
delivery op must satisfy `producer_dependency_per_consumer(p)[c, g]`. That
requires collecting and unioning the dependency sets from all uses of the
SSA value before checking, a def-use traversal rather than a local per-op
check. R5 (pairwise disjointness) would have to become cross-op too.

**Lowering cost.** Each delivery op declaring a dependency set introduces
its own point-to-point signals. A producer `p` may then need to signal
multiple consumers across different delivery ops, and lowering must emit
each signal exactly once and receive it exactly once per dependent
consumer. In full-barrier mode, multiple delivery ops on one future also
require handling duplicate barrier waits: a producer-side barrier cannot
be issued until every dependent delivery op is ready to receive.

Given that cost, the current design requires separate
`ktdp.inter_tile_produce` ops for separate delivery concerns. If real use
cases demand shared production, the restriction can be relaxed.

### 10.4 Option B — explicit coordinate map

Replace the `scatter_dim`/`gather_dim` attribute pair with a single
source-to-destination affine map, subsuming all four placement values in
one mechanism. This is the shape interface-specs PR 14 already uses
(`SHUFFLE` as source/destination coordinate sets), and the backend already
speaks per-core partitionings (`coreIdToWkSlice_` tables) rather than dim
attributes — so Option B is arguably closer to the existing contract, and
it is the only form that expresses P08 (§9).

**Direction: dim attributes first, Option B recorded as the long-term
target.** The dim-attribute form is additive and reviewable on its own,
and it covers every pattern the backend implements today. Note the honest
counter-argument: because none of the six delivery ops is built yet, the
usual "Option A is cheaper because it is incremental" argument is weaker
here than normal.

### 10.5 Post-v1 extensions to `all_to_all`

- **`all_to_all_v`** — uneven shard sizes, i.e. per-consumer split extents
  instead of a uniform `T_p[scatter_dim] / C`. This relaxes R9 into a
  per-consumer size list and needs a variadic size attribute; no current
  backend pattern requires it.
- **Multi-axis relayout** — splitting or gathering along more than one
  axis in a single op. Expressible today only as a sequence of
  `all_to_all` ops; Option B (§10.4) is the natural home for it.
- **`inter_tile_shuffle` as a naming alias** — the SDSC backend calls the
  primitive `shuffle`. `all_to_all` is kept as the op name because it is
  the established collective term and because `shuffle` is the *lowering*
  of several patterns, not just this one (§9).

---

## Appendix A. Relationship to the pre-existing ops

| Existing op | Maps to in this design |
|-------------|------------------------|
| `inter_tile_produce` | `ktdp.inter_tile_produce` — `consumer_tiles_per_group` moved to the delivery op; `producer_tile_per_group` → `producer_tiles_per_group` (generalized to multi-producer) |
| `inter_tile_consume` | `ktdp.inter_tile_consume` — unchanged semantics |
| `inter_tile_reduce` | `ktdp.inter_tile_produce` + `ktdp.inter_tile_reduce` — producer block removed from the reduction op |
| `inter_tile_reduce_scatter` | `ktdp.inter_tile_produce` + `ktdp.inter_tile_reduce_scatter` — producer block removed from the reduction op |

`ktdp.inter_tile_gather` (§6.4), `ktdp.inter_tile_all_to_all` (§6.5), and
`ktdp.inter_tile_scatter` (§6.6) have no pre-existing counterparts. The
earlier ops offered none of ordered-concatenation delivery (gather),
split-and-reassemble delivery (all-to-all), or single-producer
ordered-partition delivery (scatter).

The `!ktdp.tile_future<T, #groups>` type is shared across all ops; its
`#groups` parameter carries the group set (§1.3).

The previous `ktdp.inter_tile` single op (Approach B draft) is replaced by
this seven-op design. `ktdp.inter_tile` carried producer and optional
combiner regions in one op, with `consumer_tiles_per_group` determining the
delivery mode. Splitting production from delivery makes the mode a choice
of op rather than an inference over attribute combinations — which is what
lets §3 state the shared machinery once and §6 reduce each op to its own
cells.
