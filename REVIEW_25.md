# Review Notes — PR #25 (KFAFSP, review 4737120069)

Comments are listed in file order, numbered to match the GitHub UI.

---

## `include/Ktdp/KtdpOps.td`

### C1 — `AnyTensor` → `AnyRankedTensor` on yield ops (L217)

> "Can this really be `AnyTensor`? It seems to me that the code assumes its
> `AnyRankedTensor`. Same goes for the result type. This would also make some
> casts redundant."

**Current code:**
```tablegen
// YieldPartialOp
let arguments = (ins Variadic<AnyTensor>:$values);
// YieldReducedOp
let arguments = (ins Variadic<AnyTensor>:$values);
// InterTileReduceOp
let results = (outs Variadic<AnyTensor>:$results);
```

**Issue:** The constraint is too broad. The verifier and all downstream code
assume ranked tensors (`RankedTensorType`). Using `AnyTensor` allows unranked
tensors through the type system, which would then crash at verifier casts.

**Resolution:** Change `AnyTensor` → `AnyRankedTensor` in all three places.
This also eliminates several `cast<RankedTensorType>(...)` calls in the
verifier that become redundant once the type system encodes the constraint.

---

### C2 — Custom assembly format may be unnecessary on `inter_tile_produce` (L223)

> "I don't see why this needs a custom assembly format instead of a declarative
> one."

**Current code:**
```tablegen
let hasCustomAssemblyFormat = 1;
```

**Issue:** The op currently has a custom C++ parser/printer partly to print the
partial types before the arrow (`tensor<...> -> !ktdp.tile_future<...>`). But
those partial types are dead (see C7/C8 below). Once they are removed, the
remaining syntax may be expressible declaratively.

**Resolution:** After fixing C7/C8 (removing dead partial types from the
syntax), attempt to replace the custom format with a declarative
`let assemblyFormat = "..."`. If the region printing or attr-dict ordering
makes this impractical, keep custom but document why.

---

### C3 — Redundant cast in `getPartialTypes()` helper (L229)

> "This cast is redundant because `getFuture()` should already be a
> `TypedValue<TileFutureType>`."

**Current code (`InterTileProduceOp::extraClassDeclaration`):**
```cpp
::llvm::ArrayRef<::mlir::Type> getPartialTypes() {
  return ::llvm::cast<TileFutureType>(getFuture().getType())
      .getPartialTypes();
}
```

**Issue:** Because the result is declared as `Ktdp_TileFutureType:$future`,
MLIR generates `getFuture()` as `TypedValue<TileFutureType>`, so `.getType()`
already returns `TileFutureType` — the `cast<>` is a no-op.

**Resolution:**
```cpp
::llvm::ArrayRef<::mlir::Type> getPartialTypes() {
  return getFuture().getType().getPartialTypes();
}
```

---

### C4 — Same redundant cast in `getGroups()` helper (L234)

> "Same as above."

**Current code:**
```cpp
::mlir::IntegerSet getGroups() {
  return ::llvm::cast<TileFutureType>(getFuture().getType()).getGroups();
}
```

**Resolution:** Same as C3:
```cpp
::mlir::IntegerSet getGroups() {
  return getFuture().getType().getGroups();
}
```

Applies to both `InterTileProduceOp` and `InterTileReduceOp`.

---

## `include/Ktdp/KtdpTypes.td`

### C5 — Look-ahead required by `tile_future` type syntax (L222)

> "nit: I would prefer it if this had a declarative assembly format, but I admit
> that it is impossible with the current syntax. None of the available list
> parsers have look-ahead and thus can't detect `groups` to end the list.
> Could we choose a look-ahead free format instead?"

**Current syntax:**
```
tile_future<tensor<1x64xf16>, tensor<1x64xi32>, groups = affine_set<...>>
```

**Issue:** To parse the type list, the parser must look ahead past each comma to
decide whether the next token is a type or the `groups` keyword. MLIR's
declarative format list parsers do not support this, so a purely declarative
format is impossible.

**Resolution:** Use a look-ahead-free separator, for example wrapping the types
in parentheses:
```
tile_future<(tensor<1x64xf16>, tensor<1x64xi32>), groups = affine_set<...>>
```
This allows a declarative format: the types are in a delimited list `(...)` and
`groups` follows unambiguously. Requires updating `parse`, `print`, all tests,
and the description.

---

## `lib/Ktdp/KtdpDialect.cpp`

### C6 — `addTypes<>` should use `GET_TYPEDEF_LIST` (L43)

> "This should use `#define GET_TYPEDEF_LIST` with the generated include."

**Current code:**
```cpp
addTypes<
  AccessTileType,
  RuntimeArgType,
  TileFutureType
>();
```

**Issue:** MLIR's standard pattern for type registration uses the tablegen-
generated `GET_TYPEDEF_LIST` macro so that types don't need to be maintained
in two places.

**Resolution:**
```cpp
addTypes<
#define GET_TYPEDEF_LIST
#include "Ktdp/KtdpOps.cpp.inc"
>();
```
(The exact include file depends on whether types are generated into `KtdpTypes.cpp.inc`
or `KtdpOps.cpp.inc` — confirm against the tablegen output.)

---

## `lib/Ktdp/KtdpOps.cpp`

### C7 — Dead partial types in `InterTileProduceOp` parser (L1335)

> "These don't do anything. Also, they are already in the future type."

**Current code:**
```cpp
SmallVector<Type> partialTypes;
Type futureType;
if (parser.parseColonTypeList(partialTypes) || parser.parseArrow() ||
    parser.parseType(futureType))
  return failure();
result.addTypes(futureType);
```

**Issue:** `partialTypes` is parsed but never used — the partial types are
already embedded in the `futureType` that follows the arrow. The syntax
`tensor<1x64xf16> -> !ktdp.tile_future<tensor<1x64xf16>, groups = #g>`
duplicates information.

**Resolution:** Remove `partialTypes` from the syntax entirely. The new syntax
becomes just `-> !ktdp.tile_future<..., groups = #g>`:
```cpp
Type futureType;
if (parser.parseArrow() || parser.parseType(futureType))
  return failure();
result.addTypes(futureType);
```

---

### C8 — Dead partial types in `InterTileProduceOp` printer (L1354)

> "These don't do anything."

**Current code:**
```cpp
p << " : ";
llvm::interleaveComma(getPartialTypes(), p);
p << " -> " << getFuture().getType() << ' ';
```

**Issue:** Same as C7 — printing the partial types before `->` is redundant.

**Resolution:** Remove the partial types from the output:
```cpp
p << " -> " << getFuture().getType() << ' ';
```

---

### C9 — Redundant cast in `InterTileProduceOp::verify` (L1361)

> "Again, this should be a redundant cast."

**Current code:**
```cpp
auto futureType = cast<TileFutureType>(getFuture().getType());
```

**Resolution:**
```cpp
TileFutureType futureType = getFuture().getType();
```

---

### C10 — `InterTileProduceOp` verifier accesses region (L1366)

> "nit: This is technically not a legal verifier according to MLIR
> specifications, but I understand we have a few of these around anyways."

**Issue:** `hasVerifier = 1` is for verifying the op's attributes/types in
isolation. Accessing `getBody().front()` is accessing a nested region, which
technically requires `verifyWithRegions = 1`.

**Resolution:** Change `let hasVerifier = 1` → `let verifyWithRegions = 1` on
`InterTileProduceOp`. The reviewer acknowledges this is a known pattern in the
codebase, so it is a nit rather than a hard blocker.

---

### C11 — `InterTileProduceOp` verifier accesses block args (L1377)

> "Accessing a nested operation must go in a region verifier
> (`let verifyWithRegions = 1;`)."

**Current code:**
```cpp
Block& block = getBody().front();
// ... checks block.getArguments(), block.getTerminator() ...
```

**Issue:** Accessing `block.getTerminator()` reaches into a nested op. This is
only legal in a region verifier.

**Resolution:** Same as C10 — switch to `verifyWithRegions = 1`.

---

### C12 — `operandSegmentSizes` hardcoded without trait (L1492)

> "This operation does not implement the `AttrSizedOperandSegments` trait, and
> it does not need to. This should go. Otherwise, and only if the custom
> printer/parser can't be removed for some reason, do not use the hard-coded
> name. Use the `getOperandSegmentSizesAttrName()` provided by the trait."

**Current code:**
```cpp
result.addAttribute(
    "operandSegmentSizes",
    parser.getBuilder().getDenseI32ArrayAttr(
        {1, static_cast<int32_t>(identityOperands.size())}));
```

**Issue:** `operandSegmentSizes` is an attribute managed by the
`AttrSizedOperandSegments` trait. `InterTileReduceOp` does not declare this
trait, so the attribute name is hardcoded and fragile.

**Resolution:** Either:
- Add `AttrSizedOperandSegments` to the op's traits and use
  `getOperandSegmentSizesAttrName()`, or
- Remove the custom parser/printer and let MLIR infer operand segments
  declaratively (preferred if C2's declarative format approach works here too).

---

### C13 — Result shape check could use `RangedTypesMatchWith` (L1529)

> "nit: You can replace this entire block with a `RangedTypesMatchWith` in
> ODS."

**Current code:** ~25 lines of C++ in `InterTileReduceOp::verify` validating
that each result type is the partial type with one axis collapsed.

**Issue:** This structural type relationship can be expressed declaratively in
ODS using the `RangedTypesMatchWith` constraint, removing the C++ verification
entirely.

**Resolution:** Add a `RangedTypesMatchWith` constraint to `InterTileReduceOp`
in `KtdpOps.td` that expresses the rank-reduction relationship between
`$future`'s partial types and `$results`. This is a refactor — the verifier
logic is correct, just not in the right form.

---

### C14 — `InterTileReduceOp` verifier accesses combiner region (L1556)

> "Accessing a nested operation must go in a region verifier
> (`let verifyWithRegions = 1;`)."

**Current code:**
```cpp
Block& block = getCombiner().front();
// ... checks block args and terminator ...
```

**Resolution:** Same as C10/C11 — change `let hasVerifier = 1` →
`let verifyWithRegions = 1` on `InterTileReduceOp`.

---

### C15 — Cross-op checks not legal in a verifier (L1600)

> "nit: The checks past this point are not legal in a verifier according to the
> MLIR specification."

**Current code:**
```cpp
auto produceOp = getFuture().getDefiningOp<InterTileProduceOp>();
if (produceOp) {
  // ... compares producer_tiles_per_group against consumer_tiles_per_group ...
}
```

**Issue:** Calling `getDefiningOp()` on an operand reaches outside the current
op, which is not allowed in a verifier per the MLIR spec. Verifiers must only
inspect the op itself in isolation.

**Resolution:** Move the cross-op checks (producer/consumer set relationship,
disjointness, subset, mode gate) into a dedicated pass that runs after
construction. The verifier retains only the checks that are local to each op.
This is the most significant structural change required.

---

## `lib/Ktdp/KtdpTypes.cpp`

### C16 — Runtime ranked-tensor check should be declarative (L166)

> "Instead of this verifier, the type parameter should require
> `AnyRankedTensor`."

**Current code:**
```cpp
for (Type t : partialTypes) {
  if (!mlir::isa<RankedTensorType>(t))
    return emitError() << "tile_future partial type must be a ranked tensor...";
}
```

**Issue:** This is a runtime check on a type parameter that should be expressed
as a type constraint in the `.td` definition. The `ArrayRefParameter` should
use a constrained element type so tablegen enforces it structurally.

**Resolution:** In `KtdpTypes.td`, change the `partialTypes` parameter to use
a constrained array parameter (e.g. `TypeArrayRefParameter<"::mlir::RankedTensorType">` 
or an equivalent that restricts elements to ranked tensors). Remove the loop
from `TileFutureType::verify`.

---

## `README.md`

### C17 — Broken file reference (L23)

> "That file doesn't exist?"

**Current code:**
```
[docs/inter-tile-communication.md](docs/inter-tile-communication.md)
```

**Issue:** The file `docs/inter-tile-communication.md` does not exist in the
repo (it lives in PR #23 which is not yet merged).

**Resolution:** Either remove the link until PR #23 merges, or note it as
"coming soon" in the PR description.

---

## Priority summary

| # | File | One-liner | Priority |
|---|------|-----------|----------|
| C1 | KtdpOps.td:217 | `AnyTensor` → `AnyRankedTensor` on yield ops and results | **High** |
| C7 | KtdpOps.cpp:1335 | Remove dead partial types from produce parser | **High** |
| C8 | KtdpOps.cpp:1354 | Remove dead partial types from produce printer | **High** |
| C12 | KtdpOps.cpp:1492 | Remove/fix hardcoded `operandSegmentSizes` | **High** |
| C14 | KtdpOps.cpp:1556 | Reduce verifier accesses region → `verifyWithRegions` | **High** |
| C15 | KtdpOps.cpp:1600 | Cross-op checks not legal in verifier → move to pass | **High** |
| C6 | KtdpDialect.cpp:43 | Use `GET_TYPEDEF_LIST` macro | Medium |
| C10 | KtdpOps.cpp:1366 | Produce verifier accesses region → `verifyWithRegions` | Medium |

---

## Proposed implementation order

Dependencies drive the order: syntax changes first (so later commits land on
stable IR), then type constraints, then verifier restructuring, then cleanup.

### Step 1 — C7 + C8: Remove dead partial types from produce syntax
The `: tensor<...>` before `->` in `inter_tile_produce` is redundant now that
`tile_future` carries the types internally. Remove from both parser and printer.
This is a syntax-breaking change and must come first so all subsequent work
sees the final IR shape.

### Step 2 — C2: Try declarative format on `inter_tile_produce`
After C7/C8 simplify the syntax, attempt to replace `hasCustomAssemblyFormat`
with a declarative `assemblyFormat`. If the region printing prevents it, keep
custom and document why.

### Step 3 — C5: Choose look-ahead-free syntax for `tile_future` type
Change `tile_future<T1, T2, groups = #g>` to a syntax that does not require
look-ahead (e.g. wrapping types in parens: `tile_future<(T1, T2), groups = #g>`).
Must be done before C16 so the type constraint work lands on the final syntax.
Requires updating parse/print and all tests.

### Step 4 — C16 + C1: Declarative type constraints
C16: change the `partialTypes` parameter in `KtdpTypes.td` to use a constrained
array type (e.g. `TypeArrayRefParameter<"::mlir::RankedTensorType">`) and remove
the runtime loop from `TileFutureType::verify`.
C1: change `AnyTensor` → `AnyRankedTensor` on `YieldPartialOp`, `YieldReducedOp`,
and the `InterTileReduceOp` results. Do together as one commit since they are
the same idea in different places.

### Step 5 — C6: Use `GET_TYPEDEF_LIST` in dialect registration
Mechanical change to `KtdpDialect.cpp`, fully independent. One commit.

### Step 6 — C3 + C4 + C9: Remove redundant casts
All three are the same pattern (`cast<TileFutureType>(getFuture().getType())`
→ `getFuture().getType()`). One commit, touches `KtdpOps.td` and
`KtdpOps.cpp`.

### Step 7 — C10 + C11 + C14: `hasVerifier` → `verifyWithRegions`
Change both ops from `let hasVerifier = 1` to `let verifyWithRegions = 1`.
Mechanical but must be done before C15 so the verifier boundary is correct
before the cross-op logic is extracted.

### Step 8 — C15: Move cross-op checks out of verifier
The producer/consumer set relationship checks (disjointness, subset, mode gate)
that call `getDefiningOp()` must move to a dedicated pass. This is the largest
change. The verifier retains only local structural checks.

### Step 9 — C12: Fix `operandSegmentSizes` on reduce op
Either add `AttrSizedOperandSegments` trait and use
`getOperandSegmentSizesAttrName()`, or remove the custom parser entirely if C2's
declarative format work extends to the reduce op too.

### Step 10 — C13: Replace result shape check with `RangedTypesMatchWith`
Pure refactor of correct logic — move the rank-reduction type check from C++
into an ODS `RangedTypesMatchWith` constraint. Last because it depends on C1
(types are now `AnyRankedTensor`) and C8 (syntax is stable).

### Step 11 — C17: Fix broken README link
One-liner, bundle into the last commit or fix independently.
| C11 | KtdpOps.cpp:1377 | Produce verifier accesses block → `verifyWithRegions` | Medium |
| C16 | KtdpTypes.cpp:166 | Move ranked-tensor check to declarative type constraint | Medium |
| C2 | KtdpOps.td:223 | Replace custom produce format with declarative | Nit |
| C3 | KtdpOps.td:229 | Remove redundant cast in `getPartialTypes()` | Nit |
| C4 | KtdpOps.td:234 | Remove redundant cast in `getGroups()` | Nit |
| C5 | KtdpTypes.td:222 | Choose look-ahead-free `tile_future` type syntax | Nit |
| C9 | KtdpOps.cpp:1361 | Remove redundant cast in produce verify | Nit |
| C13 | KtdpOps.cpp:1529 | Replace result shape check with `RangedTypesMatchWith` | Nit |
| C17 | README.md:23 | Fix broken doc link | Nit |
