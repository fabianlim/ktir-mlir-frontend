//===- KtdpAttrs.td - KTDP dialect attribute definitions ---*- tablegen -*-===//
//
// Attribute definitions for the ktdp dialect.
//
// Hierarchy:
//   KtdpMemorySpaceAttr  (abstract base / AttrInterface)
//     +-- SpyreMemorySpaceAttr  (concrete, mnemonic: spyre_memory_space)
//
//===----------------------------------------------------------------------===//

#ifndef KTDP_ATTRS_TD
#define KTDP_ATTRS_TD

include "Ktdp/KtdpDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Memory-space enum
//===----------------------------------------------------------------------===//

def Ktdp_SpyreMemorySpaceKind : I32EnumAttr<
    "SpyreMemorySpaceKind",
    "Spyre memory-space enum",
    [
      I32EnumAttrCase<"unspecified", 0, "unspecified">,
      I32EnumAttrCase<"LX",  1, "LX">,
      I32EnumAttrCase<"HBM", 2, "HBM">
    ]> {
  let cppNamespace = "::mlir::ktdp";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// KtdpMemorySpaceAttr -- abstract base (AttrInterface)
//===----------------------------------------------------------------------===//

def Ktdp_MemorySpaceAttr : AttrInterface<"KtdpMemorySpaceAttr"> {
  let cppNamespace = "::mlir::ktdp";
  let description = [{
    Abstract base interface for all device-specific memory-space attributes
    within the `ktdp` dialect.  Concrete implementations (e.g.
    `SpyreMemorySpaceAttr`) implement this interface.
  }];

  // TODO: Add query methods (e.g. getMemorySpaceKind()) once a second
  // concrete implementation is introduced and the common API is clear.
  let methods = [];
}

//===----------------------------------------------------------------------===//
// SpyreMemorySpaceAttr -- concrete Spyre target memory-space attribute
//
// Assembly syntax:
//   #ktdp.spyre_memory_space<HBM>
//   #ktdp.spyre_memory_space<LX, core = 7>
//   #ktdp.spyre_memory_space<unspecified>
//
// Parameters:
//   value  -- SpyreMemorySpaceKind enum (unspecified | LX | HBM)
//   core   -- optional core affinity (default -1 = unspecified)
//===----------------------------------------------------------------------===//

def Ktdp_SpyreMemorySpaceAttr
    : AttrDef<Ktdp_Dialect, "SpyreMemorySpace",
              [DeclareAttrInterfaceMethods<Ktdp_MemorySpaceAttr>]> {
  let mnemonic = "spyre_memory_space";
  let summary = "Spyre target memory-space descriptor";
  let description = [{
    Identifies a physical memory space on the Spyre target.

    Valid values:
    - `unspecified` -- memory space not yet determined.
    - `LX`  -- shared on-chip scratchpad (cross-core accessible).
    - `HBM` -- high-bandwidth off-chip memory.

    The optional `core` parameter identifies which compute core's local
    memory is referenced (relevant for `LX`).

    Examples:
    ```mlir
    memory_space = #ktdp.spyre_memory_space<HBM>
    memory_space = #ktdp.spyre_memory_space<LX, core = 7>
    ```
  }];

  let parameters = (ins
    "::mlir::ktdp::SpyreMemorySpaceKind":$value,
    DefaultValuedParameter<"int32_t", "-1">:$core
  );

  let assemblyFormat = "`<` $value (`,` `core` `=` $core^)? `>`";

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Return true if a core affinity is specified.
    bool hasCoreSpecified() const { return getCore() != -1; }
  }];
}

//===----------------------------------------------------------------------===//
// Reduce-related attributes for ktdp.reduce
//===----------------------------------------------------------------------===//
//
// First cut: just enough surface for the split-K reduce-sum kernel.
//
//   - ReduceKindAttr  : enum of recognized combiner shorthands. Today
//                       only `sum`. The enum is a *frontend ergonomics*
//                       knob, NOT a capability claim — the region form
//                       on `ktdp.reduce` accepts any combiner body.
//
//   - ReduceModeAttr  : where the result lands. Today: `all_reduce`
//                       (replicated) and `reduce_to_core<N>` (only on
//                       core rank N). Parametric so `reduce_to_core<N>`
//                       carries its destination core inline.
//
//   - GridAxisAttr    : which grid axis the reduction collapses,
//                       indexed into the kernel's `grid = [...]` func
//                       attribute (treated as opaque here).
//
// Design note — AttrInterface adoption (deferred, by convention).
//
//   `SpyreMemorySpaceAttr` above adopts an abstract base interface
//   `Ktdp_MemorySpaceAttr` (an `AttrInterface`) that has no methods
//   yet — it exists so future concrete memory-space attrs (other
//   targets) can be grouped under a common umbrella. Per the comment
//   on that interface, the convention in this file is to introduce
//   such an interface only once a *second* concrete implementation
//   exists and the shared API has become clear. With only a single
//   concrete reduce-mode attr today (`ReduceModeAttr`), adding an
//   interface now would be premature by that same standard.

// ReduceKindAttr: a parametric AttrDef wrapping the ReduceKind enum.
// Modeled as an AttrDef (not EnumAttr) so the attribute is owned by
// the ktdp dialect and parses with the `#ktdp.reduce_kind<...>`
// syntax. The set of entries is *frontend ergonomics*, not a list of
// supported reductions: the region form on `ktdp.reduce` accepts
// arbitrary combiner bodies regardless of what is enumerated here.
// Whether a particular body lowers to a hardware collective is a
// lowering-pass concern.
def Ktdp_ReduceKind : I32EnumAttr<
    "ReduceKind",
    "Recognized combiner-shorthand kinds for ktdp.reduce",
    [
      I32EnumAttrCase<"sum", 0, "sum">
      // TODO: max, min, prod, ... — add when sugar spelling is needed.
    ]> {
  let cppNamespace = "::mlir::ktdp";
  let genSpecializedAttr = 0;
}

def Ktdp_ReduceKindAttr
    : AttrDef<Ktdp_Dialect, "ReduceKind"> {
  let mnemonic = "reduce_kind";
  let summary = "Recognized combiner shorthand for ktdp.reduce";
  let description = [{
    Names a recognized combiner shorthand for `ktdp.reduce` sugar
    form. Today only `sum` is recognized.

    Example:
    ```mlir
    kind = #ktdp.reduce_kind<sum>
    ```
  }];

  let parameters = (ins "::mlir::ktdp::ReduceKind":$value);
  let assemblyFormat = "`<` $value `>`";
}

def Ktdp_ReduceModeKind : I32EnumAttr<
    "ReduceModeKind",
    "Result-placement modes for ktdp.reduce",
    [
      I32EnumAttrCase<"all_reduce",     0, "all_reduce">,
      I32EnumAttrCase<"reduce_to_core", 1, "reduce_to_core">
    ]> {
  let cppNamespace = "::mlir::ktdp";
  let genSpecializedAttr = 0;
}

def Ktdp_ReduceModeAttr
    : AttrDef<Ktdp_Dialect, "ReduceMode"> {
  let mnemonic = "reduce_mode";
  let summary = "Result-placement mode for ktdp.reduce";
  let description = [{
    Describes where the reduced value lands relative to the
    participating cores.

    Variants:
    - `all_reduce` -- result is replicated on every core in the
      reduction group.
    - `reduce_to_core<N>` -- result is materialized only on core
      rank `N` within the group; on other cores the result value is
      unspecified / poison.

    Examples:
    ```mlir
    mode = #ktdp.reduce_mode<all_reduce>
    mode = #ktdp.reduce_mode<reduce_to_core<0>>
    ```
  }];

  let parameters = (ins
    "::mlir::ktdp::ReduceModeKind":$kind,
    DefaultValuedParameter<"int32_t", "-1">:$dst
  );

  // Custom assembly: `<all_reduce>` or `<reduce_to_core<0>>`.
  let hasCustomAssemblyFormat = 1;

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// True if a `dst` core rank is encoded.
    bool hasDst() const { return getDst() != -1; }
  }];
}

def Ktdp_GridAxisAttr
    : AttrDef<Ktdp_Dialect, "GridAxis"> {
  let mnemonic = "grid_axis";
  let summary = "Names a grid axis by position in the kernel `grid` attribute";
  let description = [{
    A non-negative integer naming a grid axis by its position in the
    kernel's `grid = [...]` `func.func` attribute. Used as the
    `across` attribute on `ktdp.reduce` to identify the grid axis the
    reduction collapses.

    Example:
    ```mlir
    across = #ktdp.grid_axis<0>
    ```
  }];

  let parameters = (ins "int32_t":$axis);

  let assemblyFormat = "`<` $axis `>`";

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Runtime argument extraction kind
//===----------------------------------------------------------------------===//

def KTDP_RuntimeArgExtractKind_Value : I32EnumAttrCase<"value", 0>;
def KTDP_RuntimeArgExtractKind_Granularity : I32EnumAttrCase<"granularity", 1>;
def KTDP_RuntimeArgExtractKind_Upperbound : I32EnumAttrCase<"upperbound", 2>;

def KTDP_RuntimeArgExtractKind : I32EnumAttr<"RuntimeArgExtractKind",
    "Kind of extraction from runtime_arg type",
    [
       KTDP_RuntimeArgExtractKind_Value,
       KTDP_RuntimeArgExtractKind_Granularity,
       KTDP_RuntimeArgExtractKind_Upperbound
    ]> {
  let cppNamespace = "::mlir::ktdp";
}

#endif // KTDP_ATTRS_TD
