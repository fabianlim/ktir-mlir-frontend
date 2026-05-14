//===- KtdpAttrs.cpp - KTDP dialect attr implementations ------------------===//
//
//===----------------------------------------------------------------------===//

#include "Ktdp/KtdpAttrs.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::ktdp;

//===----------------------------------------------------------------------===//
// SpyreMemorySpaceAttr::verify
//===----------------------------------------------------------------------===//

LogicalResult SpyreMemorySpaceAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    SpyreMemorySpaceKind value, int32_t core) {
  // Core affinity is only meaningful for core-local memory spaces.
  if (core == -1)
    return success();

  if (core < 0)
    return emitError() << "core affinity must be non-negative, but got: "
                       << core;

  if (value != SpyreMemorySpaceKind::LX) {
    return emitError()
           << "core affinity is only valid for LX memory spaces, "
              "but got memory space '"
           << stringifySpyreMemorySpaceKind(value) << "' with core = " << core;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ReduceModeAttr
//===----------------------------------------------------------------------===//

LogicalResult ReduceModeAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    ReduceModeKind kind, int32_t dst) {
  switch (kind) {
  case ReduceModeKind::all_reduce:
    if (dst != -1)
      return emitError() << "reduce_mode<all_reduce> must not carry a `dst` "
                            "parameter, but got dst = " << dst;
    return success();
  case ReduceModeKind::reduce_to_core:
    if (dst < 0)
      return emitError() << "reduce_mode<reduce_to_core<...>> requires a "
                            "non-negative `dst` core rank, but got dst = " << dst;
    return success();
  }
  return emitError() << "unknown reduce_mode kind";
}

// Assembly:
//   <all_reduce>
//   <reduce_to_core<0>>
Attribute ReduceModeAttr::parse(AsmParser &parser, Type /*odsType*/) {
  if (parser.parseLess())
    return {};

  StringRef kindStr;
  if (parser.parseKeyword(&kindStr))
    return {};

  std::optional<ReduceModeKind> kind = symbolizeReduceModeKind(kindStr);
  if (!kind) {
    parser.emitError(parser.getCurrentLocation())
        << "expected one of 'all_reduce' or 'reduce_to_core', got '"
        << kindStr << "'";
    return {};
  }

  int32_t dst = -1;
  if (*kind == ReduceModeKind::reduce_to_core) {
    if (parser.parseOptionalLess()) {
      parser.emitError(parser.getCurrentLocation())
          << "reduce_to_core requires a destination core rank: "
             "expected `reduce_to_core<N>`";
      return {};
    }
    int32_t parsedDst = 0;
    if (parser.parseInteger(parsedDst) || parser.parseGreater())
      return {};
    dst = parsedDst;
  }

  if (parser.parseGreater())
    return {};

  return parser.getChecked<ReduceModeAttr>(parser.getContext(), *kind, dst);
}

void ReduceModeAttr::print(AsmPrinter &printer) const {
  printer << "<" << stringifyReduceModeKind(getKind());
  if (getKind() == ReduceModeKind::reduce_to_core)
    printer << "<" << getDst() << ">";
  printer << ">";
}

//===----------------------------------------------------------------------===//
// GridAxisAttr
//===----------------------------------------------------------------------===//

LogicalResult GridAxisAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, int32_t axis) {
  if (axis < 0)
    return emitError() << "grid_axis must be non-negative, but got: " << axis;
  return success();
}
