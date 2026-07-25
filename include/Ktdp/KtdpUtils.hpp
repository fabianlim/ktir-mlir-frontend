//===- KtdpUtils.hpp - KTDP dialect utilities -------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//
//
// Fills gaps that MLIR provides for AffineMap (FieldParser<AffineMap> in
// DialectImplementation.h) but not for IntegerSet.  Both helpers must be
// visible before any generated .inc that uses assemblyFormat with an
// "IntegerSet" parameter.
//
//===----------------------------------------------------------------------===//
#ifndef KTDP_KTDPUTILS_HPP_
#define KTDP_KTDPUTILS_HPP_

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IntegerSet.h"

namespace mlir {

template <>
struct FieldParser<IntegerSet> {
  static FailureOr<IntegerSet> parse(AsmParser &parser) {
    IntegerSet set;
    if (failed(parser.parseIntegerSet(set)))
      return failure();
    return set;
  }
};

} // namespace mlir

#endif // KTDP_KTDPUTILS_HPP_
