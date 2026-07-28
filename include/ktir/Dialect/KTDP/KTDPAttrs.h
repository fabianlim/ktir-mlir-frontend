//===- KTDPAttrs.h - KTDP dialect attrs public header -----*- C++ -*-===//
//
//===----------------------------------------------------------------------===//
#ifndef KTIR_DIALECT_KTDP_KTDPATTRS_H_
#define KTIR_DIALECT_KTDP_KTDPATTRS_H_

#include "mlir/IR/Attributes.h"
#include "ktir/Dialect/KTDP/KTDPDialect.h"
#include "ktir/Dialect/KTDP/KTDPAttrInterfaces.hpp.inc"
#include "ktir/Dialect/KTDP/KTDPEnums.hpp.inc"
#define GET_ATTRDEF_CLASSES
#include "ktir/Dialect/KTDP/KTDPAttrs.hpp.inc"
#undef GET_ATTRDEF_CLASSES

#endif // KTIR_DIALECT_KTDP_KTDPATTRS_H_
