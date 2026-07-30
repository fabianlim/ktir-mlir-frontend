//===- KTDPAttrs.h - KTDP dialect attrs public header -----*- C++ -*-===//
//
//===----------------------------------------------------------------------===//
#ifndef KTIR_DIALECT_KTDP_KTDPATTRS_H_
#define KTIR_DIALECT_KTDP_KTDPATTRS_H_

#include "mlir/IR/Attributes.h"
#include "ktir/Dialect/KTDP/KTDPDialect.h"
#include "ktir/Dialect/KTDP/KTDPAttrInterfaces.h.inc"
#include "ktir/Dialect/KTDP/KTDPEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "ktir/Dialect/KTDP/KTDPAttrs.h.inc"

#endif // KTIR_DIALECT_KTDP_KTDPATTRS_H_
