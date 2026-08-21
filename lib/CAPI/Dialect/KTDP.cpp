//===-- KTDP.cpp ------------------------------------------------*- c++ -*-===//
//
// Copyright 2026 The KTIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "ktir-c/Dialect/KTDP.h"

#include "ktir/Conversion/Passes.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "ktir/Dialect/KTDP/KTDPDialect.h"
#include "ktir/Dialect/KTDP/KTDPTypes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Ktdp, ktdp,
                                      mlir::ktdp::KtdpDialect)

MlirType mlirKTDPAccessTileTypeGet(intptr_t rank, int64_t *shape, MlirType elementType) {
  return wrap(mlir::ktdp::AccessTileType::get(llvm::ArrayRef(shape, static_cast<size_t>(rank)), unwrap(elementType)));
}

bool mlirTypeIsAKTDPAccessTileType(MlirType t) {
  return llvm::isa<mlir::ktdp::AccessTileType>(unwrap(t));
}

MlirTypeID mlirKTDPAccessTileTypeGetTypeID() {
  return wrap(mlir::ktdp::AccessTileType::getTypeID());
}

MlirType mlirKTDPRuntimeArgTypeGet(MlirContext ctx, MlirType underlyingType,
                                    int64_t granularity, int64_t upperbound) {
  std::optional<int64_t> gran = granularity >= 0 ? std::optional<int64_t>(granularity) : std::nullopt;
  std::optional<int64_t> ub = upperbound >= 0 ? std::optional<int64_t>(upperbound) : std::nullopt;
  return wrap(mlir::ktdp::RuntimeArgType::get(unwrap(ctx), unwrap(underlyingType), gran, ub));
}

bool mlirTypeIsAKTDPRuntimeArgType(MlirType t) {
  return llvm::isa<mlir::ktdp::RuntimeArgType>(unwrap(t));
}

MlirTypeID mlirKTDPRuntimeArgTypeGetTypeID() {
  return wrap(mlir::ktdp::RuntimeArgType::getTypeID());
}

MlirType mlirKTDPRuntimeArgTypeGetUnderlyingType(MlirType t) {
  return wrap(llvm::cast<mlir::ktdp::RuntimeArgType>(unwrap(t)).getUnderlyingType());
}

int64_t mlirKTDPRuntimeArgTypeGetGranularity(MlirType t) {
  auto v = llvm::cast<mlir::ktdp::RuntimeArgType>(unwrap(t)).getGranularity();
  return v.has_value() ? v.value() : -1;
}

int64_t mlirKTDPRuntimeArgTypeGetUpperbound(MlirType t) {
  auto v = llvm::cast<mlir::ktdp::RuntimeArgType>(unwrap(t)).getUpperbound();
  return v.has_value() ? v.value() : -1;
}

//===----------------------------------------------------------------------===//
// MemorySpaceAttr
//===----------------------------------------------------------------------===//

static mlir::ktdp::MemorySpaceKind
unwrapMemorySpaceKind(MlirKTDPMemorySpaceKind kind) {
  switch (kind) {
  case MlirKTDPMemorySpaceKindGlobal:
    return mlir::ktdp::MemorySpaceKind::global;
  case MlirKTDPMemorySpaceKindCTLocal:
    return mlir::ktdp::MemorySpaceKind::ct_local;
  }
  llvm_unreachable("unhandled MlirKTDPMemorySpaceKind");
}

static MlirKTDPMemorySpaceKind
wrapMemorySpaceKind(mlir::ktdp::MemorySpaceKind kind) {
  switch (kind) {
  case mlir::ktdp::MemorySpaceKind::global:
    return MlirKTDPMemorySpaceKindGlobal;
  case mlir::ktdp::MemorySpaceKind::ct_local:
    return MlirKTDPMemorySpaceKindCTLocal;
  }
  llvm_unreachable("unhandled mlir::ktdp::MemorySpaceKind");
}

MlirAttribute mlirKTDPMemorySpaceAttrGet(MlirContext ctx,
                                         MlirKTDPMemorySpaceKind kind,
                                         int32_t ctId) {
  // getChecked so an invalid combination (e.g. ct_id on global) surfaces as a
  // diagnostic and a null attribute rather than tripping an assertion.
  return wrap(mlir::ktdp::MemorySpaceAttr::getChecked(
      mlir::UnknownLoc::get(unwrap(ctx)), unwrap(ctx),
      unwrapMemorySpaceKind(kind), ctId));
}

bool mlirAttributeIsAKTDPMemorySpaceAttr(MlirAttribute attr) {
  return llvm::isa<mlir::ktdp::MemorySpaceAttr>(unwrap(attr));
}

MlirTypeID mlirKTDPMemorySpaceAttrGetTypeID() {
  return wrap(mlir::ktdp::MemorySpaceAttr::getTypeID());
}

MlirKTDPMemorySpaceKind mlirKTDPMemorySpaceAttrGetKind(MlirAttribute attr) {
  return wrapMemorySpaceKind(
      llvm::cast<mlir::ktdp::MemorySpaceAttr>(unwrap(attr)).getKind());
}

int32_t mlirKTDPMemorySpaceAttrGetCtId(MlirAttribute attr) {
  return llvm::cast<mlir::ktdp::MemorySpaceAttr>(unwrap(attr)).getCtId();
}

void mlirKTIRRegisterPasses(void) { ktir::registerKTIRConversionPasses(); }
