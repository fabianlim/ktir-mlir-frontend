//===-- KTDP.h --------------------------------------------------*- c++ -*-===//
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

#ifndef KTIR_C_DIALECT_KTDP_H_
#define KTIR_C_DIALECT_KTDP_H_

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Ktdp, ktdp);

MLIR_CAPI_EXPORTED MlirType mlirKTDPAccessTileTypeGet(intptr_t rank,
                                                      int64_t *shape,
                                                      MlirType elementType);

MLIR_CAPI_EXPORTED bool mlirTypeIsAKTDPAccessTileType(MlirType t);

MLIR_CAPI_EXPORTED MlirTypeID mlirKTDPAccessTileTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirKTDPRuntimeArgTypeGet(MlirContext ctx,
                                                       MlirType underlyingType,
                                                       int64_t granularity,
                                                       int64_t upperbound);

MLIR_CAPI_EXPORTED bool mlirTypeIsAKTDPRuntimeArgType(MlirType t);

MLIR_CAPI_EXPORTED MlirTypeID mlirKTDPRuntimeArgTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirKTDPRuntimeArgTypeGetUnderlyingType(MlirType t);

/// Returns the granularity, or -1 if not set.
MLIR_CAPI_EXPORTED int64_t mlirKTDPRuntimeArgTypeGetGranularity(MlirType t);

/// Returns the upperbound, or -1 if not set.
MLIR_CAPI_EXPORTED int64_t mlirKTDPRuntimeArgTypeGetUpperbound(MlirType t);

//===----------------------------------------------------------------------===//
// MemorySpaceAttr
//===----------------------------------------------------------------------===//

/// Values must stay in sync with `mlir::ktdp::MemorySpaceKind` (KTDPAttrs.td).
enum MlirKTDPMemorySpaceKind {
  MlirKTDPMemorySpaceKindGlobal = 0,
  MlirKTDPMemorySpaceKindCTLocal = 1,
};

/// Builds a `#ktdp.memory_space<...>` attribute. Pass -1 for `ctId` to leave
/// the compute-tile ID unspecified. Emits a diagnostic and returns a null
/// attribute if the parameters fail verification (e.g. a `ctId` on `global`).
MLIR_CAPI_EXPORTED MlirAttribute
mlirKTDPMemorySpaceAttrGet(MlirContext ctx, enum MlirKTDPMemorySpaceKind kind,
                           int32_t ctId);

MLIR_CAPI_EXPORTED bool mlirAttributeIsAKTDPMemorySpaceAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirTypeID mlirKTDPMemorySpaceAttrGetTypeID(void);

MLIR_CAPI_EXPORTED enum MlirKTDPMemorySpaceKind
mlirKTDPMemorySpaceAttrGetKind(MlirAttribute attr);

/// Returns the compute-tile ID, or -1 if unspecified.
MLIR_CAPI_EXPORTED int32_t mlirKTDPMemorySpaceAttrGetCtId(MlirAttribute attr);

/// Register all KTIR passes with the global pass registry.  This must be
/// called through the shared CAPI aggregate library so that passes land in the
/// same PassRegistry that PassManager pipeline parsing consults.
MLIR_CAPI_EXPORTED void mlirKTIRRegisterPasses(void);

#ifdef __cplusplus
}
#endif

#endif // KTIR_C_DIALECT_KTDP_H_
