//===- ConvertToKTIR.h - Conversion to KTIR ----------------------*- C++ -*-===//
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
//
// This file declares the KTIR legality surface and the pass that checks it.
//
//===----------------------------------------------------------------------===//

#ifndef KTIR_CONVERSION_CONVERTTOKTIR_CONVERTTOKTIR_H
#define KTIR_CONVERSION_CONVERTTOKTIR_CONVERTTOKTIR_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace ktir {

#define GEN_PASS_DECL_KTIRCHECKLEGALITYPASS
#include "ktir/Conversion/Passes.h.inc"

/// Populate \p target with the static op/dialect legality surface for KTIR.
/// Legal: ktdp, func, arith, linalg, math, memref, scf, tensor.
///
/// Downstream lowering pipelines can call this to share the same legality
/// definition without duplicating it.
void populateKTIRLegalTarget(mlir::ConversionTarget &target);

} // namespace ktir

#endif // KTIR_CONVERSION_CONVERTTOKTIR_CONVERTTOKTIR_H
