//===- KtdpPasses.td - KTDP pass definitions ---------------*- tablegen -*-===//
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
// This file defines passes for the KTDP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KTDP_PASSES_TD
#define KTDP_PASSES_TD

include "mlir/Pass/PassBase.td"

def KtirCheckLegalityPass : Pass<"ktir-check-legality", "::mlir::ModuleOp"> {
  let summary = "Verify KTIR legality: static op/dialect constraints via "
                "ConversionTarget, cross-op inter-tile invariants via IR walk";
  let dependentDialects = ["::mlir::ktdp::KtdpDialect"];
}

#endif // KTDP_PASSES_TD
