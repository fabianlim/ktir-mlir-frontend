//===- KtdpInterTileHelpers.cpp - Inter-tile enumeration helpers --*- C++ -*-===//
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
// Implements the pure-enumeration helpers declared in KtdpInterTileHelpers.h.
// We enumerate concrete integer points rather than using Presburger set
// operations; see the comment in KtdpOps.cpp for the trade-offs vs. the
// pure-Presburger Option A approach.
//
//===----------------------------------------------------------------------===//

#include "KTDPInterTileHelpers.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/IR/IntegerSet.h"

mlir::FailureOr<llvm::SmallVector<int64_t>>
mlir::ktdp::groupValues(mlir::IntegerSet groupsSet) {
  FlatLinearValueConstraints cst(groupsSet);
  std::optional<int64_t> lo =
      cst.getConstantBound64(presburger::BoundType::LB, /*pos=*/0);
  std::optional<int64_t> hi =
      cst.getConstantBound64(presburger::BoundType::UB, /*pos=*/0);
  if (!lo || !hi) return failure();
  llvm::SmallVector<int64_t> vals;
  for (int64_t g = *lo; g <= *hi; ++g) {
    FlatLinearValueConstraints gCst(groupsSet);
    gCst.setAndEliminate(gCst.getVarKindOffset(presburger::VarKind::SetDim),
                         {g});
    if (!gCst.isIntegerEmpty()) vals.push_back(g);
  }
  return vals;
}

mlir::FailureOr<llvm::DenseSet<int64_t>>
mlir::ktdp::tilesOf(mlir::IntegerSet tileSet, int64_t gVal) {
  FlatLinearValueConstraints cst(tileSet);
  cst.setAndEliminate(cst.getVarKindOffset(presburger::VarKind::Symbol),
                      {gVal});
  std::optional<int64_t> hi =
      cst.getConstantBound64(presburger::BoundType::UB, /*pos=*/0);
  if (!hi) return failure();
  llvm::DenseSet<int64_t> out;
  for (int64_t i = 0; i <= *hi; ++i) {
    FlatLinearValueConstraints pt(tileSet);
    pt.setAndEliminate(pt.getVarKindOffset(presburger::VarKind::Symbol),
                       {gVal});
    pt.setAndEliminate(pt.getVarKindOffset(presburger::VarKind::SetDim), {i});
    if (!pt.isIntegerEmpty()) out.insert(i);
  }
  return out;
}

mlir::FailureOr<llvm::DenseSet<int64_t>>
mlir::ktdp::depTilesOf(mlir::IntegerSet depSet, int64_t cVal, int64_t gVal) {
  FlatLinearValueConstraints cst(depSet);
  unsigned symBase = cst.getVarKindOffset(presburger::VarKind::Symbol);
  cst.setAndEliminate(symBase, {cVal});   // fix c (first symbol)
  cst.setAndEliminate(symBase, {gVal});   // fix g (now first remaining symbol)
  std::optional<int64_t> hi =
      cst.getConstantBound64(presburger::BoundType::UB, /*pos=*/0);
  if (!hi) return failure();
  llvm::DenseSet<int64_t> out;
  for (int64_t p = 0; p <= *hi; ++p) {
    FlatLinearValueConstraints pt(depSet);
    unsigned sb = pt.getVarKindOffset(presburger::VarKind::Symbol);
    pt.setAndEliminate(sb, {cVal});
    pt.setAndEliminate(sb, {gVal});
    pt.setAndEliminate(pt.getVarKindOffset(presburger::VarKind::SetDim), {p});
    if (!pt.isIntegerEmpty()) out.insert(p);
  }
  return out;
}
