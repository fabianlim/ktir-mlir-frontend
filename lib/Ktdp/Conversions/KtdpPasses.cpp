//===- KtdpPasses.cpp - KTDP pass implementations ----------------*- C++ -*-===//
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
// This file implements passes for the KTDP dialect (issue #35).
//
//===----------------------------------------------------------------------===//

#include "Ktdp/KtdpPasses.h"
#include "Ktdp/KtdpDialect.hpp"
#include "Ktdp/KtdpOps.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "../KtdpInterTileHelpers.h"

namespace mlir::ktdp {
#define GEN_PASS_DEF_KTIRCHECKLEGALITYPASS
#include "Ktdp/KtdpPasses.hpp.inc"
} // namespace mlir::ktdp

void mlir::ktdp::populateKTIRLegalTarget(mlir::ConversionTarget &target) {
  target.addLegalDialect<mlir::ktdp::KtdpDialect>();
  target.addLegalDialect<mlir::arith::ArithDialect>();
  target.addLegalDialect<mlir::func::FuncDialect>();
  target.addLegalDialect<mlir::linalg::LinalgDialect>();
  target.addLegalDialect<mlir::math::MathDialect>();
  target.addLegalDialect<mlir::memref::MemRefDialect>();
  target.addLegalDialect<mlir::scf::SCFDialect>();
  target.addLegalDialect<mlir::tensor::TensorDialect>();
}

namespace mlir::ktdp {
namespace {

struct KtirCheckLegalityPass
    : impl::KtirCheckLegalityPassBase<KtirCheckLegalityPass> {

  void runOnOperation() override {
    bool failed = false;

    // --- Static legality via ConversionTarget + applyPartialConversion ---
    ConversionTarget target(getContext());
    populateKTIRLegalTarget(target);

    mlir::ConversionConfig config;
    mlir::DenseSet<mlir::Operation *> unlegalizedOps;
    config.unlegalizedOps = &unlegalizedOps;

    mlir::FrozenRewritePatternSet emptyPatterns;
    if (mlir::failed(mlir::applyPartialConversion(
            getOperation(), target, emptyPatterns, config)))
      failed = true;

    // --- Cross-op inter-tile invariants via IR walk ---

    // Single-use invariant on inter_tile_produce (§2.3): the future result
    // must have exactly one use — the single delivery op that consumes it.
    getOperation()->walk([&](InterTileProduceOp produceOp) {
      if (!produceOp.getFuture().hasOneUse()) {
        produceOp.emitError("future result must have exactly one use");
        failed = true;
      }
    });

    // Consumer-vs-producer checks on inter_tile_reduce.
    getOperation()->walk([&](InterTileReduceOp reduceOp) {
      auto produceOp =
          reduceOp.getFuture().getDefiningOp<InterTileProduceOp>();
      if (!produceOp) return; // dynamic future — cannot compare statically

      IntegerSet groupsSet   = reduceOp.getGroups();
      IntegerSet consumerSet = reduceOp.getConsumerTilesPerGroup().getValue();
      IntegerSet producerSet = produceOp.getProducerTilesPerGroup().getValue();

      auto groupVals = groupValues(groupsSet);
      if (mlir::failed(groupVals)) return; // unbounded group range — defer

      for (int64_t g : *groupVals) {
        auto cOpt = tilesOf(consumerSet, g);
        auto pOpt = tilesOf(producerSet, g);
        if (mlir::failed(cOpt) || mlir::failed(pOpt)) continue;
        const auto &c = *cOpt;
        const auto &p = *pOpt;

        // C⊆P check: every consumer tile must be a producer tile.
        for (int64_t tile : c) {
          if (!p.count(tile)) {
            reduceOp.emitError("consumer_tiles_per_group for group ")
                << g << " is not a subset of producer_tiles_per_group "
                << "(a consumer tile that did not produce is unsupported; "
                << "see open question Q1)";
            failed = true;
            return;
          }
        }

        // Mode gate: only all-reduce (C == P) or reduce-to-one (|C| == 1).
        if (c == p) continue;
        if (c.size() == 1) continue;
        reduceOp.emitError("consumer_tiles_per_group for group ")
            << g << " is a strict subset of producer_tiles_per_group with "
            << "more than one tile (reduce-to-subset is unsupported; only "
            << "all-reduce and reduce-to-one are supported)";
        failed = true;
        return;
      }

      // producer_dependency_per_consumer checks (§4.1), when present.
      auto depAttr = reduceOp.getProducerDependencyPerConsumerAttr();
      if (!depAttr) return;

      IntegerSet depSet = depAttr.getValue();
      if (depSet.getNumSymbols() != 2) {
        reduceOp.emitError("`producer_dependency_per_consumer` must have "
                           "exactly two symbols (c, g)");
        failed = true;
        return;
      }

      for (int64_t g : *groupVals) {
        auto cOpt = tilesOf(consumerSet, g);
        auto pOpt = tilesOf(producerSet, g);
        if (mlir::failed(cOpt) || mlir::failed(pOpt)) continue;
        llvm::DenseSet<int64_t> covered;
        for (int64_t c : *cOpt) {
          auto dOpt = depTilesOf(depSet, c, g);
          if (mlir::failed(dOpt)) continue;
          for (int64_t p : *dOpt) {
            // (a) Subset: every declared dep must be in P(g).
            if (!pOpt->count(p)) {
              reduceOp.emitError(
                  "producer_dependency_per_consumer for consumer ")
                  << c << " group " << g << " references producer tile " << p
                  << " which is not in producer_tiles_per_group";
              failed = true;
              return;
            }
            covered.insert(p);
          }
        }
        // (b) Coverage: every p in P(g) must appear in some dep set.
        for (int64_t p : *pOpt) {
          if (!covered.count(p)) {
            reduceOp.emitError(
                "producer_dependency_per_consumer for group ")
                << g << " does not cover producer tile " << p
                << " (no consumer has it as a dependency)";
            failed = true;
            return;
          }
        }
      }
    });

    if (failed) signalPassFailure();
  }
};

} // namespace
} // namespace mlir::ktdp
