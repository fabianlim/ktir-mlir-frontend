# RUN: python %s

# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from mlir_ktdp.ir import Module
from mlir_ktdp.passmanager import PassManager
from mlir_ktdp.tools import ktdp_context
from mlir_ktdp._mlir_libs._ktir import register_passes

# Register KTDP passes so the "ktir-check-legality" pipeline is known to
# PassManager.parse. This exercises the standard pass-pipeline path (see the
# CAPI aggregate fix) rather than the low-level run_check_legality() escape
# hatch, which remains available in the bindings.
register_passes()

CHECK_LEGALITY_PIPELINE = "builtin.module(ktir-check-legality)"


# ---------------------------------------------------------------------------
# Valid cases
# ---------------------------------------------------------------------------

VALID_ALL_REDUCE = """\
#group_tiles = affine_set<(i)[g] : (i - 32*g >= 0, -i + 32*(g+1) - 1 >= 0)>
#all_groups  = affine_set<(g) : (g == 0)>
module {
  func.func @reduce_single_role(%partial: tensor<64xf16>,
                                %add_id: tensor<64xf16>) -> tensor<64xf16> {
    %f = ktdp.inter_tile_produce
        producer_tiles_per_group = #group_tiles
        -> <(tensor<64xf16>), groups = #all_groups>
    { ^bb0(%gid: index): ktdp.yield_partial %partial : tensor<64xf16> }
    %r = ktdp.inter_tile_reduce(%f)
        consumer_tiles_per_group = #group_tiles,
        identity(%add_id : tensor<64xf16>)
        : <(tensor<64xf16>), groups = #all_groups> -> tensor<64xf16>
    { ^bb0(%lhs: tensor<64xf16>, %rhs: tensor<64xf16>):
        %s = linalg.add ins(%lhs, %rhs : tensor<64xf16>, tensor<64xf16>)
                        outs(%lhs : tensor<64xf16>) -> tensor<64xf16>
        ktdp.yield_reduced %s : tensor<64xf16> }
    return %r : tensor<64xf16>
  }
}
"""

# ---------------------------------------------------------------------------
# Invalid cases
# ---------------------------------------------------------------------------

INVALID_FUTURE_TWO_USES = """\
#su_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#su_grp   = affine_set<(g) : (g == 0)>
module {
  func.func @bad(%p: tensor<64xf16>, %id: tensor<64xf16>)
      -> (tensor<64xf16>, tensor<64xf16>) {
    %f = ktdp.inter_tile_produce producer_tiles_per_group = #su_tiles
        -> <(tensor<64xf16>), groups = #su_grp>
    { ^bb0(%gid: index): ktdp.yield_partial %p : tensor<64xf16> }
    %r1 = ktdp.inter_tile_reduce(%f) consumer_tiles_per_group = #su_tiles,
        identity(%id : tensor<64xf16>)
        : <(tensor<64xf16>), groups = #su_grp> -> tensor<64xf16>
    { ^bb0(%l: tensor<64xf16>, %rr: tensor<64xf16>):
        ktdp.yield_reduced %l : tensor<64xf16> }
    %r2 = ktdp.inter_tile_reduce(%f) consumer_tiles_per_group = #su_tiles,
        identity(%id : tensor<64xf16>)
        : <(tensor<64xf16>), groups = #su_grp> -> tensor<64xf16>
    { ^bb0(%l: tensor<64xf16>, %rr: tensor<64xf16>):
        ktdp.yield_reduced %l : tensor<64xf16> }
    return %r1, %r2 : tensor<64xf16>, tensor<64xf16>
  }
}
"""

INVALID_CONSUMER_NOT_SUBSET = """\
#q1_prod = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#q1_cons = affine_set<(i)[g] : (i - 4*g - 2 >= 0, -i + 4*g + 5 >= 0)>
#q1_grp  = affine_set<(g) : (g == 0)>
module {
  func.func @bad(%p: tensor<64xf16>, %id: tensor<64xf16>) -> tensor<64xf16> {
    %f = ktdp.inter_tile_produce producer_tiles_per_group = #q1_prod
        -> <(tensor<64xf16>), groups = #q1_grp>
    { ^bb0(%gid: index): ktdp.yield_partial %p : tensor<64xf16> }
    %r = ktdp.inter_tile_reduce(%f) consumer_tiles_per_group = #q1_cons,
        identity(%id : tensor<64xf16>)
        : <(tensor<64xf16>), groups = #q1_grp> -> tensor<64xf16>
    { ^bb0(%l: tensor<64xf16>, %rr: tensor<64xf16>):
        ktdp.yield_reduced %l : tensor<64xf16> }
    return %r : tensor<64xf16>
  }
}
"""

INVALID_REDUCE_TO_SUBSET = """\
#rs_prod = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#rs_cons = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 1 >= 0)>
#rs_grp  = affine_set<(g) : (g == 0)>
module {
  func.func @bad(%p: tensor<64xf16>, %id: tensor<64xf16>) -> tensor<64xf16> {
    %f = ktdp.inter_tile_produce producer_tiles_per_group = #rs_prod
        -> <(tensor<64xf16>), groups = #rs_grp>
    { ^bb0(%gid: index): ktdp.yield_partial %p : tensor<64xf16> }
    %r = ktdp.inter_tile_reduce(%f) consumer_tiles_per_group = #rs_cons,
        identity(%id : tensor<64xf16>)
        : <(tensor<64xf16>), groups = #rs_grp> -> tensor<64xf16>
    { ^bb0(%l: tensor<64xf16>, %rr: tensor<64xf16>):
        ktdp.yield_reduced %l : tensor<64xf16> }
    return %r : tensor<64xf16>
  }
}
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def check_valid(source):
    with ktdp_context() as ctx:
        mod = Module.parse(source, ctx)
        pm = PassManager.parse(CHECK_LEGALITY_PIPELINE, ctx)
        pm.run(mod.operation)


def check_invalid(source):
    with ktdp_context() as ctx:
        mod = Module.parse(source, ctx)
        pm = PassManager.parse(CHECK_LEGALITY_PIPELINE, ctx)
        raised = False
        try:
            pm.run(mod.operation)
        except Exception:
            raised = True
        assert raised, "expected ktir-check-legality to fail"


check_valid(VALID_ALL_REDUCE)

for source in [
    INVALID_FUTURE_TWO_USES,
    INVALID_CONSUMER_NOT_SUBSET,
    INVALID_REDUCE_TO_SUBSET,
]:
    check_invalid(source)

print("check_legality tests passed")
