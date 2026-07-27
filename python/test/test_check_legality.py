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

import pytest
from tools_ktdp import ktdp_context
from mlir_ktdp.ir import Module
from mlir_ktdp._mlir_libs._ktir import run_check_legality


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

def test_valid_all_reduce():
    with ktdp_context() as ctx:
        mod = Module.parse(VALID_ALL_REDUCE, ctx)
        run_check_legality(ctx, mod.operation)


@pytest.mark.parametrize("source,expected_msg", [
    (INVALID_FUTURE_TWO_USES,    "future result must have exactly one use"),
    (INVALID_CONSUMER_NOT_SUBSET,"is not a subset of producer_tiles_per_group"),
    (INVALID_REDUCE_TO_SUBSET,   "reduce-to-subset is unsupported"),
], ids=["future_two_uses", "consumer_not_subset", "reduce_to_subset"])
def test_invalid(source, expected_msg):
    with ktdp_context() as ctx:
        mod = Module.parse(source, ctx)
        with pytest.raises(ValueError):
            run_check_legality(ctx, mod.operation)
