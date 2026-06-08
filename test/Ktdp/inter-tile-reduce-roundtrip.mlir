// RUN: ktir-opt "%s" | ktir-opt | FileCheck "%s"

// Inter-tile all-reduce: ktdp.inter_tile_produce + ktdp.inter_tile_reduce.
// See docs/inter-tile-communication.md (§8.2).

// -----
// Minimal single-role reduce: collapse the leading unit within-group tile axis.
// -----

#group_tiles = affine_set<(i)[g] : (i - 32*g >= 0, -i + 32*(g+1) - 1 >= 0)>
#all_groups  = affine_set<(g) : (g == 0)>

// CHECK-LABEL: func.func @reduce_single_role
func.func @reduce_single_role(%partial: tensor<1x64xf16>,
                              %add_id: tensor<1x64xf16>) -> tensor<64xf16> {
  // CHECK: ktdp.inter_tile_produce producer_tiles_per_group = #{{.*}}, groups = #{{.*}} : tensor<1x64xf16> -> !ktdp.tile_future<tensor<1x64xf16>>
  %f = ktdp.inter_tile_produce
      producer_tiles_per_group = #group_tiles,
      groups = #all_groups
      : tensor<1x64xf16> -> !ktdp.tile_future<tensor<1x64xf16>>
  {
    ^bb0(%gid: index):
      // CHECK: ktdp.yield_partial %{{.*}} : tensor<1x64xf16>
      ktdp.yield_partial %partial : tensor<1x64xf16>
  }
  // CHECK: ktdp.inter_tile_reduce(%{{.*}}) consumer_tiles_per_group = #{{.*}}, groups = #{{.*}}, identity(%{{.*}} : tensor<1x64xf16>) : !ktdp.tile_future<tensor<1x64xf16>> -> tensor<64xf16>
  %r = ktdp.inter_tile_reduce(%f)
      consumer_tiles_per_group = #group_tiles,
      groups = #all_groups,
      identity(%add_id : tensor<1x64xf16>)
      : !ktdp.tile_future<tensor<1x64xf16>> -> tensor<64xf16>
  {
    ^bb0(%lhs: tensor<1x64xf16>, %rhs: tensor<1x64xf16>):
      %s = linalg.add ins(%lhs, %rhs : tensor<1x64xf16>, tensor<1x64xf16>)
                      outs(%lhs : tensor<1x64xf16>) -> tensor<1x64xf16>
      // CHECK: ktdp.yield_reduced %{{.*}} : tensor<1x64xf16>
      ktdp.yield_reduced %s : tensor<1x64xf16>
  }
  return %r : tensor<64xf16>
}

// -----
// Reduce-to-one: 4 producer tiles per group, only tile 4g consumes the result.
// Supported (|C| == 1, C subset of P); see §4.1 / open question Q1.
// -----

#r2o_prod = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#r2o_cons = affine_set<(i)[g] : (i - 4*g == 0)>
#r2o_grp  = affine_set<(g) : (g == 0)>

// CHECK-LABEL: func.func @reduce_to_one
func.func @reduce_to_one(%partial: tensor<1x64xf16>,
                         %add_id: tensor<1x64xf16>) -> tensor<64xf16> {
  %f = ktdp.inter_tile_produce
      producer_tiles_per_group = #r2o_prod, groups = #r2o_grp
      : tensor<1x64xf16> -> !ktdp.tile_future<tensor<1x64xf16>>
  { ^bb0(%gid: index): ktdp.yield_partial %partial : tensor<1x64xf16> }
  // CHECK: ktdp.inter_tile_reduce
  %r = ktdp.inter_tile_reduce(%f)
      consumer_tiles_per_group = #r2o_cons, groups = #r2o_grp,
      identity(%add_id : tensor<1x64xf16>)
      : !ktdp.tile_future<tensor<1x64xf16>> -> tensor<64xf16>
  { ^bb0(%lhs: tensor<1x64xf16>, %rhs: tensor<1x64xf16>):
      %s = linalg.add ins(%lhs, %rhs : tensor<1x64xf16>, tensor<1x64xf16>)
                      outs(%lhs : tensor<1x64xf16>) -> tensor<1x64xf16>
      ktdp.yield_reduced %s : tensor<1x64xf16> }
  return %r : tensor<64xf16>
}

// -----
// Multi-group reduce (collapse interior unit axis, preserve the group axis).
// docs/inter-tile-communication.md §8.2.2.
// -----

#mg_tiles  = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#mg_groups = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

// CHECK-LABEL: func.func @reduce_multi_group
func.func @reduce_multi_group(%partial: tensor<128x1x1x64xf16>,
                              %add_id: tensor<128x1x1x64xf16>) -> tensor<128x1x64xf16> {
  // CHECK: ktdp.inter_tile_produce
  %f = ktdp.inter_tile_produce
      producer_tiles_per_group = #mg_tiles,
      groups = #mg_groups
      : tensor<128x1x1x64xf16> -> !ktdp.tile_future<tensor<128x1x1x64xf16>>
  {
    ^bb0(%gid: index):
      ktdp.yield_partial %partial : tensor<128x1x1x64xf16>
  }
  // CHECK: ktdp.inter_tile_reduce
  %r = ktdp.inter_tile_reduce(%f)
      consumer_tiles_per_group = #mg_tiles,
      groups = #mg_groups,
      identity(%add_id : tensor<128x1x1x64xf16>)
      : !ktdp.tile_future<tensor<128x1x1x64xf16>> -> tensor<128x1x64xf16>
  {
    ^bb0(%lhs: tensor<128x1x1x64xf16>, %rhs: tensor<128x1x1x64xf16>):
      %s = linalg.add ins(%lhs, %rhs : tensor<128x1x1x64xf16>, tensor<128x1x1x64xf16>)
                      outs(%lhs : tensor<128x1x1x64xf16>) -> tensor<128x1x1x64xf16>
      ktdp.yield_reduced %s : tensor<128x1x1x64xf16>
  }
  return %r : tensor<128x1x64xf16>
}

// -----
// Multi-role (argmax-style, N=2) reduce with per-tile dependency set.
// -----

#bf_tiles  = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#bf_groups = affine_set<(g) : (g >= 0, -g + 7 >= 0)>
#bf_dep    = affine_set<(p)[c, g] : (p + c - 8*g - 3 == 0)>

// CHECK-LABEL: func.func @reduce_argmax
func.func @reduce_argmax(%pv: tensor<1x64xf32>, %pi: tensor<1x64xi32>,
                         %iv: tensor<1x64xf32>, %ii: tensor<1x64xi32>)
    -> (tensor<64xf32>, tensor<64xi32>) {
  %f = ktdp.inter_tile_produce
      producer_tiles_per_group = #bf_tiles,
      groups = #bf_groups
      : tensor<1x64xf32>, tensor<1x64xi32>
        -> !ktdp.tile_future<tensor<1x64xf32>, tensor<1x64xi32>>
  {
    ^bb0(%gid: index):
      ktdp.yield_partial %pv, %pi : tensor<1x64xf32>, tensor<1x64xi32>
  }
  // CHECK: producer_dependency_per_consumer = #{{.*}}
  %rv, %ri = ktdp.inter_tile_reduce(%f)
      consumer_tiles_per_group = #bf_tiles,
      groups = #bf_groups,
      producer_dependency_per_consumer = #bf_dep,
      identity(%iv : tensor<1x64xf32>, %ii : tensor<1x64xi32>)
      : !ktdp.tile_future<tensor<1x64xf32>, tensor<1x64xi32>>
        -> tensor<64xf32>, tensor<64xi32>
  {
    ^bb0(%lv: tensor<1x64xf32>, %li: tensor<1x64xi32>,
         %rv2: tensor<1x64xf32>, %ri2: tensor<1x64xi32>):
      // 2-adic argmax combiner: keep the larger value and its index.
      %take_lhs = arith.cmpf ogt, %lv, %rv2 : tensor<1x64xf32>
      %max_v = arith.maxnumf %lv, %rv2 : tensor<1x64xf32>
      %max_i = arith.select %take_lhs, %li, %ri2 : tensor<1x64xi1>, tensor<1x64xi32>
      ktdp.yield_reduced %max_v, %max_i : tensor<1x64xf32>, tensor<1x64xi32>
  }
  return %rv, %ri : tensor<64xf32>, tensor<64xi32>
}
