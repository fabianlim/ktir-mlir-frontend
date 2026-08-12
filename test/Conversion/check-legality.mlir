// RUN: ktir-opt --ktir-check-legality %s | FileCheck %s

#group_tiles = affine_set<(i)[g] : (i - 32*g >= 0, -i + 32*(g+1) - 1 >= 0)>
#all_groups  = affine_set<(g) : (g == 0)>

// CHECK-LABEL: func.func @reduce_single_role
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
