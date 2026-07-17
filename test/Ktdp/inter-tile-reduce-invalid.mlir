// RUN: ktir-opt "%s" -split-input-file -verify-diagnostics

// Note: -split-input-file parses each chunk independently, so affine-set
// aliases are declared per chunk.

// -----
// yield_partial arity must match the future's partial count.
#g  = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#ag = affine_set<(g) : (g == 0)>
func.func @bad_produce_arity(%p: tensor<1x64xf16>) {
  // expected-error @below {{yield_partial yields 1 values but the future carries 2 partial type(s)}}
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #g
      : tensor<1x64xf16> -> !ktdp.tile_future<tensor<1x64xf16>, tensor<1x64xf16>, groups = #ag>
  { ^bb0(%gid: index): ktdp.yield_partial %p : tensor<1x64xf16> }
  return
}

// -----
// future result must have exactly one use.
#g  = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#ag = affine_set<(g) : (g == 0)>
func.func @bad_multiple_uses(%p: tensor<1x64xf16>, %id: tensor<1x64xf16>) {
  // expected-error @below {{future result must have exactly one use}}
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #g
      : tensor<1x64xf16> -> !ktdp.tile_future<tensor<1x64xf16>, groups = #ag>
  { ^bb0(%gid: index): ktdp.yield_partial %p : tensor<1x64xf16> }
  %r0 = ktdp.inter_tile_reduce(%f) consumer_tiles_per_group = #g, identity(%id : tensor<1x64xf16>)
      : !ktdp.tile_future<tensor<1x64xf16>, groups = #ag> -> tensor<64xf16>
  { ^bb0(%l: tensor<1x64xf16>, %rr: tensor<1x64xf16>): ktdp.yield_reduced %l : tensor<1x64xf16> }
  %r1 = ktdp.inter_tile_reduce(%f) consumer_tiles_per_group = #g, identity(%id : tensor<1x64xf16>)
      : !ktdp.tile_future<tensor<1x64xf16>, groups = #ag> -> tensor<64xf16>
  { ^bb0(%l: tensor<1x64xf16>, %rr: tensor<1x64xf16>): ktdp.yield_reduced %l : tensor<1x64xf16> }
  return
}

// -----
// identity type must match the partial type.
#g  = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#ag = affine_set<(g) : (g == 0)>
func.func @bad_identity_type(%p: tensor<1x64xf16>, %id: tensor<2x64xf16>) -> tensor<64xf16> {
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #g
      : tensor<1x64xf16> -> !ktdp.tile_future<tensor<1x64xf16>, groups = #ag>
  { ^bb0(%gid: index): ktdp.yield_partial %p : tensor<1x64xf16> }
  // expected-error @below {{identity #0 type 'tensor<2x64xf16>' must match future partial type 'tensor<1x64xf16>'}}
  %r = ktdp.inter_tile_reduce(%f) consumer_tiles_per_group = #g, identity(%id : tensor<2x64xf16>)
      : !ktdp.tile_future<tensor<1x64xf16>, groups = #ag> -> tensor<64xf16>
  { ^bb0(%l: tensor<1x64xf16>, %rr: tensor<1x64xf16>): ktdp.yield_reduced %l : tensor<1x64xf16> }
  return %r : tensor<64xf16>
}

// -----
// result rank must be exactly one less than the partial rank.
#g  = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#ag = affine_set<(g) : (g == 0)>
func.func @bad_collapse_rank(%p: tensor<1x64xf16>, %id: tensor<1x64xf16>) -> tensor<1x64xf16> {
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #g
      : tensor<1x64xf16> -> !ktdp.tile_future<tensor<1x64xf16>, groups = #ag>
  { ^bb0(%gid: index): ktdp.yield_partial %p : tensor<1x64xf16> }
  // expected-error @below {{result #0 rank (2) must be one less than partial rank (2)}}
  %r = ktdp.inter_tile_reduce(%f) consumer_tiles_per_group = #g, identity(%id : tensor<1x64xf16>)
      : !ktdp.tile_future<tensor<1x64xf16>, groups = #ag> -> tensor<1x64xf16>
  { ^bb0(%l: tensor<1x64xf16>, %rr: tensor<1x64xf16>): ktdp.yield_reduced %l : tensor<1x64xf16> }
  return %r : tensor<1x64xf16>
}

// -----
// collapsed axis must be a unit dimension (96 cannot collapse to 64).
#g  = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#ag = affine_set<(g) : (g == 0)>
func.func @bad_collapse_nonunit(%p: tensor<96x64xf16>, %id: tensor<96x64xf16>) -> tensor<64xf16> {
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #g
      : tensor<96x64xf16> -> !ktdp.tile_future<tensor<96x64xf16>, groups = #ag>
  { ^bb0(%gid: index): ktdp.yield_partial %p : tensor<96x64xf16> }
  // expected-error @below {{result #0 shape does not match partial #0 with a single unit within-group tile axis collapsed}}
  %r = ktdp.inter_tile_reduce(%f) consumer_tiles_per_group = #g, identity(%id : tensor<96x64xf16>)
      : !ktdp.tile_future<tensor<96x64xf16>, groups = #ag> -> tensor<64xf16>
  { ^bb0(%l: tensor<96x64xf16>, %rr: tensor<96x64xf16>): ktdp.yield_reduced %l : tensor<96x64xf16> }
  return %r : tensor<64xf16>
}

// -----
// tile_future partial types must be ranked tensors.
// expected-error @below {{tile_future partial type must be a ranked tensor, but got: 'index'}}
func.func @bad_future_scalar(%a: !ktdp.tile_future<index, groups = affine_set<(g) : (g == 0)>>) { return }

// -----
// producer tile sets for distinct groups must be disjoint (§2.1).
// tiles 3g..3g+3 over groups 0,1: g=0 -> {0,1,2,3}, g=1 -> {3,4,5,6}; tile 3 overlaps.
#prod_overlap = affine_set<(i)[g] : (i - 3*g >= 0, -i + 3*g + 3 >= 0)>
#two_groups   = affine_set<(g) : (g >= 0, -g + 1 >= 0)>
func.func @bad_producer_overlap(%p: tensor<1x64xf16>) {
  // expected-error @below {{producer_tiles_per_group for groups 0 and 1 are not disjoint}}
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #prod_overlap
      : tensor<1x64xf16> -> !ktdp.tile_future<tensor<1x64xf16>, groups = #two_groups>
  { ^bb0(%gid: index): ktdp.yield_partial %p : tensor<1x64xf16> }
  return
}

// -----
// Consumer that did not produce: C = {4g+2..4g+5}, P = {4g..4g+3} -> C not subset
// of P. Unsupported (open question Q1).
#q1_prod = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#q1_cons = affine_set<(i)[g] : (i - 4*g - 2 >= 0, -i + 4*g + 5 >= 0)>
#q1_grp  = affine_set<(g) : (g == 0)>
func.func @bad_consumer_not_subset(%p: tensor<1x64xf16>, %id: tensor<1x64xf16>) -> tensor<64xf16> {
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #q1_prod
      : tensor<1x64xf16> -> !ktdp.tile_future<tensor<1x64xf16>, groups = #q1_grp>
  { ^bb0(%gid: index): ktdp.yield_partial %p : tensor<1x64xf16> }
  // expected-error @below {{consumer_tiles_per_group for group 0 is not a subset of producer_tiles_per_group}}
  %r = ktdp.inter_tile_reduce(%f) consumer_tiles_per_group = #q1_cons, identity(%id : tensor<1x64xf16>)
      : !ktdp.tile_future<tensor<1x64xf16>, groups = #q1_grp> -> tensor<64xf16>
  { ^bb0(%l: tensor<1x64xf16>, %rr: tensor<1x64xf16>): ktdp.yield_reduced %l : tensor<1x64xf16> }
  return %r : tensor<64xf16>
}

// -----
// Reduce-to-subset: C = {4g, 4g+1} is a strict subset of P = {4g..4g+3} with >1
// tile. Spec-legal but UNSUPPORTED by this implementation (only all-reduce and
// reduce-to-one are supported).
#rs_prod = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#rs_cons = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 1 >= 0)>
#rs_grp  = affine_set<(g) : (g == 0)>
func.func @bad_reduce_to_subset(%p: tensor<1x64xf16>, %id: tensor<1x64xf16>) -> tensor<64xf16> {
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #rs_prod
      : tensor<1x64xf16> -> !ktdp.tile_future<tensor<1x64xf16>, groups = #rs_grp>
  { ^bb0(%gid: index): ktdp.yield_partial %p : tensor<1x64xf16> }
  // expected-error @below {{reduce-to-subset is unsupported; only all-reduce and reduce-to-one are supported}}
  %r = ktdp.inter_tile_reduce(%f) consumer_tiles_per_group = #rs_cons, identity(%id : tensor<1x64xf16>)
      : !ktdp.tile_future<tensor<1x64xf16>, groups = #rs_grp> -> tensor<64xf16>
  { ^bb0(%l: tensor<1x64xf16>, %rr: tensor<1x64xf16>): ktdp.yield_reduced %l : tensor<1x64xf16> }
  return %r : tensor<64xf16>
}

// -----
// tile_future groups must have exactly one dimension (not zero dims).
// expected-error @below {{tile_future `groups` must have exactly one dimension (g)}}
func.func @bad_future_groups_zero_dims(%a: !ktdp.tile_future<tensor<1x64xf16>, groups = affine_set<() : (0 == 0)>>) { return }

// -----
// tile_future groups must have exactly one dimension (not two dims).
// expected-error @below {{tile_future `groups` must have exactly one dimension (g)}}
func.func @bad_future_groups_two_dims(%a: !ktdp.tile_future<tensor<1x64xf16>, groups = affine_set<(g, h) : (g == 0, h == 0)>>) { return }

// -----
// tile_future groups must have no symbols.
// expected-error @below {{tile_future `groups` must have no symbols}}
func.func @bad_future_groups_has_symbol(%a: !ktdp.tile_future<tensor<1x64xf16>, groups = affine_set<(g)[s] : (g == s)>>) { return }
