// RUN: ktir-opt --ktir-check-legality --split-input-file --verify-diagnostics %s

// Single-use invariant (§2.3): future must have exactly one use.

#su_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#su_grp   = affine_set<(g) : (g == 0)>

func.func @bad_future_two_uses(%p: tensor<64xf16>, %id: tensor<64xf16>)
    -> (tensor<64xf16>, tensor<64xf16>) {
  // expected-error @below {{future result must have exactly one use}}
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

// -----

// C⊆P check: C = {4g+2..4g+5}, P = {4g..4g+3} — tiles 4g+4,4g+5 not in P.

#q1_prod = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#q1_cons = affine_set<(i)[g] : (i - 4*g - 2 >= 0, -i + 4*g + 5 >= 0)>
#q1_grp  = affine_set<(g) : (g == 0)>

func.func @bad_consumer_not_subset(%p: tensor<64xf16>, %id: tensor<64xf16>)
    -> tensor<64xf16> {
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #q1_prod
      -> <(tensor<64xf16>), groups = #q1_grp>
  { ^bb0(%gid: index): ktdp.yield_partial %p : tensor<64xf16> }
  // expected-error @below {{consumer_tiles_per_group for group 0 is not a subset of producer_tiles_per_group}}
  %r = ktdp.inter_tile_reduce(%f) consumer_tiles_per_group = #q1_cons,
      identity(%id : tensor<64xf16>)
      : <(tensor<64xf16>), groups = #q1_grp> -> tensor<64xf16>
  { ^bb0(%l: tensor<64xf16>, %rr: tensor<64xf16>):
      ktdp.yield_reduced %l : tensor<64xf16> }
  return %r : tensor<64xf16>
}

// -----

// Mode gate: reduce-to-subset (|C|=2, C ⊊ P) unsupported.
// C = {4g, 4g+1}, P = {4g..4g+3}.

#rs_prod = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#rs_cons = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 1 >= 0)>
#rs_grp  = affine_set<(g) : (g == 0)>

func.func @bad_reduce_to_subset(%p: tensor<64xf16>, %id: tensor<64xf16>)
    -> tensor<64xf16> {
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #rs_prod
      -> <(tensor<64xf16>), groups = #rs_grp>
  { ^bb0(%gid: index): ktdp.yield_partial %p : tensor<64xf16> }
  // expected-error @below {{reduce-to-subset is unsupported; only all-reduce and reduce-to-one are supported}}
  %r = ktdp.inter_tile_reduce(%f) consumer_tiles_per_group = #rs_cons,
      identity(%id : tensor<64xf16>)
      : <(tensor<64xf16>), groups = #rs_grp> -> tensor<64xf16>
  { ^bb0(%l: tensor<64xf16>, %rr: tensor<64xf16>):
      ktdp.yield_reduced %l : tensor<64xf16> }
  return %r : tensor<64xf16>
}

// -----

// A dependency set may carry one symbol (c) or two (c, g); three is not a
// legal spelling.

#ds_prod = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#ds_grp  = affine_set<(g) : (g == 0)>
#ds_bad  = affine_set<(p)[c, g, x] : (p - c == 0, x >= 0)>

func.func @bad_dep_symbol_count(%p: tensor<64xf16>, %id: tensor<64xf16>)
    -> tensor<64xf16> {
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #ds_prod
      -> <(tensor<64xf16>), groups = #ds_grp>
  { ^bb0(%gid: index): ktdp.yield_partial %p : tensor<64xf16> }
  // expected-error @below {{`producer_dependency_per_consumer` must have one symbol (c) or two symbols (c, g)}}
  %r = ktdp.inter_tile_reduce(%f) consumer_tiles_per_group = #ds_prod,
      producer_dependency_per_consumer = #ds_bad,
      identity(%id : tensor<64xf16>)
      : <(tensor<64xf16>), groups = #ds_grp> -> tensor<64xf16>
  { ^bb0(%l: tensor<64xf16>, %rr: tensor<64xf16>):
      ktdp.yield_reduced %l : tensor<64xf16> }
  return %r : tensor<64xf16>
}
