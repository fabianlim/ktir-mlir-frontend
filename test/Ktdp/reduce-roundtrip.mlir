// RUN: ktir-opt "%s" | ktir-opt | FileCheck "%s"

// Round-trip the `ktdp.reduce` op in both spelling forms (sugar and
// region) and both modes (`all_reduce`, `reduce_to_core<N>`).
//
// NOTE: attributes print in alphabetical order, so the printed form
// is `{across = ..., kind = ..., mode = ...}` regardless of source
// order in the input.

// -----
// Sugar form, all_reduce
// -----

// CHECK-LABEL: func.func @reduce_sugar_all_reduce
// CHECK: ktdp.reduce
// CHECK-SAME: across = #ktdp.grid_axis<0>, kind = #ktdp.reduce_kind<sum>, mode = #ktdp.reduce_mode<all_reduce>
// CHECK-SAME: tensor<32xf16> -> tensor<32xf16>
func.func @reduce_sugar_all_reduce(%a: tensor<32xf16>) -> tensor<32xf16> {
  %r = ktdp.reduce %a {
    kind   = #ktdp.reduce_kind<sum>,
    mode   = #ktdp.reduce_mode<all_reduce>,
    across = #ktdp.grid_axis<0>
  } : tensor<32xf16> -> tensor<32xf16>
  return %r : tensor<32xf16>
}

// -----
// Sugar form, reduce_to_core<N> — multiple shapes / dst values
// -----

// CHECK-LABEL: func.func @reduce_sugar_to_core_zero
// CHECK: ktdp.reduce
// CHECK-SAME: across = #ktdp.grid_axis<1>, kind = #ktdp.reduce_kind<sum>, mode = #ktdp.reduce_mode<reduce_to_core<0>>
// CHECK-SAME: tensor<32x512xf32> -> tensor<32x512xf32>
func.func @reduce_sugar_to_core_zero(%a: tensor<32x512xf32>) -> tensor<32x512xf32> {
  %r = ktdp.reduce %a {
    kind   = #ktdp.reduce_kind<sum>,
    mode   = #ktdp.reduce_mode<reduce_to_core<0>>,
    across = #ktdp.grid_axis<1>
  } : tensor<32x512xf32> -> tensor<32x512xf32>
  return %r : tensor<32x512xf32>
}

// CHECK-LABEL: func.func @reduce_sugar_to_core_three
// CHECK: ktdp.reduce
// CHECK-SAME: across = #ktdp.grid_axis<0>, kind = #ktdp.reduce_kind<sum>, mode = #ktdp.reduce_mode<reduce_to_core<3>>
// CHECK-SAME: tensor<8xf16> -> tensor<8xf16>
func.func @reduce_sugar_to_core_three(%a: tensor<8xf16>) -> tensor<8xf16> {
  %r = ktdp.reduce %a {
    kind   = #ktdp.reduce_kind<sum>,
    mode   = #ktdp.reduce_mode<reduce_to_core<3>>,
    across = #ktdp.grid_axis<0>
  } : tensor<8xf16> -> tensor<8xf16>
  return %r : tensor<8xf16>
}

// -----
// Region form, all_reduce — addf combiner over f16
// -----

// CHECK-LABEL: func.func @reduce_region_all_reduce_addf
// CHECK: ktdp.reduce {{.*}} combiner = {
// CHECK-NEXT: ^bb0(%{{.*}}: f16, %{{.*}}: f16):
// CHECK-NEXT: arith.addf
// CHECK-NEXT: ktdp.yield
// CHECK: across = #ktdp.grid_axis<0>, mode = #ktdp.reduce_mode<all_reduce>
// CHECK-SAME: tensor<32xf16> -> tensor<32xf16>
func.func @reduce_region_all_reduce_addf(%a: tensor<32xf16>) -> tensor<32xf16> {
  %r = ktdp.reduce %a combiner = {
    ^bb0(%x: f16, %y: f16):
      %s = arith.addf %x, %y : f16
      ktdp.yield %s : f16
  } {
    mode   = #ktdp.reduce_mode<all_reduce>,
    across = #ktdp.grid_axis<0>
  } : tensor<32xf16> -> tensor<32xf16>
  return %r : tensor<32xf16>
}

// -----
// Region form, reduce_to_core — maximumf combiner over f32
//
// Demonstrates that the region form accepts arbitrary combiner bodies.
// The verifier validates structure (block-arg arity/types, yield types),
// not the arithmetic. Whether such a body lowers to a hardware
// collective is a downstream concern, not an op-level claim.
// -----

// CHECK-LABEL: func.func @reduce_region_max_to_core
// CHECK: ktdp.reduce {{.*}} combiner = {
// CHECK-NEXT: ^bb0(%{{.*}}: f32, %{{.*}}: f32):
// CHECK-NEXT: arith.maximumf
// CHECK-NEXT: ktdp.yield
// CHECK: across = #ktdp.grid_axis<0>, mode = #ktdp.reduce_mode<reduce_to_core<0>>
// CHECK-SAME: tensor<16xf32> -> tensor<16xf32>
func.func @reduce_region_max_to_core(%a: tensor<16xf32>) -> tensor<16xf32> {
  %r = ktdp.reduce %a combiner = {
    ^bb0(%x: f32, %y: f32):
      %m = arith.maximumf %x, %y : f32
      ktdp.yield %m : f32
  } {
    mode   = #ktdp.reduce_mode<reduce_to_core<0>>,
    across = #ktdp.grid_axis<0>
  } : tensor<16xf32> -> tensor<16xf32>
  return %r : tensor<16xf32>
}
