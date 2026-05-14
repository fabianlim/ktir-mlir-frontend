// RUN: ktir-opt "%s" | ktir-opt | FileCheck "%s"

// NOTE: Attributes print in alphabetical order, so the printed form is
// `{across = ..., kind = ..., mode = ...}` regardless of source order.

// Sugar form, all_reduce, 1-D f16 tensor, axis 0.
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

// Sugar form, reduce_to_core<0>, 2-D f32 tensor, axis 1.
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

// Sugar form, reduce_to_core with non-zero dst.
// CHECK-LABEL: func.func @reduce_sugar_to_core_three
// CHECK: mode = #ktdp.reduce_mode<reduce_to_core<3>>
func.func @reduce_sugar_to_core_three(%a: tensor<8xf16>) -> tensor<8xf16> {
  %r = ktdp.reduce %a {
    kind   = #ktdp.reduce_kind<sum>,
    mode   = #ktdp.reduce_mode<reduce_to_core<3>>,
    across = #ktdp.grid_axis<0>
  } : tensor<8xf16> -> tensor<8xf16>
  return %r : tensor<8xf16>
}

// Region form, all_reduce, addf combiner over f16.
// CHECK-LABEL: func.func @reduce_region_all_reduce_addf
// CHECK: ktdp.reduce {{.*}} combiner = {
// CHECK-NEXT: ^bb0(%{{.*}}: f16, %{{.*}}: f16):
// CHECK-NEXT: arith.addf
// CHECK-NEXT: ktdp.yield
// CHECK: mode = #ktdp.reduce_mode<all_reduce>
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

// Region form, reduce_to_core, maximumf combiner over f32. Demonstrates
// that the region escape hatch accepts arbitrary combiners — the verifier
// validates structure, not arithmetic. Whether this lowers is a downstream
// concern, not an op-level claim.
// CHECK-LABEL: func.func @reduce_region_max_to_core
// CHECK: ktdp.reduce {{.*}} combiner = {
// CHECK-NEXT: ^bb0(%{{.*}}: f32, %{{.*}}: f32):
// CHECK-NEXT: arith.maximumf
// CHECK: mode = #ktdp.reduce_mode<reduce_to_core<0>>
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
