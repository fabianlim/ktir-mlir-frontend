// RUN: ktir-opt "%s" | ktir-opt | FileCheck "%s"

// Round-trip the new reduce-related attributes on
// func.func arg/return positions, parallel to attrs-roundtrip.mlir.

// -----
// ReduceKindAttr
// -----

// CHECK-LABEL: func.func @reduce_kind_sum
// CHECK-SAME: kind = #ktdp.reduce_kind<sum>
func.func @reduce_kind_sum(%arg0: tensor<32xf16>) -> tensor<32xf16>
    attributes {kind = #ktdp.reduce_kind<sum>} {
  return %arg0 : tensor<32xf16>
}

// -----
// ReduceModeAttr — all_reduce
// -----

// CHECK-LABEL: func.func @reduce_mode_all_reduce
// CHECK-SAME: mode = #ktdp.reduce_mode<all_reduce>
func.func @reduce_mode_all_reduce(%arg0: tensor<32xf16>) -> tensor<32xf16>
    attributes {mode = #ktdp.reduce_mode<all_reduce>} {
  return %arg0 : tensor<32xf16>
}

// -----
// ReduceModeAttr — reduce_to_core<N>
// -----

// CHECK-LABEL: func.func @reduce_mode_to_core_zero
// CHECK-SAME: mode = #ktdp.reduce_mode<reduce_to_core<0>>
func.func @reduce_mode_to_core_zero(%arg0: tensor<32xf16>) -> tensor<32xf16>
    attributes {mode = #ktdp.reduce_mode<reduce_to_core<0>>} {
  return %arg0 : tensor<32xf16>
}

// CHECK-LABEL: func.func @reduce_mode_to_core_seven
// CHECK-SAME: mode = #ktdp.reduce_mode<reduce_to_core<7>>
func.func @reduce_mode_to_core_seven(%arg0: tensor<32xf16>) -> tensor<32xf16>
    attributes {mode = #ktdp.reduce_mode<reduce_to_core<7>>} {
  return %arg0 : tensor<32xf16>
}

// -----
// GridAxisAttr
// -----

// CHECK-LABEL: func.func @grid_axis_zero
// CHECK-SAME: across = #ktdp.grid_axis<0>
func.func @grid_axis_zero(%arg0: tensor<32xf16>) -> tensor<32xf16>
    attributes {across = #ktdp.grid_axis<0>} {
  return %arg0 : tensor<32xf16>
}

// CHECK-LABEL: func.func @grid_axis_one
// CHECK-SAME: across = #ktdp.grid_axis<1>
func.func @grid_axis_one(%arg0: tensor<32xf16>) -> tensor<32xf16>
    attributes {across = #ktdp.grid_axis<1>} {
  return %arg0 : tensor<32xf16>
}

// -----
// All three together
// -----

// CHECK-LABEL: func.func @all_three_attrs
// CHECK-SAME: across = #ktdp.grid_axis<0>
// CHECK-SAME: kind = #ktdp.reduce_kind<sum>
// CHECK-SAME: mode = #ktdp.reduce_mode<reduce_to_core<3>>
func.func @all_three_attrs(%arg0: tensor<32xf16>) -> tensor<32xf16>
    attributes {kind = #ktdp.reduce_kind<sum>,
                mode = #ktdp.reduce_mode<reduce_to_core<3>>,
                across = #ktdp.grid_axis<0>} {
  return %arg0 : tensor<32xf16>
}
