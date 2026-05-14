// RUN: ktir-opt "%s" -split-input-file -verify-diagnostics

// -----

// Verify: GridAxisAttr requires non-negative axis.
func.func @bad_grid_axis_negative(%arg0: tensor<32xf16>) -> tensor<32xf16>
    // expected-error @+1 {{grid_axis must be non-negative, but got: -1}}
    attributes {across = #ktdp.grid_axis<-1>} {
  return %arg0 : tensor<32xf16>
}

// -----

// Verify: reduce_to_core requires a destination core rank.
func.func @bad_reduce_to_core_missing_dst(%arg0: tensor<32xf16>) -> tensor<32xf16>
    // expected-error @+1 {{reduce_to_core requires a destination core rank}}
    attributes {mode = #ktdp.reduce_mode<reduce_to_core>} {
  return %arg0 : tensor<32xf16>
}

// -----

// Verify: reduce_to_core dst must be non-negative.
func.func @bad_reduce_to_core_negative_dst(%arg0: tensor<32xf16>) -> tensor<32xf16>
    // expected-error @+1 {{reduce_mode<reduce_to_core<...>> requires a non-negative `dst`}}
    attributes {mode = #ktdp.reduce_mode<reduce_to_core<-1>>} {
  return %arg0 : tensor<32xf16>
}
