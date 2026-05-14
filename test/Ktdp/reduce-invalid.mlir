// RUN: ktir-opt "%s" -split-input-file -verify-diagnostics

// -----

// Both `kind` and combiner region present.
func.func @bad_both_kind_and_region(%a: tensor<32xf16>) -> tensor<32xf16> {
  // expected-error @below {{exactly one of `kind` (sugar form) or a non-empty combiner region must be provided, but both were given}}
  %r = ktdp.reduce %a combiner = {
    ^bb0(%x: f16, %y: f16):
      %s = arith.addf %x, %y : f16
      ktdp.yield %s : f16
  } {
    kind   = #ktdp.reduce_kind<sum>,
    mode   = #ktdp.reduce_mode<all_reduce>,
    across = #ktdp.grid_axis<0>
  } : tensor<32xf16> -> tensor<32xf16>
  return %r : tensor<32xf16>
}

// -----

// Neither `kind` nor combiner region.
func.func @bad_neither_kind_nor_region(%a: tensor<32xf16>) -> tensor<32xf16> {
  // expected-error @below {{exactly one of `kind` (sugar form) or a non-empty combiner region must be provided, but neither was given}}
  %r = ktdp.reduce %a {
    mode   = #ktdp.reduce_mode<all_reduce>,
    across = #ktdp.grid_axis<0>
  } : tensor<32xf16> -> tensor<32xf16>
  return %r : tensor<32xf16>
}

// -----

// Result type does not match input type.
func.func @bad_result_type(%a: tensor<32xf16>) -> tensor<32xf32> {
  // expected-error @below {{result type 'tensor<32xf32>' must match input type 'tensor<32xf16>'}}
  %r = ktdp.reduce %a {
    kind   = #ktdp.reduce_kind<sum>,
    mode   = #ktdp.reduce_mode<all_reduce>,
    across = #ktdp.grid_axis<0>
  } : tensor<32xf16> -> tensor<32xf32>
  return %r : tensor<32xf32>
}

// -----

// Combiner block has wrong number of arguments (3 instead of 2).
func.func @bad_combiner_block_arity(%a: tensor<32xf16>) -> tensor<32xf16> {
  // expected-error @below {{combiner block must have 2 arguments (2 per input), but got 3}}
  %r = ktdp.reduce %a combiner = {
    ^bb0(%x: f16, %y: f16, %z: f16):
      %s = arith.addf %x, %y : f16
      ktdp.yield %s : f16
  } {
    mode   = #ktdp.reduce_mode<all_reduce>,
    across = #ktdp.grid_axis<0>
  } : tensor<32xf16> -> tensor<32xf16>
  return %r : tensor<32xf16>
}

// -----

// Combiner block argument element type does not match input element type.
func.func @bad_combiner_arg_type(%a: tensor<32xf16>) -> tensor<32xf16> {
  // expected-error @below {{combiner block arguments at indices 0 and 1 must have type 'f16', but got 'f32' and 'f32'}}
  %r = ktdp.reduce %a combiner = {
    ^bb0(%x: f32, %y: f32):
      %s = arith.addf %x, %y : f32
      ktdp.yield %s : f32
  } {
    mode   = #ktdp.reduce_mode<all_reduce>,
    across = #ktdp.grid_axis<0>
  } : tensor<32xf16> -> tensor<32xf16>
  return %r : tensor<32xf16>
}

// -----

// `ktdp.yield` produces value of wrong type.
func.func @bad_yield_type(%a: tensor<32xf16>) -> tensor<32xf16> {
  // expected-error @below {{combiner `ktdp.yield` operand #0 has type 'f32', but expected element type 'f16'}}
  %r = ktdp.reduce %a combiner = {
    ^bb0(%x: f16, %y: f16):
      %s = arith.extf %x : f16 to f32
      ktdp.yield %s : f32
  } {
    mode   = #ktdp.reduce_mode<all_reduce>,
    across = #ktdp.grid_axis<0>
  } : tensor<32xf16> -> tensor<32xf16>
  return %r : tensor<32xf16>
}
