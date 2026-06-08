// RUN: ktir-opt "%s" | ktir-opt | FileCheck "%s"

// CHECK-LABEL: func.func @future_single
// CHECK-SAME: !ktdp.tile_future<tensor<1x64xf16>>
func.func @future_single(%arg0: !ktdp.tile_future<tensor<1x64xf16>>)
    -> !ktdp.tile_future<tensor<1x64xf16>> {
  return %arg0 : !ktdp.tile_future<tensor<1x64xf16>>
}

// CHECK-LABEL: func.func @future_multi
// CHECK-SAME: !ktdp.tile_future<tensor<128xf32>, tensor<128xi32>>
func.func @future_multi(%arg0: !ktdp.tile_future<tensor<128xf32>, tensor<128xi32>>)
    -> !ktdp.tile_future<tensor<128xf32>, tensor<128xi32>> {
  return %arg0 : !ktdp.tile_future<tensor<128xf32>, tensor<128xi32>>
}
