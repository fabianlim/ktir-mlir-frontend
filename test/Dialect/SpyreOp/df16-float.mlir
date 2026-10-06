// RUN: ktir-opt "%s" | ktir-opt | FileCheck "%s"
// RUN: ktir-opt "%s" --emit-bytecode | ktir-opt | FileCheck "%s"
// RUN: ktir-opt "%s" --mlir-print-op-generic | ktir-opt | FileCheck "%s"

// `!spyreop.df16` as an element type of `arith`, `math`, `linalg` and `ktdp`,
// in the shapes the Triton Spyre pipeline produces (reduced from the KTIR of
// its `softmax_on_stick` fixture). Parsing at all is what `FloatTypeInterface`
// buys: without it every `arith`/`math` op here fails its float-like operand
// constraint.

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#set1 = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>

// CHECK-LABEL: func.func @softmax_df16(
func.func @softmax_df16(%arg0: index, %arg1: index, %arg2: index) {
  %c0 = arith.constant 0 : index
  // The reduction identities. The hex spelling is the placeholder IEEE-half
  // encoding of -inf, which is what makes this a placeholder.
  // CHECK-DAG: arith.constant 0.000000e+00 : !spyreop.df16
  // CHECK-DAG: arith.constant 0xFC00 : !spyreop.df16
  // CHECK-DAG: arith.constant dense<1.000000e+00> : tensor<64x!spyreop.df16>
  %zero = arith.constant 0.000000e+00 : !spyreop.df16
  %ninf = arith.constant 0xFC00 : !spyreop.df16
  %one = arith.constant dense<1.000000e+00> : tensor<64x!spyreop.df16>

  // CHECK: ktdp.construct_memory_view {{.*}} : memref<64x128x!spyreop.df16>
  %in = ktdp.construct_memory_view %arg0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x128x!spyreop.df16>
  %out = ktdp.construct_memory_view %arg1, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x128x!spyreop.df16>
  %wide = ktdp.construct_memory_view %arg2, sizes: [64], strides: [1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<64xf32>

  // CHECK: ktdp.load {{.*}} -> tensor<64x128x!spyreop.df16>
  %t = ktdp.construct_access_tile %in[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128x!spyreop.df16> -> !ktdp.access_tile<64x128xindex>
  %x = ktdp.load %t : <64x128xindex> -> tensor<64x128x!spyreop.df16>

  // CHECK: linalg.reduce
  // CHECK: arith.maxnumf %{{.*}}, %{{.*}} : !spyreop.df16
  %e0 = tensor.empty() : tensor<64x!spyreop.df16>
  %f0 = linalg.fill ins(%ninf : !spyreop.df16) outs(%e0 : tensor<64x!spyreop.df16>) -> tensor<64x!spyreop.df16>
  %max = linalg.reduce ins(%x : tensor<64x128x!spyreop.df16>) outs(%f0 : tensor<64x!spyreop.df16>) dimensions = [1]
    (%a: !spyreop.df16, %b: !spyreop.df16) {
      %m = arith.maxnumf %a, %b : !spyreop.df16
      linalg.yield %m : !spyreop.df16
    }

  // CHECK: arith.subf {{.*}} : tensor<64x128x!spyreop.df16>
  // CHECK: math.exp {{.*}} : tensor<64x128x!spyreop.df16>
  %e1 = tensor.empty() : tensor<64x128x!spyreop.df16>
  %bmax = linalg.broadcast ins(%max : tensor<64x!spyreop.df16>) outs(%e1 : tensor<64x128x!spyreop.df16>) dimensions = [1]
  %shifted = arith.subf %x, %bmax : tensor<64x128x!spyreop.df16>
  %exp = math.exp %shifted : tensor<64x128x!spyreop.df16>

  // CHECK: arith.addf %{{.*}}, %{{.*}} : !spyreop.df16
  %f1 = linalg.fill ins(%zero : !spyreop.df16) outs(%e0 : tensor<64x!spyreop.df16>) -> tensor<64x!spyreop.df16>
  %sum = linalg.reduce ins(%exp : tensor<64x128x!spyreop.df16>) outs(%f1 : tensor<64x!spyreop.df16>) dimensions = [1]
    (%a: !spyreop.df16, %b: !spyreop.df16) {
      %s = arith.addf %a, %b : !spyreop.df16
      linalg.yield %s : !spyreop.df16
    }

  // CHECK: arith.divf {{.*}} : tensor<64x!spyreop.df16>
  // CHECK: arith.mulf {{.*}} : tensor<64x128x!spyreop.df16>
  %recip = arith.divf %one, %sum : tensor<64x!spyreop.df16>
  %brecip = linalg.broadcast ins(%recip : tensor<64x!spyreop.df16>) outs(%e1 : tensor<64x128x!spyreop.df16>) dimensions = [1]
  %y = arith.mulf %exp, %brecip : tensor<64x128x!spyreop.df16>

  // CHECK: ktdp.store {{.*}} : tensor<64x128x!spyreop.df16>
  %ot = ktdp.construct_access_tile %out[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128x!spyreop.df16> -> !ktdp.access_tile<64x128xindex>
  ktdp.store %y, %ot : tensor<64x128x!spyreop.df16>, <64x128xindex>

  // The widening and narrowing casts the pipeline emits around f32 compute.
  // CHECK: arith.extf {{.*}} : tensor<64x!spyreop.df16> to tensor<64xf32>
  // CHECK: arith.truncf {{.*}} : tensor<64xf32> to tensor<64x!spyreop.df16>
  %w = arith.extf %sum : tensor<64x!spyreop.df16> to tensor<64xf32>
  %n = arith.truncf %w : tensor<64xf32> to tensor<64x!spyreop.df16>
  %wt = ktdp.construct_access_tile %wide[%c0] {access_tile_order = #map1, access_tile_set = #set1} : memref<64xf32> -> !ktdp.access_tile<64xindex>
  ktdp.store %w, %wt : tensor<64xf32>, <64xindex>
  return
}
