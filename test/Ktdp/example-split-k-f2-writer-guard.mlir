// RUN: ktir-opt "%s" | ktir-opt | FileCheck "%s"

// Split-K matmul, reduce_to_core form. Smallest kernel that
// exercises `mode = reduce_to_core<0>`: every core contributes a
// partial, only core 0 ends up holding the full sum, and only core 0
// stores to HBM C — guarded by `scf.if pid_k == 0`.
//
// Shapes mirror the simple all_reduce example (1-D grid, 32 cores
// along K). Output C is now [32, 32] (one tile, written once) instead
// of [1024, 32] (32 row-blocks, written by all cores).

#a_set     = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1023 >= 0, d1 >= 0, -d1 + 1023 >= 0)>
#b_set     = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31   >= 0, d1 >= 0, -d1 + 31   >= 0)>
#c_set     = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31   >= 0, d1 >= 0, -d1 + 31   >= 0)>
#a_acc     = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31   >= 0, d1 >= 0, -d1 + 31   >= 0)>
#identity  = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @split_k_f2_writer_guard
// CHECK: ktdp.reduce
// CHECK-SAME: across = #ktdp.grid_axis<0>
// CHECK-SAME: kind = #ktdp.reduce_kind<sum>
// CHECK-SAME: mode = #ktdp.reduce_mode<reduce_to_core<0>>
// CHECK: scf.if
// CHECK: ktdp.store
func.func @split_k_f2_writer_guard(
    %a_ptr: index,
    %b_ptr: index,
    %c_ptr: index
) attributes {grid = array<i64: 32>} {
  %pid_k = ktdp.get_compute_tile_id : index

  %c0   = arith.constant 0  : index
  %TILE = arith.constant 32 : index

  // ---------- A: full [1024, 1024] in HBM ----------
  %a_view = ktdp.construct_memory_view %a_ptr,
              sizes: [1024, 1024], strides: [1024, 1] {
    coordinate_set = #a_set,
    memory_space   = #ktdp.spyre_memory_space<HBM>
  } : memref<1024x1024xf16>

  // ---------- B: this core's [32, 32] K-shard, in HBM ----------
  %b_view = ktdp.construct_memory_view %b_ptr,
              sizes: [32, 32], strides: [32, 1] {
    coordinate_set = #b_set,
    memory_space   = #ktdp.spyre_memory_space<HBM>
  } : memref<32x32xf16>

  // ---------- C: single [32, 32] tile in HBM ----------
  // Only core 0 writes here.
  %c_view = ktdp.construct_memory_view %c_ptr,
              sizes: [32, 32], strides: [32, 1] {
    coordinate_set = #c_set,
    memory_space   = #ktdp.spyre_memory_space<HBM>
  } : memref<32x32xf16>

  // ---------- Per-core K offset into A ----------
  %offs_k = arith.muli %pid_k, %TILE : index

  // ---------- Load A's row 0..31 of this core's K-slice ----------
  // Shape [32, 32]: just one M-tile to keep the example minimal.
  %a_acc = ktdp.construct_access_tile %a_view[%c0, %offs_k] {
    access_tile_set   = #a_acc,
    access_tile_order = #identity
  } : memref<1024x1024xf16> -> !ktdp.access_tile<32x32xindex>

  %a = ktdp.load %a_acc
        : !ktdp.access_tile<32x32xindex> -> tensor<32x32xf16>

  // ---------- Load this core's B-shard ----------
  %b_acc = ktdp.construct_access_tile %b_view[%c0, %c0] {
    access_tile_set   = #b_set,
    access_tile_order = #identity
  } : memref<32x32xf16> -> !ktdp.access_tile<32x32xindex>

  %b = ktdp.load %b_acc
        : !ktdp.access_tile<32x32xindex> -> tensor<32x32xf16>

  // ---------- Local partial matmul ----------
  %c_init = tensor.empty() : tensor<32x32xf16>
  %psum = linalg.matmul
            ins(%a, %b : tensor<32x32xf16>, tensor<32x32xf16>)
            outs(%c_init : tensor<32x32xf16>)
          -> tensor<32x32xf16>

  // ---------- reduce_to_core<0> across grid_axis<0> ----------
  // Result is materialized only on core 0; on other cores `%full`
  // is unspecified / poison.
  %full = ktdp.reduce %psum {
    kind   = #ktdp.reduce_kind<sum>,
    mode   = #ktdp.reduce_mode<reduce_to_core<0>>,
    across = #ktdp.grid_axis<0>
  } : tensor<32x32xf16> -> tensor<32x32xf16>

  // ---------- Writer guard: only core 0 stores to C ----------
  %is_writer = arith.cmpi eq, %pid_k, %c0 : index
  scf.if %is_writer {
    %c_acc = ktdp.construct_access_tile %c_view[%c0, %c0] {
      access_tile_set   = #c_set,
      access_tile_order = #identity
    } : memref<32x32xf16> -> !ktdp.access_tile<32x32xindex>

    ktdp.store %full, %c_acc
      : tensor<32x32xf16>, !ktdp.access_tile<32x32xindex>
  }

  return
}
