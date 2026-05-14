// RUN: ktir-opt "%s" | ktir-opt | FileCheck "%s"

// Split-K matmul, 1-D grid, all_reduce form.
//
//   - 32-core 1-D grid, axis 0 distributes the K dimension.
//   - A is [1024, 1024] (HBM), B per-core shard is [32, 32] (HBM),
//     C is [1024, 32] (HBM).
//   - Each core: load its K-column-range of A, load its B-shard,
//     local matmul -> [1024, 32] partial, all_reduce(sum) across
//     grid axis 0, extract its 32-row block, store to C.
//
// Smallest complete all_reduce kernel: every core participates, every
// core ends with the replicated full sum, and every core stores a
// unique row-block (no writer guard).
//
// Picture (drawn at N = 4 cores for clarity; the kernel below uses
// N = 32). Each core c owns a TILE-wide K-slice of A and one B-shard.
// After all_reduce, every core holds the full [M_FULL, W] product;
// each then extracts a unique row-block of C and stores it.
//
//   A  [N·TILE, N·TILE]                  B [N·TILE, W]   C [N·TILE, W]
//   ┌──────┬──────┬──────┬──────┐        ┌──────┐        ┌──────┐
//   │ A0,0 │ A0,1 │ A0,2 │ A0,3 │        │ B_0  │◄ c=0   │ C_0  │◄ c=0
//   ├──────┼──────┼──────┼──────┤        ├──────┤        ├──────┤
//   │ A1,0 │ A1,1 │ A1,2 │ A1,3 │        │ B_1  │◄ c=1   │ C_1  │◄ c=1
//   ├──────┼──────┼──────┼──────┤        ├──────┤        ├──────┤
//   │ A2,0 │ A2,1 │ A2,2 │ A2,3 │        │ B_2  │◄ c=2   │ C_2  │◄ c=2
//   ├──────┼──────┼──────┼──────┤        ├──────┤        ├──────┤
//   │ A3,0 │ A3,1 │ A3,2 │ A3,3 │        │ B_3  │◄ c=3   │ C_3  │◄ c=3
//   └──────┴──────┴──────┴──────┘        └──────┘        └──────┘
//             ▲                                            ▲
//             │ each core c reads the full K-column
//             │ A[:, c·TILE : (c+1)·TILE]
//             │
//   per-core flow (one core c):
//     a_data = A[:, c·TILE..(c+1)·TILE]   (shape [M_FULL, TILE])
//     b_data = B_c                        (shape [TILE, W])
//     psum   = a_data @ b_data            (local, contracts c's TILE)
//     full   = all_reduce(sum, axis=0) psum   (every core holds full)
//     C_c    = full[c·TILE:(c+1)·TILE, :] (each core's unique row-block)
//     store C_c -> C[c·TILE.., :]         (32 disjoint HBM writes)

#a_set     = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1023 >= 0, d1 >= 0, -d1 + 1023 >= 0)>
#b_set     = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31   >= 0, d1 >= 0, -d1 + 31   >= 0)>
#c_set     = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1023 >= 0, d1 >= 0, -d1 + 31   >= 0)>
#a_acc     = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1023 >= 0, d1 >= 0, -d1 + 31   >= 0)>
#shard_acc = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31   >= 0, d1 >= 0, -d1 + 31   >= 0)>
#identity  = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @split_k_all_reduce
// CHECK: ktdp.reduce
// CHECK-SAME: across = #ktdp.grid_axis<0>
// CHECK-SAME: kind = #ktdp.reduce_kind<sum>
// CHECK-SAME: mode = #ktdp.reduce_mode<all_reduce>
// CHECK-SAME: tensor<1024x32xf16> -> tensor<1024x32xf16>
// Distinguishing pattern for all_reduce form: extract a per-core slice,
// then store it (no writer guard) — every core writes a unique block.
// CHECK: tensor.extract_slice
// CHECK: ktdp.store
func.func @split_k_all_reduce(
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

  // ---------- B: this core's [32, 32] shard, in HBM ----------
  %b_view = ktdp.construct_memory_view %b_ptr,
              sizes: [32, 32], strides: [32, 1] {
    coordinate_set = #b_set,
    memory_space   = #ktdp.spyre_memory_space<HBM>
  } : memref<32x32xf16>

  // ---------- C: full [1024, 32] in HBM ----------
  %c_view = ktdp.construct_memory_view %c_ptr,
              sizes: [1024, 32], strides: [32, 1] {
    coordinate_set = #c_set,
    memory_space   = #ktdp.spyre_memory_space<HBM>
  } : memref<1024x32xf16>

  // ---------- Per-core column offset into A ----------
  %offs_k = arith.muli %pid_k, %TILE : index

  // ---------- Load A's full column-range for this core ----------
  // Shape [1024, 32]
  %a_acc = ktdp.construct_access_tile %a_view[%c0, %offs_k] {
    access_tile_set   = #a_acc,
    access_tile_order = #identity
  } : memref<1024x1024xf16> -> !ktdp.access_tile<1024x32xindex>

  %a = ktdp.load %a_acc
        : !ktdp.access_tile<1024x32xindex> -> tensor<1024x32xf16>

  // ---------- Load this core's B-shard ----------
  %b_acc = ktdp.construct_access_tile %b_view[%c0, %c0] {
    access_tile_set   = #b_set,
    access_tile_order = #identity
  } : memref<32x32xf16> -> !ktdp.access_tile<32x32xindex>

  %b = ktdp.load %b_acc
        : !ktdp.access_tile<32x32xindex> -> tensor<32x32xf16>

  // ---------- Local partial matmul ----------
  %c_init = tensor.empty() : tensor<1024x32xf16>
  %psum = linalg.matmul
            ins(%a, %b : tensor<1024x32xf16>, tensor<32x32xf16>)
            outs(%c_init : tensor<1024x32xf16>)
          -> tensor<1024x32xf16>

  // ---------- all_reduce(sum) across grid_axis<0> ----------
  // Sugar form: kind = sum, mode = all_reduce.
  %full = ktdp.reduce %psum {
    kind   = #ktdp.reduce_kind<sum>,
    mode   = #ktdp.reduce_mode<all_reduce>,
    across = #ktdp.grid_axis<0>
  } : tensor<1024x32xf16> -> tensor<1024x32xf16>

  // ---------- Extract this core's row-block ----------
  %offs_m = arith.muli %pid_k, %TILE : index
  %shard = tensor.extract_slice %full[%offs_m, 0] [32, 32] [1, 1]
            : tensor<1024x32xf16> to tensor<32x32xf16>

  // ---------- Store this core's row-block to HBM C ----------
  %c_acc = ktdp.construct_access_tile %c_view[%offs_m, %c0] {
    access_tile_set   = #shard_acc,
    access_tile_order = #identity
  } : memref<1024x32xf16> -> !ktdp.access_tile<32x32xindex>

  ktdp.store %shard, %c_acc
    : tensor<32x32xf16>, !ktdp.access_tile<32x32xindex>

  return
}
