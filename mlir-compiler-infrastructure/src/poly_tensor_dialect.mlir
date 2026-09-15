// ==============================================================================
// Multi-Level Intermediate Representation (MLIR) Dialect Lowering Pipeline
// Dialect: poly_engine -> affine -> vector -> llvm
// ==============================================================================

module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"} {

  // Stage 1: High-Level Polyhedral Tensor Contraction
  func.func @poly_tensor_contraction_4x4(
      %A: memref<4x4xf32>,
      %B: memref<4x4xf32>,
      %C: memref<4x4xf32>
  ) {
    // Stage 2: Lowered Affine Polyhedral Nest with Tiling & Loop Fusion
    affine.for %i = 0 to 4 {
      affine.for %j = 0 to 4 {
        %c_val = affine.load %C[%i, %j] : memref<4x4xf32>
        %sum = affine.for %k = 0 to 4 iter_args(%acc = %c_val) -> (f32) {
          %a_val = affine.load %A[%i, %k] : memref<4x4xf32>
          %b_val = affine.load %B[%k, %j] : memref<4x4xf32>
          %prod = arith.mulf %a_val, %b_val : f32
          %next_acc = arith.addf %acc, %prod : f32
          affine.yield %next_acc : f32
        }
        affine.store %sum, %C[%i, %j] : memref<4x4xf32>
      }
    }
    return
  }

  // Stage 3: Lowered Vector Dialect SIMD 4-lane FMA
  func.func @vector_fma_4xf32(%a: vector<4xf32>, %b: vector<4xf32>, %c: vector<4xf32>) -> vector<4xf32> {
    %res = vector.fma %a, %b, %c : vector<4xf32>
    return %res : vector<4xf32>
  }
}
