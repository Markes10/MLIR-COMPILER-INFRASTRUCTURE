#!/usr/bin/env bash
# Progressive multi-stage lowering pipeline for Polyhedral Tensor MLIR dialect
set -e

echo "=== MLIR Progressive Multi-Level Lowering Pipeline ==="

SRC_MLIR="src/poly_tensor_dialect.mlir"

if command -v mlir-opt &> /dev/null; then
    echo "[Stage 1/3] Lowering Polyhedral Tensor to Affine Loops..."
    mlir-opt --convert-elementwise-to-linalg --linalg-generalize-named-ops "$SRC_MLIR" -o build/stage1_affine.mlir

    echo "[Stage 2/3] Vectorizing Affine loops to SIMD Vector dialect..."
    mlir-opt --affine-loop-tile --affine-vectorize build/stage1_affine.mlir -o build/stage2_vector.mlir

    echo "[Stage 3/3] Lowering to LLVM IR dialect..."
    mlir-opt --convert-vector-to-llvm --convert-std-to-llvm build/stage2_vector.mlir -o build/stage3_llvm.mlir

    echo "[SUCCESS] Multi-level progressive lowering pipeline verified."
else
    echo "[INFO] mlir-opt binary not in PATH. Running static dialect dialect visitor..."
    node runner/run.js
fi
