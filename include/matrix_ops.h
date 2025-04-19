#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include <stdint.h>

// 共享内存块大小，用于矩阵乘法的分块计算
// 较小的值可以减少共享内存使用，但可能影响性能
// 较大的值可能提高性能，但需要更多共享内存
#define TILE_SIZE 8

#ifdef __cplusplus
extern "C" {
#endif

// 矩阵维度结构体
typedef struct {
    int rows;
    int cols;
} MatrixDim;

// 核函数声明
__global__ void quantizeToINT8Kernel(const float* input, int8_t* output, int size);
__global__ void dequantizeFromINT8Kernel(const int32_t* input, float* output, int size);
__global__ void convertToFP16Kernel(const float* input, __half* output, int size);
__global__ void convertToFP32Kernel(const __half* input, float* output, int size);
__global__ void convertToFP8Kernel(const float* input, uint8_t* output, int size);
__global__ void convertFromFP8Kernel(const uint8_t* input, float* output, int size);

// 共享内存核函数声明
__global__ void matrixMulFP32SharedKernel(const float* A, const float* B, float* C,
                                        const MatrixDim dimA, const MatrixDim dimB);
__global__ void matrixMulFP16SharedKernel(const __half* A, const __half* B, __half* C,
                                        const MatrixDim dimA, const MatrixDim dimB);
__global__ void matrixMulFP8SharedKernel(const uint8_t* A, const uint8_t* B, uint8_t* C,
                                       const MatrixDim dimA, const MatrixDim dimB);
__global__ void matrixMulINT8SharedKernel(const int8_t* A, const int8_t* B, int32_t* C,
                                        const MatrixDim dimA, const MatrixDim dimB);

// cuBLAS初始化和清理
void initCublas();
void cleanupCublas();

// 矩阵乘法函数
void matrixMulFP32(float* A, float* B, float* C, MatrixDim dimA, MatrixDim dimB);
void matrixMulFP16(float* A, float* B, float* C, MatrixDim dimA, MatrixDim dimB);
void matrixMulFP8(float* A, float* B, float* C, MatrixDim dimA, MatrixDim dimB);
void matrixMulINT8(float* A, float* B, float* C, MatrixDim dimA, MatrixDim dimB);

// 使用共享内存的矩阵乘法函数
void matrixMulFP32Shared(float* A, float* B, float* C, MatrixDim dimA, MatrixDim dimB);
void matrixMulFP16Shared(float* A, float* B, float* C, MatrixDim dimA, MatrixDim dimB);
void matrixMulFP8Shared(float* A, float* B, float* C, MatrixDim dimA, MatrixDim dimB);
void matrixMulINT8Shared(float* A, float* B, float* C, MatrixDim dimA, MatrixDim dimB);

// cuBLAS矩阵乘法函数
void matrixMulCublasFP32(float* A, float* B, float* C, MatrixDim dimA, MatrixDim dimB);
void matrixMulCublasFP16(float* A, float* B, float* C, MatrixDim dimA, MatrixDim dimB);

// 量化函数
void quantizeToFP8(const float* input, uint8_t* output, int size);
void quantizeToINT8(const float* input, int8_t* output, int size);

// 反量化函数
void dequantizeFromFP8(const uint8_t* input, float* output, int size);
void dequantizeFromINT8(const int8_t* input, float* output, int size);

// 性能分析函数
void analyzePerformance(float* A, float* B, float* C, MatrixDim dimA, MatrixDim dimB);

// 量化效果分析函数
void analyzeQuantizationEffect(float* A, float* B, float* C, MatrixDim dimA, MatrixDim dimB);

// 融合的量化-反量化-矩阵乘法函数
void fusedFP8MatrixMul(float* A, float* B, float* C, MatrixDim dimA, MatrixDim dimB);
void fusedINT8MatrixMul(float* A, float* B, float* C, MatrixDim dimA, MatrixDim dimB);

#ifdef __cplusplus
}
#endif

#endif // MATRIX_OPS_H 