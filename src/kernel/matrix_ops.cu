#include "../../include/matrix_ops.h"
#include "../../include/cuda_utils.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>

// 共享内存块大小已在matrix_ops.h中定义
// 使用TILE_SIZE进行矩阵分块计算

// FP8相关常量
#define FP8_SCALE 127.0f
#define FP8_MAX 127
#define FP8_MIN -127

// FP32到FP16转换核函数
__global__ void convertToFP16Kernel(const float* input, __half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        // 检查是否为NaN或Inf
        if (isnan(val) || isinf(val)) {
            output[idx] = __float2half(0.0f);  // 将NaN/Inf转换为0
        } else {
            // 限制在FP16的范围内
            val = fmaxf(fminf(val, 65504.0f), -65504.0f);
            output[idx] = __float2half(val);
        }
    }
}

// FP16到FP32转换核函数
__global__ void convertToFP32Kernel(const __half* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __half2float(input[idx]);
    }
}

// FP32到FP8转换核函数
__global__ void convertToFP8Kernel(const float* input, uint8_t* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // 使用简单的量化方法，将FP32转换为FP8
        float val = input[idx];
        // 限制在[-1, 1]范围内
        val = fmaxf(fminf(val, 1.0f), -1.0f);
        // 量化到[-127, 127]
        val = val * FP8_SCALE;
        // 转换为uint8_t
        output[idx] = (uint8_t)(val + 128);
    }
}

// FP8到FP32转换核函数
__global__ void convertFromFP8Kernel(const uint8_t* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // 将uint8_t转换回float
        float val = (float)(input[idx] - 128) / FP8_SCALE;
        output[idx] = val;
    }
}

// 使用共享内存的FP32矩阵乘法核函数
__global__ void matrixMulFP32SharedKernel(const float* A, const float* B, float* C,
                                        const MatrixDim dimA, const MatrixDim dimB) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // 遍历所有tile
    for (int t = 0; t < (dimA.cols + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 加载数据到共享内存
        if (row < dimA.rows && t * TILE_SIZE + tx < dimA.cols) {
            As[ty][tx] = A[row * dimA.cols + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (t * TILE_SIZE + ty < dimA.cols && col < dimB.cols) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * dimB.cols + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // 计算当前tile的乘积
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // 写入结果
    if (row < dimA.rows && col < dimB.cols) {
        C[row * dimB.cols + col] = sum;
    }
}

// 使用共享内存的FP16矩阵乘法核函数
__global__ void matrixMulFP16SharedKernel(const __half* A, const __half* B, __half* C,
                                        const MatrixDim dimA, const MatrixDim dimB) {
    __shared__ __half As[TILE_SIZE][TILE_SIZE];
    __shared__ __half Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    __half sum = __float2half(0.0f);
    
    // 遍历所有tile
    for (int t = 0; t < (dimA.cols + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 加载数据到共享内存
        if (row < dimA.rows && t * TILE_SIZE + tx < dimA.cols) {
            As[ty][tx] = A[row * dimA.cols + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = __float2half(0.0f);
        }
        
        if (t * TILE_SIZE + ty < dimA.cols && col < dimB.cols) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * dimB.cols + col];
        } else {
            Bs[ty][tx] = __float2half(0.0f);
        }
        
        __syncthreads();
        
        // 计算当前tile的乘积
        for (int k = 0; k < TILE_SIZE; k++) {
            __half a = As[ty][k];
            __half b = Bs[k][tx];
            
            // 检查是否为NaN
            if (__hisnan(a) || __hisnan(b)) {
                continue;  // 跳过NaN值
            }
            
            __half prod = __hmul(a, b);
            if (!__hisnan(prod)) {
                sum = __hadd(sum, prod);
            }
        }
        
        __syncthreads();
    }
    
    // 写入结果
    if (row < dimA.rows && col < dimB.cols) {
        // 检查结果是否为NaN
        if (__hisnan(sum)) {
            C[row * dimB.cols + col] = __float2half(0.0f);
        } else {
            C[row * dimB.cols + col] = sum;
        }
    }
}

// 使用共享内存的FP8矩阵乘法核函数
__global__ void matrixMulFP8SharedKernel(const uint8_t* A, const uint8_t* B, uint8_t* C,
                                       const MatrixDim dimA, const MatrixDim dimB) {
    __shared__ uint8_t As[TILE_SIZE][TILE_SIZE];
    __shared__ uint8_t Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // 遍历所有tile
    for (int t = 0; t < (dimA.cols + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 加载数据到共享内存
        if (row < dimA.rows && t * TILE_SIZE + tx < dimA.cols) {
            As[ty][tx] = A[row * dimA.cols + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 128;  // 0.0 in FP8
        }
        
        if (t * TILE_SIZE + ty < dimA.cols && col < dimB.cols) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * dimB.cols + col];
        } else {
            Bs[ty][tx] = 128;  // 0.0 in FP8
        }
        
        __syncthreads();
        
        // 计算当前tile的乘积
        for (int k = 0; k < TILE_SIZE; k++) {
            float a = (float)(As[ty][k] - 128) / FP8_SCALE;
            float b = (float)(Bs[k][tx] - 128) / FP8_SCALE;
            sum += a * b;
        }
        
        __syncthreads();
    }
    
    // 写入结果
    if (row < dimA.rows && col < dimB.cols) {
        sum = fmaxf(fminf(sum, 1.0f), -1.0f);
        sum = sum * FP8_SCALE;
        C[row * dimB.cols + col] = (uint8_t)(sum + 128);
    }
}

// 使用共享内存的INT8矩阵乘法核函数
__global__ void matrixMulINT8SharedKernel(const int8_t* A, const int8_t* B, int32_t* C,
                                        const MatrixDim dimA, const MatrixDim dimB) {
    __shared__ int8_t As[TILE_SIZE][TILE_SIZE];
    __shared__ int8_t Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    int32_t sum = 0;
    
    // 遍历所有tile
    for (int t = 0; t < (dimA.cols + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 加载数据到共享内存
        if (row < dimA.rows && t * TILE_SIZE + tx < dimA.cols) {
            As[ty][tx] = A[row * dimA.cols + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0;
        }
        
        if (t * TILE_SIZE + ty < dimA.cols && col < dimB.cols) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * dimB.cols + col];
        } else {
            Bs[ty][tx] = 0;
        }
        
        __syncthreads();
        
        // 计算当前tile的乘积
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // 写入结果
    if (row < dimA.rows && col < dimB.cols) {
        C[row * dimB.cols + col] = sum;
    }
}

// 使用共享内存的FP32矩阵乘法
void matrixMulFP32Shared(float* A, float* B, float* C, MatrixDim dimA, MatrixDim dimB) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((dimB.cols + TILE_SIZE - 1) / TILE_SIZE,
                 (dimA.rows + TILE_SIZE - 1) / TILE_SIZE);
    
    matrixMulFP32SharedKernel<<<gridDim, blockDim>>>(A, B, C, dimA, dimB);
    cudaDeviceSynchronize();
}

// 使用共享内存的FP16矩阵乘法
void matrixMulFP16Shared(float* A, float* B, float* C, MatrixDim dimA, MatrixDim dimB) {
    __half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, dimA.rows * dimA.cols * sizeof(__half));
    cudaMalloc(&d_B, dimB.rows * dimB.cols * sizeof(__half));
    cudaMalloc(&d_C, dimA.rows * dimB.cols * sizeof(__half));
    
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((dimB.cols + TILE_SIZE - 1) / TILE_SIZE,
                 (dimA.rows + TILE_SIZE - 1) / TILE_SIZE);
    
    matrixMulFP16SharedKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, dimA, dimB);
    cudaDeviceSynchronize();
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// 使用共享内存的FP8矩阵乘法
void matrixMulFP8Shared(float* A, float* B, float* C, MatrixDim dimA, MatrixDim dimB) {
    uint8_t *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, dimA.rows * dimA.cols * sizeof(uint8_t));
    cudaMalloc(&d_B, dimB.rows * dimB.cols * sizeof(uint8_t));
    cudaMalloc(&d_C, dimA.rows * dimB.cols * sizeof(uint8_t));
    
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((dimB.cols + TILE_SIZE - 1) / TILE_SIZE,
                 (dimA.rows + TILE_SIZE - 1) / TILE_SIZE);
    
    matrixMulFP8SharedKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, dimA, dimB);
    cudaDeviceSynchronize();
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// 使用共享内存的INT8矩阵乘法
void matrixMulINT8Shared(float* A, float* B, float* C, MatrixDim dimA, MatrixDim dimB) {
    int8_t *d_A, *d_B;
    int32_t *d_C;
    cudaMalloc(&d_A, dimA.rows * dimA.cols * sizeof(int8_t));
    cudaMalloc(&d_B, dimB.rows * dimB.cols * sizeof(int8_t));
    cudaMalloc(&d_C, dimA.rows * dimB.cols * sizeof(int32_t));
    
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((dimB.cols + TILE_SIZE - 1) / TILE_SIZE,
                 (dimA.rows + TILE_SIZE - 1) / TILE_SIZE);
    
    matrixMulINT8SharedKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, dimA, dimB);
    cudaDeviceSynchronize();
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// cuBLAS句柄
static cublasHandle_t cublasHandle;

// 初始化cuBLAS
void initCublas() {
    cublasCreate(&cublasHandle);
}

// 清理cuBLAS
void cleanupCublas() {
    cublasDestroy(cublasHandle);
}

// cuBLAS FP32矩阵乘法
void matrixMulCublasFP32(float* A, float* B, float* C, MatrixDim dimA, MatrixDim dimB) {
    float alpha = 1.0f;
    float beta = 0.0f;
    
    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                dimB.cols, dimA.rows, dimA.cols,
                &alpha, B, dimB.cols,
                A, dimA.cols,
                &beta, C, dimB.cols);
}

// cuBLAS FP16矩阵乘法
void matrixMulCublasFP16(float* A, float* B, float* C, MatrixDim dimA, MatrixDim dimB) {
    __half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, dimA.rows * dimA.cols * sizeof(__half));
    cudaMalloc(&d_B, dimB.rows * dimB.cols * sizeof(__half));
    cudaMalloc(&d_C, dimA.rows * dimB.cols * sizeof(__half));
    
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);
    
    cublasHgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                dimB.cols, dimA.rows, dimA.cols,
                &alpha, d_B, dimB.cols,
                d_A, dimA.cols,
                &beta, d_C, dimB.cols);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// 计算相对误差
__global__ void computeRelativeErrorKernel(const float* reference, const float* result, 
                                         float* error, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float ref_val = reference[idx];
        float res_val = result[idx];
        if (fabsf(ref_val) > 1e-6f) {
            error[idx] = fabsf((res_val - ref_val) / ref_val);
        } else {
            error[idx] = fabsf(res_val - ref_val);
        }
    }
}

// 分析量化效果
void analyzeQuantizationEffect(float* A, float* B, float* C, MatrixDim dimA, MatrixDim dimB) {
    // 分配内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, dimA.rows * dimA.cols * sizeof(float));
    cudaMalloc(&d_B, dimB.rows * dimB.cols * sizeof(float));
    cudaMalloc(&d_C, dimA.rows * dimB.cols * sizeof(float));

    // 复制输入数据到设备
    cudaMemcpy(d_A, A, dimA.rows * dimA.cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, dimB.rows * dimB.cols * sizeof(float), cudaMemcpyHostToDevice);

    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 测试FP32性能
    cudaEventRecord(start);
    matrixMulFP32(d_A, d_B, d_C, dimA, dimB);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float fp32_time;
    cudaEventElapsedTime(&fp32_time, start, stop);

    // 测试FP16性能
    cudaEventRecord(start);
    matrixMulFP16(d_A, d_B, d_C, dimA, dimB);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float fp16_time;
    cudaEventElapsedTime(&fp16_time, start, stop);

    // 测试FP8性能
    cudaEventRecord(start);
    matrixMulFP8(d_A, d_B, d_C, dimA, dimB);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float fp8_time;
    cudaEventElapsedTime(&fp8_time, start, stop);

    // 测试INT8性能
    cudaEventRecord(start);
    matrixMulINT8(d_A, d_B, d_C, dimA, dimB);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float int8_time;
    cudaEventElapsedTime(&int8_time, start, stop);

    // 测试融合FP8性能
    cudaEventRecord(start);
    fusedFP8MatrixMul(d_A, d_B, d_C, dimA, dimB);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float fused_fp8_time;
    cudaEventElapsedTime(&fused_fp8_time, start, stop);

    // 测试融合INT8性能
    cudaEventRecord(start);
    fusedINT8MatrixMul(d_A, d_B, d_C, dimA, dimB);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float fused_int8_time;
    cudaEventElapsedTime(&fused_int8_time, start, stop);

    // 打印性能结果
    printf("性能分析结果:\n");
    printf("FP32: %.3f ms\n", fp32_time);
    printf("FP16: %.3f ms (%.2fx 加速)\n", fp16_time, fp32_time/fp16_time);
    printf("FP8: %.3f ms (%.2fx 加速)\n", fp8_time, fp32_time/fp8_time);
    printf("INT8: %.3f ms (%.2fx 加速)\n", int8_time, fp32_time/int8_time);
    printf("融合FP8: %.3f ms (%.2fx 加速)\n", fused_fp8_time, fp32_time/fused_fp8_time);
    printf("融合INT8: %.3f ms (%.2fx 加速)\n", fused_int8_time, fp32_time/fused_int8_time);

    // 清理资源
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// 性能分析函数
void analyzePerformance(float* A, float* B, float* C, MatrixDim dimA, MatrixDim dimB) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float fp32_time, fp16_time, fp8_time, int8_time;
    float cublas_fp32_time, cublas_fp16_time;
    float fused_fp8_time, fused_int8_time;
    
    // 测试FP32
    cudaEventRecord(start);
    matrixMulFP32(A, B, C, dimA, dimB);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&fp32_time, start, stop);
    
    // 测试FP16
    cudaEventRecord(start);
    matrixMulFP16(A, B, C, dimA, dimB);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&fp16_time, start, stop);
    
    // 测试FP8
    cudaEventRecord(start);
    matrixMulFP8(A, B, C, dimA, dimB);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&fp8_time, start, stop);
    
    // 测试INT8
    cudaEventRecord(start);
    matrixMulINT8(A, B, C, dimA, dimB);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&int8_time, start, stop);
    
    // 测试cuBLAS FP32
    cudaEventRecord(start);
    matrixMulCublasFP32(A, B, C, dimA, dimB);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cublas_fp32_time, start, stop);
    
    // 测试cuBLAS FP16
    cudaEventRecord(start);
    matrixMulCublasFP16(A, B, C, dimA, dimB);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cublas_fp16_time, start, stop);

    // 测试融合FP8
    cudaEventRecord(start);
    fusedFP8MatrixMul(A, B, C, dimA, dimB);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&fused_fp8_time, start, stop);

    // 测试融合INT8
    cudaEventRecord(start);
    fusedINT8MatrixMul(A, B, C, dimA, dimB);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&fused_int8_time, start, stop);
    
    printf("\n性能分析结果:\n");
    printf("FP32: %.3f ms\n", fp32_time);
    printf("FP16: %.3f ms (%.2fx 加速)\n", fp16_time, fp32_time/fp16_time);
    printf("FP8: %.3f ms (%.2fx 加速)\n", fp8_time, fp32_time/fp8_time);
    printf("INT8: %.3f ms (%.2fx 加速)\n", int8_time, fp32_time/int8_time);
    printf("cuBLAS FP32: %.3f ms (%.2fx 加速)\n", cublas_fp32_time, fp32_time/cublas_fp32_time);
    printf("cuBLAS FP16: %.3f ms (%.2fx 加速)\n", cublas_fp16_time, fp32_time/cublas_fp16_time);
    printf("融合FP8: %.3f ms (%.2fx 加速)\n", fused_fp8_time, fp32_time/fused_fp8_time);
    printf("融合INT8: %.3f ms (%.2fx 加速)\n", fused_int8_time, fp32_time/fused_int8_time);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// 融合的FP8量化-反量化-矩阵乘法核函数
__global__ void fusedFP8MatrixMulKernel(const float* A, const float* B, float* C,
                                      const MatrixDim dimA, const MatrixDim dimB) {
    __shared__ uint8_t As[TILE_SIZE][TILE_SIZE];
    __shared__ uint8_t Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // 遍历所有tile
    for (int t = 0; t < (dimA.cols + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 加载并量化数据到共享内存
        if (row < dimA.rows && t * TILE_SIZE + tx < dimA.cols) {
            float val = A[row * dimA.cols + t * TILE_SIZE + tx];
            val = fmaxf(fminf(val, 1.0f), -1.0f);
            As[ty][tx] = (uint8_t)(val * FP8_SCALE + 128);
        } else {
            As[ty][tx] = 128;  // 0.0 in FP8
        }
        
        if (t * TILE_SIZE + ty < dimA.cols && col < dimB.cols) {
            float val = B[(t * TILE_SIZE + ty) * dimB.cols + col];
            val = fmaxf(fminf(val, 1.0f), -1.0f);
            Bs[ty][tx] = (uint8_t)(val * FP8_SCALE + 128);
        } else {
            Bs[ty][tx] = 128;  // 0.0 in FP8
        }
        
        __syncthreads();
        
        // 计算当前tile的乘积
        for (int k = 0; k < TILE_SIZE; k++) {
            float a = (float)(As[ty][k] - 128) / FP8_SCALE;
            float b = (float)(Bs[k][tx] - 128) / FP8_SCALE;
            sum += a * b;
        }
        
        __syncthreads();
    }
    
    // 写入结果
    if (row < dimA.rows && col < dimB.cols) {
        C[row * dimB.cols + col] = sum;
    }
}

// 融合的INT8量化-反量化-矩阵乘法核函数
__global__ void fusedINT8MatrixMulKernel(const float* A, const float* B, float* C,
                                       const MatrixDim dimA, const MatrixDim dimB) {
    __shared__ int8_t As[TILE_SIZE][TILE_SIZE];
    __shared__ int8_t Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // 遍历所有tile
    for (int t = 0; t < (dimA.cols + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 加载并量化数据到共享内存
        if (row < dimA.rows && t * TILE_SIZE + tx < dimA.cols) {
            float val = A[row * dimA.cols + t * TILE_SIZE + tx];
            val = fmaxf(fminf(val, 1.0f), -1.0f);
            As[ty][tx] = (int8_t)(val * 127.0f);
        } else {
            As[ty][tx] = 0;
        }
        
        if (t * TILE_SIZE + ty < dimA.cols && col < dimB.cols) {
            float val = B[(t * TILE_SIZE + ty) * dimB.cols + col];
            val = fmaxf(fminf(val, 1.0f), -1.0f);
            Bs[ty][tx] = (int8_t)(val * 127.0f);
        } else {
            Bs[ty][tx] = 0;
        }
        
        __syncthreads();
        
        // 计算当前tile的乘积
        for (int k = 0; k < TILE_SIZE; k++) {
            float a = (float)As[ty][k] / 127.0f;
            float b = (float)Bs[k][tx] / 127.0f;
            sum += a * b;
        }
        
        __syncthreads();
    }
    
    // 写入结果
    if (row < dimA.rows && col < dimB.cols) {
        C[row * dimB.cols + col] = sum;
    }
}

// 融合的FP8量化-反量化-矩阵乘法
void fusedFP8MatrixMul(float* A, float* B, float* C, MatrixDim dimA, MatrixDim dimB) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((dimB.cols + TILE_SIZE - 1) / TILE_SIZE,
                 (dimA.rows + TILE_SIZE - 1) / TILE_SIZE);
    
    fusedFP8MatrixMulKernel<<<gridDim, blockDim>>>(A, B, C, dimA, dimB);
    cudaDeviceSynchronize();
}

// 融合的INT8量化-反量化-矩阵乘法
void fusedINT8MatrixMul(float* A, float* B, float* C, MatrixDim dimA, MatrixDim dimB) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((dimB.cols + TILE_SIZE - 1) / TILE_SIZE,
                 (dimA.rows + TILE_SIZE - 1) / TILE_SIZE);
    
    fusedINT8MatrixMulKernel<<<gridDim, blockDim>>>(A, B, C, dimA, dimB);
    cudaDeviceSynchronize();
}

// 基础矩阵乘法函数实现
void matrixMulFP32(float* A, float* B, float* C, MatrixDim dimA, MatrixDim dimB) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((dimB.cols + TILE_SIZE - 1) / TILE_SIZE,
                 (dimA.rows + TILE_SIZE - 1) / TILE_SIZE);
    
    matrixMulFP32SharedKernel<<<gridDim, blockDim>>>(A, B, C, dimA, dimB);
    cudaDeviceSynchronize();
}

void matrixMulFP16(float* A, float* B, float* C, MatrixDim dimA, MatrixDim dimB) {
    // 分配FP16内存
    __half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, dimA.rows * dimA.cols * sizeof(__half));
    cudaMalloc(&d_B, dimB.rows * dimB.cols * sizeof(__half));
    cudaMalloc(&d_C, dimA.rows * dimB.cols * sizeof(__half));

    // 转换为FP16
    dim3 blockDim(256);
    dim3 gridDim((dimA.rows * dimA.cols + blockDim.x - 1) / blockDim.x);
    convertToFP16Kernel<<<gridDim, blockDim>>>(A, d_A, dimA.rows * dimA.cols);
    
    gridDim.x = (dimB.rows * dimB.cols + blockDim.x - 1) / blockDim.x;
    convertToFP16Kernel<<<gridDim, blockDim>>>(B, d_B, dimB.rows * dimB.cols);

    // 执行FP16矩阵乘法
    dim3 matBlockDim(TILE_SIZE, TILE_SIZE);
    dim3 matGridDim((dimB.cols + matBlockDim.x - 1) / matBlockDim.x,
                    (dimA.rows + matBlockDim.y - 1) / matBlockDim.y);
    matrixMulFP16SharedKernel<<<matGridDim, matBlockDim>>>(d_A, d_B, d_C, dimA, dimB);

    // 转换回FP32
    gridDim.x = (dimA.rows * dimB.cols + blockDim.x - 1) / blockDim.x;
    convertToFP32Kernel<<<gridDim, blockDim>>>(d_C, C, dimA.rows * dimB.cols);

    cudaDeviceSynchronize();

    // 清理内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void matrixMulFP8(float* A, float* B, float* C, MatrixDim dimA, MatrixDim dimB) {
    // 分配FP8内存
    uint8_t *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, dimA.rows * dimA.cols * sizeof(uint8_t));
    cudaMalloc(&d_B, dimB.rows * dimB.cols * sizeof(uint8_t));
    cudaMalloc(&d_C, dimA.rows * dimB.cols * sizeof(uint8_t));

    // 转换为FP8
    dim3 blockDim(256);
    dim3 gridDim((dimA.rows * dimA.cols + blockDim.x - 1) / blockDim.x);
    convertToFP8Kernel<<<gridDim, blockDim>>>(A, d_A, dimA.rows * dimA.cols);
    
    gridDim.x = (dimB.rows * dimB.cols + blockDim.x - 1) / blockDim.x;
    convertToFP8Kernel<<<gridDim, blockDim>>>(B, d_B, dimB.rows * dimB.cols);

    // 执行FP8矩阵乘法
    dim3 matBlockDim(TILE_SIZE, TILE_SIZE);
    dim3 matGridDim((dimB.cols + matBlockDim.x - 1) / matBlockDim.x,
                    (dimA.rows + matBlockDim.y - 1) / matBlockDim.y);
    matrixMulFP8SharedKernel<<<matGridDim, matBlockDim>>>(d_A, d_B, d_C, dimA, dimB);

    // 转换回FP32
    gridDim.x = (dimA.rows * dimB.cols + blockDim.x - 1) / blockDim.x;
    convertFromFP8Kernel<<<gridDim, blockDim>>>(d_C, C, dimA.rows * dimB.cols);

    cudaDeviceSynchronize();

    // 清理内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void matrixMulINT8(float* A, float* B, float* C, MatrixDim dimA, MatrixDim dimB) {
    // 分配INT8内存
    int8_t *d_A, *d_B;
    int32_t *d_C;
    cudaMalloc(&d_A, dimA.rows * dimA.cols * sizeof(int8_t));
    cudaMalloc(&d_B, dimB.rows * dimB.cols * sizeof(int8_t));
    cudaMalloc(&d_C, dimA.rows * dimB.cols * sizeof(int32_t));

    // 转换为INT8
    dim3 blockDim(256);
    dim3 gridDim((dimA.rows * dimA.cols + blockDim.x - 1) / blockDim.x);
    quantizeToINT8Kernel<<<gridDim, blockDim>>>(A, d_A, dimA.rows * dimA.cols);
    
    gridDim.x = (dimB.rows * dimB.cols + blockDim.x - 1) / blockDim.x;
    quantizeToINT8Kernel<<<gridDim, blockDim>>>(B, d_B, dimB.rows * dimB.cols);

    // 执行INT8矩阵乘法
    dim3 matBlockDim(TILE_SIZE, TILE_SIZE);
    dim3 matGridDim((dimB.cols + matBlockDim.x - 1) / matBlockDim.x,
                    (dimA.rows + matBlockDim.y - 1) / matBlockDim.y);
    matrixMulINT8SharedKernel<<<matGridDim, matBlockDim>>>(d_A, d_B, d_C, dimA, dimB);

    // 转换回FP32
    gridDim.x = (dimA.rows * dimB.cols + blockDim.x - 1) / blockDim.x;
    dequantizeFromINT8Kernel<<<gridDim, blockDim>>>(d_C, C, dimA.rows * dimB.cols);

    cudaDeviceSynchronize();

    // 清理内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// INT8量化和反量化核函数
__global__ void quantizeToINT8Kernel(const float* input, int8_t* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        // 限制在[-1, 1]范围内
        val = fmaxf(fminf(val, 1.0f), -1.0f);
        // 量化到[-127, 127]
        val = val * 127.0f;
        // 转换为int8_t
        output[idx] = (int8_t)val;
    }
}

__global__ void dequantizeFromINT8Kernel(const int32_t* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // 将int32_t转换回float
        float val = (float)input[idx] / 127.0f;
        output[idx] = val;
    }
} 