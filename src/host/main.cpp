#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <chrono>
#include "../../include/matrix_ops.h"
#include "../../include/cuda_utils.h"

// 初始化随机矩阵
void initRandomMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;  // 范围[-1, 1]
    }
}

// 验证结果
void verifyResults(const float* A, const float* B, const float* C, 
                  const MatrixDim dimA, const MatrixDim dimB) {
    float maxError = 0.0f;
    int errorCount = 0;
    
    for (int i = 0; i < dimA.rows; i++) {
        for (int j = 0; j < dimB.cols; j++) {
            float expected = 0.0f;
            for (int k = 0; k < dimA.cols; k++) {
                expected += A[i * dimA.cols + k] * B[k * dimB.cols + j];
            }
            
            float error = fabs(C[i * dimB.cols + j] - expected);
            maxError = fmax(maxError, error);
            if (error > 1e-5) {
                errorCount++;
            }
        }
    }
    
    printf("最大误差: %e\n", maxError);
    printf("误差超过阈值的元素数量: %d\n", errorCount);
}

// 性能测试函数
void benchmarkMatrixMul(const float* d_A, const float* d_B, float* d_C,
                       const MatrixDim dimA, const MatrixDim dimB,
                       void (*matrixMulFunc)(const float*, const float*, float*, const MatrixDim, const MatrixDim),
                       const char* precision) {
    const int numIterations = 10;
    float totalTime = 0.0f;
    
    // 预热
    matrixMulFunc(d_A, d_B, d_C, dimA, dimB);
    checkCudaError(cudaDeviceSynchronize());
    
    // 性能测试
    for (int i = 0; i < numIterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        matrixMulFunc(d_A, d_B, d_C, dimA, dimB);
        checkCudaError(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        
        float time = std::chrono::duration<float, std::milli>(end - start).count();
        totalTime += time;
    }
    
    float avgTime = totalTime / numIterations;
    float gflops = (2.0f * dimA.rows * dimA.cols * dimB.cols) / (avgTime * 1e6);
    
    printf("%s 矩阵乘法性能:\n", precision);
    printf("  平均时间: %.2f ms\n", avgTime);
    printf("  计算性能: %.2f GFLOPS\n", gflops);
}

int main() {
    // 打印CUDA设备信息
    printDeviceInfo();
    
    // 设置矩阵大小
    const int M = 2048;  // A矩阵行数
    const int K = 2048;  // A矩阵列数/B矩阵行数
    const int N = 2048;  // B矩阵列数
    
    MatrixDim dimA = {M, K};
    MatrixDim dimB = {K, N};
    
    // 分配主机内存
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));
    
    // 初始化随机矩阵
    srand(time(NULL));
    for (int i = 0; i < M * K; i++) {
        h_A[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    
    // 分配设备内存
    float *d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc(&d_A, M * K * sizeof(float)));
    checkCudaError(cudaMalloc(&d_B, K * N * sizeof(float)));
    checkCudaError(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    // 复制数据到设备
    checkCudaError(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // 运行性能分析
    analyzePerformance(d_A, d_B, d_C, dimA, dimB);
    
    // 释放内存
    free(h_A);
    free(h_B);
    free(h_C);
    checkCudaError(cudaFree(d_A));
    checkCudaError(cudaFree(d_B));
    checkCudaError(cudaFree(d_C));
    
    return 0;
} 