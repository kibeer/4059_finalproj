#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>
#include "../include/matrix_ops.h"
#include "../include/cuda_utils.h"

// 测试矩阵维度
const int TEST_SIZES[] = {64, 128, 256, 512, 1024, 2048};
const int NUM_SIZES = sizeof(TEST_SIZES) / sizeof(TEST_SIZES[0]);

// 性能测试结果结构体
struct PerformanceResult {
    float avgTime;
    float gflops;
    float memoryBandwidth;
    float maxError;
    float avgError;
    float stdError;
};

// 误差统计结构体
struct ErrorStats {
    float maxError;
    float avgError;
    float stdDev;
};

// 初始化随机矩阵
void initRandomMatrix(float* matrix, int size) {
    if (!matrix) return;
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;  // 范围[-1, 1]
    }
}

// CPU矩阵乘法
void cpuMatrixMul(const float* A, const float* B, float* C, 
                  const MatrixDim dimA, const MatrixDim dimB) {
    for (int i = 0; i < dimA.rows; i++) {
        for (int j = 0; j < dimB.cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < dimA.cols; k++) {
                sum += A[i * dimA.cols + k] * B[k * dimB.cols + j];
            }
            C[i * dimB.cols + j] = sum;
        }
    }
}

// 验证结果
void verifyResults(const float* A, const float* B, const float* C, 
                  const MatrixDim dimA, const MatrixDim dimB,
                  const char* precision,
                  float& maxError, float& avgError, float& stdError) {
    if (!C) {
        printf("错误：无效的设备内存指针\n");
        return;
    }
    
    // 分配主机内存
    float* h_C = new float[dimA.rows * dimB.cols];
    float* ref_C = new float[dimA.rows * dimB.cols];
    float* h_A = new float[dimA.rows * dimA.cols];
    float* h_B = new float[dimB.rows * dimB.cols];
    
    if (!h_C || !ref_C || !h_A || !h_B) {
        printf("错误：无法分配主机内存\n");
        if (h_C) delete[] h_C;
        if (ref_C) delete[] ref_C;
        if (h_A) delete[] h_A;
        if (h_B) delete[] h_B;
        return;
    }
    
    // 从设备复制数据到主机
    cudaError_t err = cudaMemcpy(h_C, C, dimA.rows * dimB.cols * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("错误：无法从设备复制数据: %s\n", cudaGetErrorString(err));
        delete[] h_C;
        delete[] ref_C;
        delete[] h_A;
        delete[] h_B;
        return;
    }
    
    err = cudaMemcpy(h_A, A, dimA.rows * dimA.cols * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("错误：无法从设备复制数据: %s\n", cudaGetErrorString(err));
        delete[] h_C;
        delete[] ref_C;
        delete[] h_A;
        delete[] h_B;
        return;
    }
    
    err = cudaMemcpy(h_B, B, dimB.rows * dimB.cols * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("错误：无法从设备复制数据: %s\n", cudaGetErrorString(err));
        delete[] h_C;
        delete[] ref_C;
        delete[] h_A;
        delete[] h_B;
        return;
    }
    
    // 使用CPU计算参考结果
    cpuMatrixMul(h_A, h_B, ref_C, dimA, dimB);
    
    // 计算相对误差
    maxError = 0.0f;
    avgError = 0.0f;
    stdError = 0.0f;
    int validCount = 0;  // 用于计算有效的误差样本数
    
    // 第一遍：计算最大误差和平均误差
    for (int i = 0; i < dimA.rows * dimB.cols; i++) {
        float ref_val = ref_C[i];
        float gpu_val = h_C[i];
        
        // 检查是否为有效值
        if (isnan(ref_val) || isnan(gpu_val) || 
            isinf(ref_val) || isinf(gpu_val)) {
            continue;
        }
        
        float abs_error = fabs(gpu_val - ref_val);
        float rel_error;
        
        if (fabs(ref_val) > 1e-6f) {
            rel_error = abs_error / fabs(ref_val);
        } else if (abs_error < 1e-6f) {
            rel_error = 0.0f;  // 如果参考值和计算值都接近0，则误差为0
        } else {
            rel_error = abs_error;  // 如果参考值接近0但计算值不是，使用绝对误差
        }
        
        maxError = fmax(maxError, rel_error);
        avgError += rel_error;
        validCount++;
    }
    
    if (validCount > 0) {
        avgError /= validCount;
        
        // 第二遍：计算标准差
        float variance = 0.0f;
        for (int i = 0; i < dimA.rows * dimB.cols; i++) {
            float ref_val = ref_C[i];
            float gpu_val = h_C[i];
            
            // 检查是否为有效值
            if (isnan(ref_val) || isnan(gpu_val) || 
                isinf(ref_val) || isinf(gpu_val)) {
                continue;
            }
            
            float abs_error = fabs(gpu_val - ref_val);
            float rel_error;
            
            if (fabs(ref_val) > 1e-6f) {
                rel_error = abs_error / fabs(ref_val);
            } else if (abs_error < 1e-6f) {
                rel_error = 0.0f;
            } else {
                rel_error = abs_error;
            }
            
            variance += (rel_error - avgError) * (rel_error - avgError);
        }
        
        stdError = sqrt(variance / validCount);
    } else {
        // 如果没有有效样本，设置为NaN
        avgError = nanf("");
        stdError = nanf("");
    }
    
    printf("%s 结果验证:\n", precision);
    printf("  最大相对误差: %e\n", maxError);
    printf("  平均相对误差: %e\n", avgError);
    printf("  相对误差标准差: %e\n", stdError);
    printf("  有效样本数: %d / %d\n", validCount, dimA.rows * dimB.cols);
    
    // 如果出现异常值，打印详细信息
    if (maxError > 1.0f || isnan(avgError) || isnan(stdError)) {
        printf("警告：检测到异常值，打印详细信息：\n");
        for (int i = 0; i < min(10, dimA.rows * dimB.cols); i++) {
            float ref_val = ref_C[i];
            float gpu_val = h_C[i];
            printf("  索引 %d: 参考值 = %e, GPU值 = %e\n", i, ref_val, gpu_val);
        }
    }
    
    delete[] h_C;
    delete[] ref_C;
    delete[] h_A;
    delete[] h_B;
}

// 性能测试函数
PerformanceResult benchmarkMatrixMul(float* A, float* B, float* C,
                                    const MatrixDim dimA, const MatrixDim dimB,
                                    void (*matrixMulFunc)(float*, float*, float*, MatrixDim, MatrixDim),
                                    const char* precision) {
    PerformanceResult result = {0};
    if (!A || !B || !C || !matrixMulFunc) return result;
    
    printf("开始测试 %s 矩阵乘法...\n", precision);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 预热
    printf("预热...\n");
    matrixMulFunc(C, A, B, dimA, dimB);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("预热失败: %s\n", cudaGetErrorString(err));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return result;
    }
    cudaDeviceSynchronize();
    
    // 性能测试
    const int numIterations = 10;
    float totalTime = 0.0f;
    
    printf("开始性能测试...\n");
    for (int i = 0; i < numIterations; i++) {
        cudaEventRecord(start);
        matrixMulFunc(C, A, B, dimA, dimB);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("迭代 %d 失败: %s\n", i, cudaGetErrorString(err));
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return result;
        }
        
        float time;
        cudaEventElapsedTime(&time, start, stop);
        totalTime += time;
    }
    
    result.avgTime = totalTime / numIterations;
    result.gflops = (2.0f * dimA.rows * dimA.cols * dimB.cols) / (result.avgTime * 1e6);
    
    // 计算内存带宽 (GB/s)
    size_t bytesProcessed = (dimA.rows * dimA.cols + dimB.rows * dimB.cols + 
                           dimA.rows * dimB.cols) * sizeof(float);
    result.memoryBandwidth = (bytesProcessed / 1e9) / (result.avgTime / 1000.0f);
    
    printf("计算误差统计...\n");
    // 计算误差统计
    float maxError, avgError, stdError;
    verifyResults(A, B, C, dimA, dimB, precision, maxError, avgError, stdError);
    
    // 保存误差统计结果
    result.maxError = maxError;
    result.avgError = avgError;
    result.stdError = stdError;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("%s 测试完成\n", precision);
    return result;
}

class MatrixOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 初始化CUDA设备
        printDeviceInfo();
        
        // 分配主机内存
        size_t maxSize = TEST_SIZES[NUM_SIZES-1] * TEST_SIZES[NUM_SIZES-1];
        h_A = new float[maxSize];
        h_B = new float[maxSize];
        h_C = new float[maxSize];
        
        if (!h_A || !h_B || !h_C) {
            printf("主机内存分配失败\n");
            TearDown();  // 清理已分配的内存
            return;
        }
        
        // 初始化随机矩阵
        srand(time(NULL));
        initRandomMatrix(h_A, maxSize);
        initRandomMatrix(h_B, maxSize);
        
        // 分配设备内存
        cudaError_t err;
        err = cudaMalloc(&d_A, maxSize * sizeof(float));
        if (err != cudaSuccess) {
            printf("设备内存分配失败 d_A: %s\n", cudaGetErrorString(err));
            TearDown();  // 清理已分配的内存
            return;
        }
        
        err = cudaMalloc(&d_B, maxSize * sizeof(float));
        if (err != cudaSuccess) {
            printf("设备内存分配失败 d_B: %s\n", cudaGetErrorString(err));
            TearDown();  // 清理已分配的内存
            return;
        }
        
        err = cudaMalloc(&d_C, maxSize * sizeof(float));
        if (err != cudaSuccess) {
            printf("设备内存分配失败 d_C: %s\n", cudaGetErrorString(err));
            TearDown();  // 清理已分配的内存
            return;
        }
        
        // 复制数据到设备
        err = cudaMemcpy(d_A, h_A, maxSize * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("数据复制失败 d_A: %s\n", cudaGetErrorString(err));
            TearDown();  // 清理已分配的内存
            return;
        }
        
        err = cudaMemcpy(d_B, h_B, maxSize * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("数据复制失败 d_B: %s\n", cudaGetErrorString(err));
            TearDown();  // 清理已分配的内存
            return;
        }
    }
    
    void TearDown() override {
        if (h_A) {
            delete[] h_A;
            h_A = nullptr;
        }
        if (h_B) {
            delete[] h_B;
            h_B = nullptr;
        }
        if (h_C) {
            delete[] h_C;
            h_C = nullptr;
        }
        
        if (d_A) {
            cudaFree(d_A);
            d_A = nullptr;
        }
        if (d_B) {
            cudaFree(d_B);
            d_B = nullptr;
        }
        if (d_C) {
            cudaFree(d_C);
            d_C = nullptr;
        }
    }
    
    float *h_A = nullptr, *h_B = nullptr, *h_C = nullptr;  // 主机内存
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;  // 设备内存
};

// 测试不同大小的矩阵
TEST_F(MatrixOpsTest, DifferentSizes) {
    // 只测试最小的矩阵大小，确保基本功能正常
    int size = TEST_SIZES[0];  // 64
    MatrixDim dimA = {size, size};
    MatrixDim dimB = {size, size};
    
    printf("\n测试矩阵大小: %d x %d\n", size, size);
    
    // 分配当前大小的内存
    float *d_current_A = nullptr, *d_current_B = nullptr, *d_current_C = nullptr;
    cudaError_t err;
    
    err = cudaMalloc(&d_current_A, size * size * sizeof(float));
    if (err != cudaSuccess) {
        printf("内存分配失败: %s\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaMalloc(&d_current_B, size * size * sizeof(float));
    if (err != cudaSuccess) {
        printf("内存分配失败: %s\n", cudaGetErrorString(err));
        if (d_current_A) cudaFree(d_current_A);
        return;
    }
    
    err = cudaMalloc(&d_current_C, size * size * sizeof(float));
    if (err != cudaSuccess) {
        printf("内存分配失败: %s\n", cudaGetErrorString(err));
        if (d_current_A) cudaFree(d_current_A);
        if (d_current_B) cudaFree(d_current_B);
        return;
    }
    
    // 复制数据 - 使用主机内存作为源
    err = cudaMemcpy(d_current_A, h_A, size * size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("数据复制失败: %s\n", cudaGetErrorString(err));
        if (d_current_A) cudaFree(d_current_A);
        if (d_current_B) cudaFree(d_current_B);
        if (d_current_C) cudaFree(d_current_C);
        return;
    }
    
    err = cudaMemcpy(d_current_B, h_B, size * size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("数据复制失败: %s\n", cudaGetErrorString(err));
        if (d_current_A) cudaFree(d_current_A);
        if (d_current_B) cudaFree(d_current_B);
        if (d_current_C) cudaFree(d_current_C);
        return;
    }
    
    // 只测试FP32，确保基本功能正常
    printf("开始FP32测试...\n");
    PerformanceResult fp32_result = benchmarkMatrixMul(d_current_A, d_current_B, d_current_C, 
                                                     dimA, dimB, matrixMulFP32, "FP32");
    
    printf("FP32测试完成，结果:\n");
    printf("  时间: %.2f ms\n", fp32_result.avgTime);
    printf("  GFLOPS: %.2f\n", fp32_result.gflops);
    printf("  最大误差: %.2e\n", fp32_result.maxError);
    
    // 释放当前大小的内存
    if (d_current_A) cudaFree(d_current_A);
    if (d_current_B) cudaFree(d_current_B);
    if (d_current_C) cudaFree(d_current_C);
}

// 测试边界条件
TEST_F(MatrixOpsTest, EdgeCases) {
    // 测试小矩阵
    MatrixDim dimA = {16, 16};
    MatrixDim dimB = {16, 16};
    
    float *d_small_A, *d_small_B, *d_small_C;
    cudaError_t err;
    
    err = cudaMalloc(&d_small_A, 16 * 16 * sizeof(float));
    if (err != cudaSuccess) {
        printf("内存分配失败: %s\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaMalloc(&d_small_B, 16 * 16 * sizeof(float));
    if (err != cudaSuccess) {
        printf("内存分配失败: %s\n", cudaGetErrorString(err));
        cudaFree(d_small_A);
        return;
    }
    
    err = cudaMalloc(&d_small_C, 16 * 16 * sizeof(float));
    if (err != cudaSuccess) {
        printf("内存分配失败: %s\n", cudaGetErrorString(err));
        cudaFree(d_small_A);
        cudaFree(d_small_B);
        return;
    }
    
    // 复制数据 - 使用主机内存作为源
    err = cudaMemcpy(d_small_A, h_A, 16 * 16 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("数据复制失败: %s\n", cudaGetErrorString(err));
        cudaFree(d_small_A);
        cudaFree(d_small_B);
        cudaFree(d_small_C);
        return;
    }
    
    err = cudaMemcpy(d_small_B, h_B, 16 * 16 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("数据复制失败: %s\n", cudaGetErrorString(err));
        cudaFree(d_small_A);
        cudaFree(d_small_B);
        cudaFree(d_small_C);
        return;
    }
    
    printf("测试小矩阵 (16x16)...\n");
    PerformanceResult result = benchmarkMatrixMul(d_small_A, d_small_B, d_small_C, dimA, dimB,
                                                matrixMulFP32, "FP32");
    EXPECT_GT(result.gflops, 0.0f);
    
    // 释放内存
    cudaFree(d_small_A);
    cudaFree(d_small_B);
    cudaFree(d_small_C);
}

// 添加新的测试用例
TEST_F(MatrixOpsTest, PrecisionComparison) {
    // 测试所有矩阵大小
    for (int i = 0; i < NUM_SIZES; i++) {
        int size = TEST_SIZES[i];
        MatrixDim dimA = {size, size};
        MatrixDim dimB = {size, size};
        
        printf("\n测试矩阵大小: %d x %d\n", size, size);
        
        // 分配当前大小的内存
        float *d_current_A = nullptr, *d_current_B = nullptr, *d_current_C = nullptr;
        cudaMalloc(&d_current_A, size * size * sizeof(float));
        cudaMalloc(&d_current_B, size * size * sizeof(float));
        cudaMalloc(&d_current_C, size * size * sizeof(float));
        
        // 复制数据
        cudaMemcpy(d_current_A, h_A, size * size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_current_B, h_B, size * size * sizeof(float), cudaMemcpyHostToDevice);
        
        // 测试FP32
        printf("\n=== FP32测试 ===\n");
        PerformanceResult fp32_result = benchmarkMatrixMul(d_current_A, d_current_B, d_current_C,
                                                         dimA, dimB, matrixMulFP32, "FP32");
        
        // 测试FP16
        printf("\n=== FP16测试 ===\n");
        PerformanceResult fp16_result = benchmarkMatrixMul(d_current_A, d_current_B, d_current_C,
                                                         dimA, dimB, matrixMulFP16, "FP16");
        
        // 测试FP8
        printf("\n=== FP8测试 ===\n");
        PerformanceResult fp8_result = benchmarkMatrixMul(d_current_A, d_current_B, d_current_C,
                                                        dimA, dimB, matrixMulFP8, "FP8");
        
        // 测试INT8
        printf("\n=== INT8测试 ===\n");
        PerformanceResult int8_result = benchmarkMatrixMul(d_current_A, d_current_B, d_current_C,
                                                         dimA, dimB, matrixMulINT8, "INT8");
        
        // 打印性能比较
        printf("\n性能比较 (矩阵大小: %dx%d):\n", size, size);
        printf("格式    时间(ms)  GFLOPS   最大相对误差  平均相对误差  内存带宽(GB/s)\n");
        printf("FP32    %8.3f  %7.2f  %e  %e  %7.2f\n",
               fp32_result.avgTime, fp32_result.gflops, fp32_result.maxError,
               fp32_result.avgError, fp32_result.memoryBandwidth);
        printf("FP16    %8.3f  %7.2f  %e  %e  %7.2f\n",
               fp16_result.avgTime, fp16_result.gflops, fp16_result.maxError,
               fp16_result.avgError, fp16_result.memoryBandwidth);
        printf("FP8     %8.3f  %7.2f  %e  %e  %7.2f\n",
               fp8_result.avgTime, fp8_result.gflops, fp8_result.maxError,
               fp8_result.avgError, fp8_result.memoryBandwidth);
        printf("INT8    %8.3f  %7.2f  %e  %e  %7.2f\n",
               int8_result.avgTime, int8_result.gflops, int8_result.maxError,
               int8_result.avgError, int8_result.memoryBandwidth);
        
        // 清理内存
        cudaFree(d_current_A);
        cudaFree(d_current_B);
        cudaFree(d_current_C);
    }
}

// 测试融合操作
TEST_F(MatrixOpsTest, FusedOperations) {
    // 测试所有矩阵大小
    for (int i = 0; i < NUM_SIZES; i++) {
        int size = TEST_SIZES[i];
        MatrixDim dimA = {size, size};
        MatrixDim dimB = {size, size};
        
        printf("\n测试融合操作 - 矩阵大小: %d x %d\n", size, size);
        
        // 分配当前大小的内存
        float *d_current_A = nullptr, *d_current_B = nullptr, *d_current_C = nullptr;
        cudaMalloc(&d_current_A, size * size * sizeof(float));
        cudaMalloc(&d_current_B, size * size * sizeof(float));
        cudaMalloc(&d_current_C, size * size * sizeof(float));
        
        // 复制数据
        cudaMemcpy(d_current_A, h_A, size * size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_current_B, h_B, size * size * sizeof(float), cudaMemcpyHostToDevice);
        
        // 测试融合FP8
        printf("\n=== 融合FP8测试 ===\n");
        PerformanceResult fused_fp8_result = benchmarkMatrixMul(d_current_A, d_current_B, d_current_C,
                                                              dimA, dimB, fusedFP8MatrixMul, "融合FP8");
        
        // 测试融合INT8
        printf("\n=== 融合INT8测试 ===\n");
        PerformanceResult fused_int8_result = benchmarkMatrixMul(d_current_A, d_current_B, d_current_C,
                                                               dimA, dimB, fusedINT8MatrixMul, "融合INT8");
        
        // 打印性能比较
        printf("\n融合操作性能比较 (矩阵大小: %dx%d):\n", size, size);
        printf("操作      时间(ms)  GFLOPS   最大相对误差  平均相对误差  内存带宽(GB/s)\n");
        printf("融合FP8   %8.3f  %7.2f  %e  %e  %7.2f\n",
               fused_fp8_result.avgTime, fused_fp8_result.gflops,
               fused_fp8_result.maxError, fused_fp8_result.avgError,
               fused_fp8_result.memoryBandwidth);
        printf("融合INT8  %8.3f  %7.2f  %e  %e  %7.2f\n",
               fused_int8_result.avgTime, fused_int8_result.gflops,
               fused_int8_result.maxError, fused_int8_result.avgError,
               fused_int8_result.memoryBandwidth);
        
        // 清理内存
        cudaFree(d_current_A);
        cudaFree(d_current_B);
        cudaFree(d_current_C);
    }
}

// 测试特殊矩阵
TEST_F(MatrixOpsTest, SpecialMatrices) {
    const int size = 128;  // 使用中等大小的矩阵
    MatrixDim dimA = {size, size};
    MatrixDim dimB = {size, size};
    
    // 分配内存
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, size * size * sizeof(float));
    cudaMalloc(&d_B, size * size * sizeof(float));
    cudaMalloc(&d_C, size * size * sizeof(float));
    
    // 创建单位矩阵
    float* identity = new float[size * size];
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            identity[i * size + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    
    printf("\n=== 特殊矩阵测试 ===\n");
    
    // 测试单位矩阵乘法
    printf("\n单位矩阵测试:\n");
    cudaMemcpy(d_A, h_A, size * size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, identity, size * size * sizeof(float), cudaMemcpyHostToDevice);
    
    printf("A * I:\n");
    PerformanceResult identity_result = benchmarkMatrixMul(d_A, d_B, d_C,
                                                         dimA, dimB, matrixMulFP32, "单位矩阵");
    
    // 测试全零矩阵
    printf("\n全零矩阵测试:\n");
    cudaMemset(d_B, 0, size * size * sizeof(float));
    
    printf("A * 0:\n");
    PerformanceResult zero_result = benchmarkMatrixMul(d_A, d_B, d_C,
                                                     dimA, dimB, matrixMulFP32, "全零矩阵");
    
    // 测试非方阵
    printf("\n非方阵测试:\n");
    MatrixDim dimNonSquareA = {size, size/2};
    MatrixDim dimNonSquareB = {size/2, size};
    
    float* nonsquare = new float[size * size/2];
    for (int i = 0; i < size * size/2; i++) {
        nonsquare[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    
    cudaMemcpy(d_A, nonsquare, size * size/2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, nonsquare, size/2 * size * sizeof(float), cudaMemcpyHostToDevice);
    
    printf("非方阵 (%dx%d) * (%dx%d):\n", size, size/2, size/2, size);
    PerformanceResult nonsquare_result = benchmarkMatrixMul(d_A, d_B, d_C,
                                                          dimNonSquareA, dimNonSquareB,
                                                          matrixMulFP32, "非方阵");
    
    // 清理内存
    delete[] identity;
    delete[] nonsquare;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// 测试量化效果
TEST_F(MatrixOpsTest, QuantizationAnalysis) {
    const int size = 1024;  // 使用较大的矩阵以获得更有意义的统计结果
    MatrixDim dimA = {size, size};
    MatrixDim dimB = {size, size};
    
    // 分配内存
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, size * size * sizeof(float));
    cudaMalloc(&d_B, size * size * sizeof(float));
    cudaMalloc(&d_C, size * size * sizeof(float));
    
    // 生成特殊的测试数据
    float* special_data = new float[size * size];
    for (int i = 0; i < size * size; i++) {
        // 生成一些极端值和常见值的混合
        float r = (float)rand() / RAND_MAX;
        if (r < 0.1f) {
            special_data[i] = 1e-6f;  // 非常小的值
        } else if (r < 0.2f) {
            special_data[i] = 1e6f;   // 非常大的值
        } else {
            special_data[i] = r * 2.0f - 1.0f;  // 正常范围的值
        }
    }
    
    printf("\n=== 量化效果分析 ===\n");
    
    // 复制特殊数据到设备
    cudaMemcpy(d_A, special_data, size * size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, special_data, size * size * sizeof(float), cudaMemcpyHostToDevice);
    
    // 测试不同精度格式
    printf("\nFP32 (基准):\n");
    PerformanceResult fp32_result = benchmarkMatrixMul(d_A, d_B, d_C,
                                                     dimA, dimB, matrixMulFP32, "FP32");
    
    printf("\nFP16:\n");
    PerformanceResult fp16_result = benchmarkMatrixMul(d_A, d_B, d_C,
                                                     dimA, dimB, matrixMulFP16, "FP16");
    
    printf("\nFP8:\n");
    PerformanceResult fp8_result = benchmarkMatrixMul(d_A, d_B, d_C,
                                                    dimA, dimB, matrixMulFP8, "FP8");
    
    printf("\nINT8:\n");
    PerformanceResult int8_result = benchmarkMatrixMul(d_A, d_B, d_C,
                                                     dimA, dimB, matrixMulINT8, "INT8");
    
    // 打印量化分析结果
    printf("\n量化效果分析结果:\n");
    printf("格式    相对误差范围          平均相对误差        内存节省\n");
    printf("FP32    [0, %e]  %e     0%%\n",
           fp32_result.maxError, fp32_result.avgError);
    printf("FP16    [0, %e]  %e    50%%\n",
           fp16_result.maxError, fp16_result.avgError);
    printf("FP8     [0, %e]  %e    75%%\n",
           fp8_result.maxError, fp8_result.avgError);
    printf("INT8    [0, %e]  %e    75%%\n",
           int8_result.maxError, int8_result.avgError);
    
    // 清理内存
    delete[] special_data;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// 主函数
int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 