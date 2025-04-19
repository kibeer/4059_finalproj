#include <cuda_runtime.h>
#include "../../include/cuda_utils.h"

// CUDA核心函数：向量加法
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 主机端函数：分配内存并调用CUDA核心
extern "C" void launchVectorAdd(const float* h_a, const float* h_b, float* h_c, int n) {
    float *d_a, *d_b, *d_c;
    
    // 分配设备内存
    checkCudaError(cudaMalloc((void**)&d_a, n * sizeof(float)), "分配设备内存 d_a");
    checkCudaError(cudaMalloc((void**)&d_b, n * sizeof(float)), "分配设备内存 d_b");
    checkCudaError(cudaMalloc((void**)&d_c, n * sizeof(float)), "分配设备内存 d_c");
    
    // 将数据从主机复制到设备
    checkCudaError(cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice), "复制 h_a 到 d_a");
    checkCudaError(cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice), "复制 h_b 到 d_b");
    
    // 启动CUDA核心
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    
    // 检查核心执行错误
    checkCudaError(cudaGetLastError(), "启动核心");
    checkCudaError(cudaDeviceSynchronize(), "同步设备");
    
    // 将结果从设备复制回主机
    checkCudaError(cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost), "复制 d_c 到 h_c");
    
    // 释放设备内存
    checkCudaError(cudaFree(d_a), "释放 d_a");
    checkCudaError(cudaFree(d_b), "释放 d_b");
    checkCudaError(cudaFree(d_c), "释放 d_c");
} 