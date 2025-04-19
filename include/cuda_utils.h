#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <stdio.h>

// 检查CUDA错误
inline void checkCudaError(cudaError_t error, const char* message = "CUDA错误") {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(error));
        exit(-1);
    }
}

// 打印设备信息
inline void printDeviceInfo() {
    int deviceCount;
    checkCudaError(cudaGetDeviceCount(&deviceCount), "获取设备数量失败");
    printf("找到 %d 个CUDA设备\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        checkCudaError(cudaGetDeviceProperties(&prop, i), "获取设备属性失败");
        
        printf("设备 %d: %s\n", i, prop.name);
        printf("  计算能力: %d.%d\n", prop.major, prop.minor);
        printf("  总内存: %zu MB\n", prop.totalGlobalMem / (1024*1024));
        printf("  多处理器数量: %d\n", prop.multiProcessorCount);
        printf("  最大线程数/块: %d\n", prop.maxThreadsPerBlock);
        printf("  最大共享内存/块: %zu KB\n", prop.sharedMemPerBlock / 1024);
    }
}

#endif // CUDA_UTILS_H 