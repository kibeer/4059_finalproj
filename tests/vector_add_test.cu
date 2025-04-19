#include <gtest/gtest.h>
#include <stdlib.h>
#include <time.h>
#include "../include/cuda_utils.h"

// 声明CUDA函数
extern "C" void launchVectorAdd(const float* h_a, const float* h_b, float* h_c, int n);

// 测试向量加法
TEST(VectorAddTest, SmallVector) {
    const int n = 1000;
    float* h_a = new float[n];
    float* h_b = new float[n];
    float* h_c = new float[n];
    
    // 初始化输入数据
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        h_a[i] = (float)rand() / RAND_MAX;
        h_b[i] = (float)rand() / RAND_MAX;
    }
    
    // 调用CUDA函数
    launchVectorAdd(h_a, h_b, h_c, n);
    
    // 验证结果
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(h_c[i], h_a[i] + h_b[i], 1e-5);
    }
    
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
}

// 测试向量加法 - 大向量
TEST(VectorAddTest, LargeVector) {
    const int n = 1000000;
    float* h_a = new float[n];
    float* h_b = new float[n];
    float* h_c = new float[n];
    
    // 初始化输入数据
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        h_a[i] = (float)rand() / RAND_MAX;
        h_b[i] = (float)rand() / RAND_MAX;
    }
    
    // 调用CUDA函数
    launchVectorAdd(h_a, h_b, h_c, n);
    
    // 验证结果
    for (int i = 0; i < n; i += 100000) {
        EXPECT_NEAR(h_c[i], h_a[i] + h_b[i], 1e-5);
    }
    
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
} 