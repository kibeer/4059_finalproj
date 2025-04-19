# 量化矩阵乘法：基于CUDA的FP8/INT8实现与性能分析
# Quantized Matrix Multiplication: CUDA-Based FP8/INT8 Implementation and Performance Analysis

## 项目简介 / Project Introduction
本项目实现了基于CUDA的低精度矩阵乘法，包括FP32、FP16、FP8和INT8格式，并进行了全面的性能分析。项目特别关注了量化对性能、内存使用和数值精度的影响。

## 功能特点 / Features
- 支持多种精度格式的矩阵乘法
- 实现了融合的量化-反量化-矩阵乘法操作
- 提供详细的性能分析和精度评估
- 包含完整的测试套件

## 环境要求 / Requirements
- CUDA 11.0或更高版本
- CMake 3.8或更高版本
- C++14或更高版本
- NVIDIA GPU（支持FP16/FP8/INT8）

## 安装说明 / Installation
```bash
mkdir build
cd build
cmake ..
make
```

## 使用方法 / Usage
1. 运行主程序：
```bash
./CUDA_Project_main
```

2. 运行测试：
```bash
./matrix_ops_test
```

## 性能分析 / Performance Analysis
详细的性能分析结果请参考 `analysis_results.md`。

主要发现：
- FP16在性能和内存使用上取得较好平衡
- INT8在特定场景下具有优势
- 融合操作在大矩阵上效果更好

## 项目结构 / Project Structure
```
.
├── include/           # 头文件
├── src/              # 源代码
│   ├── kernel/       # CUDA核函数
│   └── host/         # 主机端代码
├── tests/            # 测试代码
├── CMakeLists.txt    # CMake配置
└── README.md         # 项目文档
```

## 开发指南 / Development Guide
1. 代码风格遵循CUDA最佳实践
2. 所有新功能需要添加相应的测试
3. 性能优化需要考虑不同精度格式的特点

## 测试说明 / Testing
- 包含单元测试和性能测试
- 支持不同矩阵大小的测试
- 包含特殊矩阵测试（单位矩阵、全零矩阵等）

## 贡献指南 / Contributing
欢迎提交问题和改进建议。请确保：
1. 代码符合项目规范
2. 添加适当的测试
3. 更新相关文档

## 许可证 / License
MIT License

## 联系方式 / Contact
如有问题，请提交Issue或联系项目维护者。 