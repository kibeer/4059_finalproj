# 量化矩阵乘法：基于CUDA的FP8/INT8实现与性能分析
# Quantized Matrix Multiplication: CUDA-Based FP8/INT8 Implementation and Performance Analysis

## 项目简介 / Project Introduction
本项目实现了基于CUDA的低精度矩阵乘法，包括FP32、FP16、FP8和INT8格式，并进行了全面的性能分析。项目特别关注了量化对性能、内存使用和数值精度的影响。

This project implements CUDA-based low-precision matrix multiplication in FP32, FP16, FP8, and INT8 formats, with comprehensive performance analysis. The project specifically focuses on the impact of quantization on performance, memory usage, and numerical precision.

## 功能特点 / Features
- 支持多种精度格式的矩阵乘法 / Support for matrix multiplication in multiple precision formats
- 实现了融合的量化-反量化-矩阵乘法操作 / Implementation of fused quantize-dequantize-matrix multiplication operations
- 提供详细的性能分析和精度评估 / Detailed performance analysis and accuracy evaluation
- 包含完整的测试套件 / Comprehensive test suite

## 环境要求 / Requirements
- CUDA 11.0或更高版本 / CUDA 11.0 or higher
- CMake 3.8或更高版本 / CMake 3.8 or higher
- C++14或更高版本 / C++14 or higher
- NVIDIA GPU（支持FP16/FP8/INT8）/ NVIDIA GPU (supporting FP16/FP8/INT8)

## 安装说明 / Installation
```bash
# 克隆仓库 / Clone the repository
git clone https://github.com/YOUR_USERNAME/quantized-matrix-multiplication.git
cd quantized-matrix-multiplication

# 创建构建目录 / Create build directory
mkdir build
cd build

# 配置和编译 / Configure and build
cmake ..
make -j$(nproc)
```

## 使用方法 / Usage
1. 运行主程序 / Run the main program:
```bash
./CUDA_Project_main
```

2. 运行测试 / Run tests:
```bash
# 运行所有测试 / Run all tests
ctest

# 运行特定测试 / Run specific tests
./cuda_tests
```

## 性能分析 / Performance Analysis
详细的性能分析结果请参考 `analysis_results.md` 和 `analysis_results_en.md`。

Detailed performance analysis results can be found in `analysis_results.md` and `analysis_results_en.md`.

主要发现 / Key Findings:
- FP16在性能和内存使用上取得较好平衡 / FP16 achieves a good balance between performance and memory usage
- INT8在特定场景下具有优势 / INT8 shows advantages in specific scenarios
- 融合操作在大矩阵上效果更好 / Fused operations perform better on larger matrices

## 项目结构 / Project Structure
```
.
├── include/           # 头文件 / Header files
├── src/              # 源代码 / Source code
│   ├── kernel/       # CUDA核函数 / CUDA kernels
│   └── host/         # 主机端代码 / Host code
├── tests/            # 测试代码 / Test code
├── docs/             # 文档 / Documentation
├── CMakeLists.txt    # CMake配置 / CMake configuration
└── README.md         # 项目文档 / Project documentation
```

## 开发指南 / Development Guide
1. 代码风格遵循CUDA最佳实践 / Code style follows CUDA best practices
2. 所有新功能需要添加相应的测试 / All new features require corresponding tests
3. 性能优化需要考虑不同精度格式的特点 / Performance optimization should consider characteristics of different precision formats

## 测试说明 / Testing
- 包含单元测试和性能测试 / Includes unit tests and performance tests
- 支持不同矩阵大小的测试 / Supports tests with different matrix sizes
- 包含特殊矩阵测试（单位矩阵、全零矩阵等）/ Includes special matrix tests (identity matrix, zero matrix, etc.)

## 贡献指南 / Contributing
欢迎提交问题和改进建议。请确保：
Welcome to submit issues and improvement suggestions. Please ensure:
1. 代码符合项目规范 / Code follows project standards
2. 添加适当的测试 / Add appropriate tests
3. 更新相关文档 / Update relevant documentation

## 许可证 / License
MIT License

## 联系方式 / Contact
如有问题，请提交Issue或联系项目维护者。
For any issues, please submit an Issue or contact the project maintainer. 