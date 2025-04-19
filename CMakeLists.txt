cmake_minimum_required(VERSION 3.8)

# 启用CUDA
project(CUDA_Project CUDA CXX)

# 设置CUDA架构策略
cmake_policy(SET CMP0104 NEW)

# 设置多个CUDA架构
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")  # 支持RTX 20/30/40系列

# 设置CUDA编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DFP8_SUPPORT")

# 添加CUDA编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_75")

# 查找CUDA包
find_package(CUDA REQUIRED)

# 包含目录
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

# 添加源文件
file(GLOB_RECURSE SOURCES 
    "src/host/*.cpp"
    "src/host/*.cu"
    "src/kernel/*.cu"
)

# 创建库
add_library(${PROJECT_NAME} STATIC ${SOURCES})

# 链接CUDA库
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${CUDA_LIBRARIES}
    cublas
)

# 创建主程序可执行文件
add_executable(${PROJECT_NAME}_main src/host/main.cpp)
target_link_libraries(${PROJECT_NAME}_main PRIVATE ${PROJECT_NAME})

# 添加测试
enable_testing()
add_subdirectory(tests) 