# 查找GTest包
find_package(GTest REQUIRED)

# 添加测试源文件
set(TEST_SOURCES
    matrix_ops_test.cu
)

# 创建测试可执行文件
add_executable(cuda_tests ${TEST_SOURCES})

# 设置包含目录
target_include_directories(cuda_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# 链接GTest和项目库
target_link_libraries(cuda_tests
    PRIVATE
    GTest::GTest
    GTest::Main
    CUDA_Project
)

# 添加测试
add_test(NAME cuda_tests COMMAND cuda_tests) 