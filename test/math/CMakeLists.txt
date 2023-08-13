# Now simply link against gtest or gtest_main as needed. Eg
add_executable(math_utils_test utilsTest.cpp ../../src/math/utils.cpp)
target_link_libraries(math_utils_test gtest_main)

add_executable(vector_test VectorTest.cpp ../../src/math/utils.cpp  ../../src/FileIO.cpp)
target_link_libraries(vector_test gtest_main)

add_executable(matrix_test MatrixTest.cpp ../../src/math/utils.cpp)
target_link_libraries(matrix_test gtest_main)

cuda_add_executable(cuda_test CudaTest.cpp kernel.cu)
set_property(TARGET cuda_test PROPERTY CUDA_STANDARD 20)
target_include_directories(cuda_test PRIVATE ${CUDA_INCLUDE_DIRS})

target_link_libraries(cuda_test ${CUDA_LIBRARIES} gtest_main)
