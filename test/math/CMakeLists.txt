# Now simply link against gtest or gtest_main as needed. Eg
add_executable(math_utils_test utilsTest.cpp ../../src/math/utils.cpp)
target_link_libraries(math_utils_test gtest_main)

add_executable(vector_test VectorTest.cpp ../../src/math/utils.cpp  ../../src/FileIO.cpp)
target_link_libraries(vector_test gtest_main)

add_executable(matrix_test MatrixTest.cpp ../../src/math/utils.cpp)
target_link_libraries(matrix_test gtest_main)
