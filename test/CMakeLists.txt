# Now simply link against gtest or gtest_main as needed. Eg
add_executable(neuronet_test VectorTest.cpp MatrixTest.cpp)
target_link_libraries(neuronet_test gtest_main)
# add_test(NAME example_test COMMAND example)


add_executable(matrix_performance MatrixPerformance.cpp)
target_link_libraries(matrix_performance gtest_main)
