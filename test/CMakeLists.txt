# Now simply link against gtest or gtest_main as needed. Eg
add_executable(neuronet_test VectorTest.cpp MatrixTest.cpp ../src/Utils.cpp ../src/math/utils.cpp)
target_link_libraries(neuronet_test gtest_main)
# add_test(NAME example_test COMMAND example)


add_executable(matrix_performance MatrixPerformance.cpp ../src/Utils.cpp ../src/math/utils.cpp)
target_link_libraries(matrix_performance gtest_main)

add_executable(layer_test LayerTest.cpp ../src/Utils.cpp ../src/math/utils.cpp)
target_link_libraries(layer_test gtest_main)

add_executable(file_io_test  FileIoTest.cpp ../src/Utils.cpp ../src/math/utils.cpp ../src/FileIO.cpp)
target_link_libraries(file_io_test gtest_main)


add_subdirectory(math)
