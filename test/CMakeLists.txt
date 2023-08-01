add_executable(layer_test LayerTest.cpp ../src/Utils.cpp ../src/math/utils.cpp)
target_link_libraries(layer_test gtest_main)

add_executable(file_io_test  FileIoTest.cpp ../src/Utils.cpp ../src/math/utils.cpp ../src/FileIO.cpp)
target_link_libraries(file_io_test gtest_main)


add_executable(network_test NetworkTest.cpp ../src/Utils.cpp ../src/math/utils.cpp ../src/MnistData.cpp ../src/NetworkTrainer.cpp)
target_link_libraries(network_test gtest_main)

# Pass MNIST data directory to network_test
target_compile_definitions(network_test PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")

add_subdirectory(math)
