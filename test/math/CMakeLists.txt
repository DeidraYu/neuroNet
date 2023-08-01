# Now simply link against gtest or gtest_main as needed. Eg
add_executable(math_test VectorTest.cpp  utilsTest.cpp ../../src/math/utils.cpp ../../src/FileIO.cpp)
target_link_libraries(math_test gtest_main)
