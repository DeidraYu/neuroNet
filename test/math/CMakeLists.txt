# Now simply link against gtest or gtest_main as needed. Eg
add_executable(vector_test VectorTest.cpp)
target_link_libraries(vector_test gtest_main)
