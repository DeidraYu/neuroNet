# Now simply link against gtest or gtest_main as needed. Eg
add_executable(neuronet_test MathUtilsTest.cpp )
target_link_libraries(neuronet_test gtest_main)
# add_test(NAME example_test COMMAND example)
