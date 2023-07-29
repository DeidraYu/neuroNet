cmake_minimum_required(VERSION 3.13)  # CMake version check
project(NeuroNet )                    # Create project "simple_example"

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo")

# Set default compile flags for GCC
if(CMAKE_COMPILER_IS_GNUCXX)
    message(STATUS "GCC detected, adding compile flags")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++98 -pedantic -Wall -Wextra")

    add_compile_options(
        -Wshadow
        -Wpointer-arith
        -Wcast-qual
        -Wcast-align
        -Wall
        -Wextra
        -pedantic
        # -Wmissing-declarations # I suppresed this because there is an Issue with gtest on Linux
        -Wunused
        -Wpedantic
        # -Wconversion
        -Wlogical-op
        # -Wdouble-promotion # For now we want double promotion. Turn on later.
        -Wformat=2
        -fmessage-length=0
        # -fno-exceptions
        # -march=broadwell
    )
    
    add_compile_options("$<$<CONFIG:RELEASE>:-O2;-Werror>")
else()
    # Set additional compiler flags for Release build (e.g., -O3)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
endif(CMAKE_COMPILER_IS_GNUCXX)


# .. -> hint, that the mnist package is one directory level above.
# When using just "find_package(MNIST REQUIRED)", "MNIST_DIR"
#    cmake variable has to be set correctly.
find_package(MNIST PATHS ../mnist)
if(NOT MNIST_FOUND)
    message(FATAL_ERROR "MNIST loader could not be found. It is available under https://github.com/wichtounet/mnist")
endif(NOT MNIST_FOUND)

include_directories(${MNIST_INCLUDE_DIR})

##
# GTEST
# see https://github.com/google/googletest/blob/main/googletest/README.md
##
include(FetchContent)
FetchContent_Declare(
  googletest
  # Specify the commit you depend on and update it regularly.
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Now simply link against gtest or gtest_main as needed. Eg
#add_executable(example example.cpp)
#target_link_libraries(example gtest_main)
#add_test(NAME example_test COMMAND example)

##
# Our own application
##
add_subdirectory(src)
add_subdirectory(test)
