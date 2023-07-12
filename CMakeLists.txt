cmake_minimum_required(VERSION 3.13)  # CMake version check
project(NeuroNet )                    # Create project "simple_example"

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release")

# Set default compile flags for GCC
if(CMAKE_COMPILER_IS_GNUCXX)
    message(STATUS "GCC detected, adding compile flags")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++98 -pedantic -Wall -Wextra")

    add_compile_options(
        -Wshadow
        -Wpointer-arith
        -Wcast-qual
        -Wcast-align
        -Wconversion
        -Wall
        -Wextra
        -pedantic
        -Wmissing-declarations
        -Wunused
        -Wpedantic
        -Wconversion
        -Wlogical-op
        -Wdouble-promotion
        -Wformat=2
        -fmessage-length=0
        # -fno-exceptions
        # -march=broadwell
    )
    
    add_compile_options("$<$<CONFIG:RELEASE>:-O2;-Werror>")
endif(CMAKE_COMPILER_IS_GNUCXX)


# .. -> hint, that the mnist package is one directory level above.
# When using just "find_package(MNIST REQUIRED)", "MNIST_DIR"
#    cmake variable has to be set correctly.
find_package(MNIST PATHS ../mnist)
if(NOT MNIST_FOUND)
    message(FATAL_ERROR "MNIST loader could not be found. It is available under https://github.com/wichtounet/mnist")
endif(NOT MNIST_FOUND)

include_directories(${MNIST_INCLUDE_DIR})


add_subdirectory(src)
