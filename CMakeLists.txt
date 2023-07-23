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

##
# GTEST
# see https://github.com/google/googletest/blob/main/googletest/README.md
##
include(FetchContent)
FetchContent_Declare(
  googletest
  # Specify the commit you depend on and update it regularly.
  # URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
  # URL https://github.com/google/googletest/archive/ff74f02ab1c106696f288540e9c623d56b76e3f7.zip
  URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
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
