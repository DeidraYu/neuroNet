cmake_minimum_required(VERSION 3.13)  # CMake version check
project(NeuroNet )                    # Create project "simple_example"

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo")

# # Set default compile flags for GCC
# if(CMAKE_COMPILER_IS_GNUCXX)
#     message(STATUS "GCC detected, adding compile flags")
# #    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++98 -pedantic -Wall -Wextra")

#     add_compile_options(
#         -Wshadow
#         -Wpointer-arith
#         -Wcast-qual
#         -Wcast-align
#         -Wall
#         -Wextra
#         -pedantic
#         # -Wmissing-declarations # I suppresed this because there is an Issue with gtest on Linux
#         -Wunused
#         -Wpedantic
#         # -Wconversion
#         -Wlogical-op
#         # -Wdouble-promotion # For now we want double promotion. Turn on later.
#         -Wformat=2
#         -fmessage-length=0
#         # -fno-exceptions
#         # -march=broadwell
#     )
    
#     add_compile_options("$<$<CONFIG:RELEASE>:-O3;-Werror>")
# else()
#     # Set additional compiler flags for Release build (e.g., -O3)
#     set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
# endif(CMAKE_COMPILER_IS_GNUCXX)


# The matrix and vector code relies on the compiler vectorizing its inner loops. By default
# a compiler targets a baseline instruction set that is many years old, with only 128 bit
# vector registers, which leaves most of that speed unused. This lets it use what the
# machine building the code actually has.
#
# The resulting binaries are tuned for this machine and may refuse to start on an older
# one, so turn it off when building something to hand to someone else.
#
# Results stay reproducible either way. The order in which the inner loops add things up
# is fixed by the code rather than by the instruction set, so a wider vector register
# makes it faster without changing what it computes.
option(NEURONET_TUNE_FOR_THIS_MACHINE "Build for the instruction set of the build machine" ON)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # using Clang
  add_compile_options("$<$<CONFIG:RELEASE>:-O3;-Werror>")
  if (NEURONET_TUNE_FOR_THIS_MACHINE)
    add_compile_options("$<$<CONFIG:RELEASE>:-march=native>")
  endif()

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # using GCC
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
      
      add_compile_options("$<$<CONFIG:RELEASE>:-O3;-Werror>")
      if (NEURONET_TUNE_FOR_THIS_MACHINE)
        add_compile_options("$<$<CONFIG:RELEASE>:-march=native>")
      endif()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  # using Intel C++
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # using Visual Studio C++
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
  if (NEURONET_TUNE_FOR_THIS_MACHINE)
    # MSVC has no equivalent of -march=native. AVX2 is the widest setting that is
    # safe to assume on anything recent.
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /arch:AVX2")
  endif()
endif()

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

# enable testing functionality
enable_testing()

# define tests
add_test(
  NAME math_utils_test
  COMMAND $<TARGET_FILE:vector_test>
  )
  
add_test(
  NAME vector_test
  COMMAND $<TARGET_FILE:vector_test>
  )

add_test(
  NAME matrix_test
  COMMAND $<TARGET_FILE:matrix_test>
  )
  
add_test(
  NAME layer_test
  COMMAND $<TARGET_FILE:layer_test>
  )

add_test(
  NAME network_test
  COMMAND $<TARGET_FILE:network_test>
  )

