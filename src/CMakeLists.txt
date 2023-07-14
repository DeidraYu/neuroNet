add_executable(neuronet main.cpp Net.cpp Trainer.cpp Evaluator.cpp Utils.cpp)

target_link_libraries(neuronet)
#target_compile_features(neuronet PRIVATE cxx_range_for)

# Pass MNIST data directory to main.cpp
target_compile_definitions(neuronet PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")