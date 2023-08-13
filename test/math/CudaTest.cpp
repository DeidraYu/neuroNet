#include "gtest/gtest.h"
#include <cuda_runtime.h>

#include "kernel.h"

TEST(CudaTest, linking)
{
    doit();
}