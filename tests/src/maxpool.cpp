#include <gmock/gmock.h>
#include <gtest/gtest.h>

#ifdef USE_CUDA

#include "MaxPoolKernels.cuh"
#include "PxTensor.h"

using namespace px;
using namespace testing;

TEST(MaxPoolCudaTest, AsymmetricPaddingAndAccumulatingBackward)
{
    PxCudaVector input({ 1.0f, 2.0f, 3.0f,
                         4.0f, 5.0f, 6.0f,
                         7.0f, 8.0f, 9.0f });
    PxCudaVector output(9, 0.0f);
    PxCudaVectorT<int> indexes(9, -1);

    maxPoolForwardGpu(input.data(), output.data(), indexes.data(), 1, 1, 3, 3, 3, 3, 2, 1, 1);

    EXPECT_THAT(output.asVector(), ElementsAre(5.0f, 6.0f, 6.0f,
                                                8.0f, 9.0f, 9.0f,
                                                8.0f, 9.0f, 9.0f));

    PxCudaVector delta(9, 1.0f);
    PxCudaVector grad(9, 1.0f);
    maxPoolBackwardGpu(delta.data(), indexes.data(), grad.data(), 9);

    EXPECT_THAT(grad.asVector(), ElementsAre(1.0f, 1.0f, 1.0f,
                                              1.0f, 2.0f, 3.0f,
                                              1.0f, 3.0f, 5.0f));
}

#endif
