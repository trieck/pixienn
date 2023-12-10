/********************************************************************************
* Copyright 2020-2023 Thomas A. Rieck, All Rights Reserved
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
********************************************************************************/

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "MaxPoolAlgo.h"

using namespace px;
using namespace testing;

struct MaxPoolTestParams
{
    int batch = 0;
    int channels = 0;
    int height = 0;
    int width = 0;
    int outHeight = 0;
    int outWidth = 0;
    int kernel = 0;
    int stride = 0;
    int padding = 0;
};

class MaxPoolCpuTest : public Test
{
protected:
    void SetUp(const MaxPoolTestParams& params)
    {
        output_ = PxCpuVector(params.batch * params.channels * params.outHeight * params.outWidth);
        indexes_ = PxCpuVectorT<int>(params.batch * params.channels * params.outHeight * params.outWidth);
    }

    void MaxPoolTest(const PxCpuVector& input, const PxCpuVector& expected, const MaxPoolTestParams& params)
    {
        SetUp(params);

        MaxPoolContext ctxt;
        ctxt.input = &input;
        ctxt.output = &output_;
        ctxt.indexes = &indexes_;
        ctxt.batch = params.batch;
        ctxt.channels = params.channels;
        ctxt.height = params.height;
        ctxt.width = params.width;
        ctxt.kernel = params.kernel;
        ctxt.stride = params.stride;
        ctxt.padding = params.padding;
        ctxt.outHeight = params.outHeight;
        ctxt.outWidth = params.outWidth;

        maxPoolForward(ctxt);

        EXPECT_THAT(output_, Pointwise(FloatNear(1e-4), expected));
    }

private:
    PxCpuVector output_;
    PxCpuVectorT<int> indexes_;
};

TEST_F(MaxPoolCpuTest, SimpleMaxPoolCpu)
{
    MaxPoolTestParams params;
    params.batch = 1;
    params.channels = 1;
    params.height = 4;
    params.kernel = 2;
    params.outHeight = 2;
    params.outWidth = 2;
    params.padding = 0;
    params.stride = 2;
    params.width = 4;

    PxCpuVector input = {
            1.0f, 2.0f, 5.0f, 6.0f,
            3.0f, 4.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 13.0f, 14.0f,
            11.0f, 12.0f, 15.0f, 16.0f
    };

    PxCpuVector expected{ 4.0f, 8.0f, 12.0f, 16.0f };

    MaxPoolTest(input, expected, params);
}

#ifdef USE_CUDA

class MaxPoolCudaTest : public Test
{
protected:
    void SetUp(const MaxPoolTestParams& params)
    {
        output_ = PxCudaVector(params.batch * params.channels * params.outHeight * params.outWidth);
        indexes_ = PxCudaVectorT<int>(params.batch * params.channels * params.outHeight * params.outWidth);
    }

    void MaxPoolTest(const PxCudaVector& input, const PxCudaVector& expected, const MaxPoolTestParams& params)
    {
        SetUp(params);

        MaxPoolContext ctxt;
        ctxt.inputGpu = &input;
        ctxt.outputGpu = &output_;
        ctxt.indexesGpu = &indexes_;
        ctxt.batch = params.batch;
        ctxt.channels = params.channels;
        ctxt.height = params.height;
        ctxt.width = params.width;
        ctxt.kernel = params.kernel;
        ctxt.stride = params.stride;
        ctxt.padding = params.padding;
        ctxt.outHeight = params.outHeight;
        ctxt.outWidth = params.outWidth;

        maxPoolForwardGpu(ctxt);

        EXPECT_THAT(output_.asVector(), Pointwise(FloatNear(1e-4), expected.asVector()));
    }

private:
    PxCudaVector output_;
    PxCudaVectorT<int> indexes_;
};


TEST_F(MaxPoolCudaTest, SimpleMaxPoolCuda)
{
    MaxPoolTestParams params;
    params.batch = 1;
    params.channels = 1;
    params.height = 4;
    params.kernel = 2;
    params.outHeight = 2;
    params.outWidth = 2;
    params.padding = 0;
    params.stride = 2;
    params.width = 4;

    PxCudaVector input{
            1.0f, 2.0f, 5.0f, 6.0f,
            3.0f, 4.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 13.0f, 14.0f,
            11.0f, 12.0f, 15.0f, 16.0f
    };

    PxCudaVector expected{ 4.0f, 8.0f, 12.0f, 16.0f };

    MaxPoolTest(input, expected, params);
}

#endif // USE_CUDA
