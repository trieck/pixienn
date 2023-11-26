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

#include "UpsampleAlgo.h"

using namespace px;
using namespace testing;

struct UpsampleTestParams
{
    bool forward = true;
    float scale = 0.0f;
    int batch = 0;
    int channels = 0;
    int height = 0;
    int outChannels = 0;
    int outHeight = 0;
    int outWidth = 0;
    int stride = 0;
    int width = 0;
};

class UpsampleLayerCpuTest : public Test
{
protected:
    void SetUp(const UpsampleTestParams& params)
    {
        output_ = PxCpuVector(params.batch * params.outChannels * params.outHeight * params.outWidth, 0.0f);
    }

    void UpsampleCpuTest(const PxCpuVector& input, const PxCpuVector& expected, const UpsampleTestParams& params)
    {
        SetUp(params);

        UpsampleContext ctxt;
        ctxt.batch = params.batch;
        ctxt.channels = params.channels;
        ctxt.height = params.height;
        ctxt.input = &input;
        ctxt.outChannels = params.outChannels;
        ctxt.outHeight = params.outHeight;
        ctxt.outWidth = params.outWidth;
        ctxt.output = &output_;
        ctxt.scale = params.scale;
        ctxt.stride = params.stride;
        ctxt.width = params.width;

        upsampleForward(ctxt);

        EXPECT_THAT(*ctxt.output, Pointwise(FloatNear(1e-4), expected));
    }

private:
    PxCpuVector output_;
};

TEST_F(UpsampleLayerCpuTest, SimpleUpsample)
{
    UpsampleTestParams params;

    params.batch = 1;
    params.channels = 1;
    params.outChannels = 1;
    params.height = 2;
    params.width = 2;
    params.outHeight = 4;
    params.outWidth = 4;
    params.stride = 2;
    params.scale = 2.0f;

    PxCpuVector input{ 1.0, 2.0, 3.0, 4.0 };
    PxCpuVector expected{ 2.0, 2.0, 4.0, 4.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, 8.0, 8.0, 6.0, 6.0, 8.0, 8.0 };

    UpsampleCpuTest(input, expected, params);
}

#ifdef USE_CUDA

class UpsampleCudaTest : public Test
{
protected:
    void SetUp(const UpsampleTestParams& params)
    {
        output_ = PxCudaVector(params.batch * params.outChannels * params.outHeight * params.outWidth, 0.0f);
    }

    void UpsampleTest(const PxCudaVector& input, const PxCudaVector& expected, const UpsampleTestParams& params)
    {
        SetUp(params);

        UpsampleContext ctxt;
        ctxt.batch = params.batch;
        ctxt.channels = params.channels;
        ctxt.height = params.height;
        ctxt.inputGpu = &input;
        ctxt.outChannels = params.outChannels;
        ctxt.outHeight = params.outHeight;
        ctxt.outWidth = params.outWidth;
        ctxt.outputGpu = &output_;
        ctxt.scale = params.scale;
        ctxt.stride = params.stride;
        ctxt.width = params.width;

        upsampleForwardGpu(ctxt);

        EXPECT_THAT(ctxt.outputGpu->asVector(), Pointwise(FloatNear(1e-4), expected.asVector()));
    }

private:
    PxCudaVector output_;
};

TEST_F(UpsampleCudaTest, SimpleUpsampleCuda)
{
    UpsampleTestParams params;

    params.batch = 1;
    params.channels = 1;
    params.outChannels = 1;
    params.height = 2;
    params.width = 2;
    params.outHeight = 4;
    params.outWidth = 4;
    params.stride = 2;
    params.scale = 2.0f;

    PxCudaVector input{ 1.0, 2.0, 3.0, 4.0 };
    PxCudaVector expected{ 2.0, 2.0, 4.0, 4.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, 8.0, 8.0, 6.0, 6.0, 8.0, 8.0 };

    UpsampleTest(input, expected, params);
}

#endif  // USE_CUDA
