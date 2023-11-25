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

#include "BatchNormAlgo.h"

using namespace px;
using namespace testing;

class BatchNormTest : public Test
{

protected:
    struct BNTestParams
    {
        int batch = 0;
        int channels = 0;
        int outHeight = 0;
        int outWidth = 0;

        PxCpuVector biases;
        PxCpuVector scales;
        PxCpuVector rollingMean;
        PxCpuVector rollingVar;
    };

    void SetUp(const BNTestParams& params)
    {
        biases_ = PxCpuTensor<1>({ (size_t) params.channels }, params.biases);
        scales_ = PxCpuTensor<1>({ (size_t) params.channels }, params.scales);
        rollingMean_ = PxCpuTensor<1>({ (size_t) params.channels }, params.rollingMean);
        rollingVar_ = PxCpuTensor<1>({ (size_t) params.channels }, params.rollingVar);
        output_ = PxCpuVector(params.batch * params.channels * params.outHeight * params.outWidth);
    }

    void BNTest(const PxCpuVector& input, const PxCpuVector& expected, const BNTestParams& params)
    {
        SetUp(params);

        BNContext ctxt;
        ctxt.input = &input;
        ctxt.biases = &biases_;
        ctxt.scales = &scales_;
        ctxt.rollingMean = &rollingMean_;
        ctxt.rollingVar = &rollingVar_;

        ctxt.output = &output_;
        ctxt.batch = params.batch;
        ctxt.channels = params.channels;
        ctxt.outHeight = params.outHeight;
        ctxt.outWidth = params.outWidth;

        batchNormForward(ctxt);

        EXPECT_THAT(*ctxt.output, Pointwise(FloatNear(1e-5), expected));
    }

private:
    PxCpuTensor<1> biases_;
    PxCpuTensor<1> scales_;
    PxCpuTensor<1> rollingMean_;
    PxCpuTensor<1> rollingVar_;
    PxCpuVector output_;
};

TEST_F(BatchNormTest, SimpleBatchNorm)
{
    BNTestParams params;
    params.batch = 1;
    params.channels = 2;
    params.outHeight = 1;
    params.outWidth = 1;

    params.biases = { 1.0f, 2.0f };
    params.scales = { 0.5f, 1.0f };
    params.rollingMean = { 0.0f, 0.0f };
    params.rollingVar = { 1.0f, 1.0f };

    PxCpuVector input{ 1.0f, 2.0f };
    PxCpuVector expected{ 1.5f, 4.0f };

    BNTest(input, expected, params);
}
