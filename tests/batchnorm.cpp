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

class BatchNormTest : public Test
{

protected:
    void SetUp(const BNTestParams& params)
    {
        biases_ = PxCpuTensor<1>({ (size_t) params.channels }, params.biases);
        scales_ = PxCpuTensor<1>({ (size_t) params.channels }, params.scales);
        rollingMean_ = PxCpuTensor<1>({ (size_t) params.channels }, params.rollingMean);
        rollingVar_ = PxCpuTensor<1>({ (size_t) params.channels }, params.rollingVar);
        output_ = PxCpuVector(params.batch * params.channels * params.outHeight * params.outWidth, 0.0f);
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

        EXPECT_THAT(*ctxt.output, Pointwise(FloatNear(1e-4), expected));
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

TEST_F(BatchNormTest, LargerBatchNorm)
{
    BNTestParams params;
    params.batch = 2;
    params.channels = 3;
    params.outHeight = 2;
    params.outWidth = 2;

    params.biases = { 1.0f, 2.0f, 3.0f };
    params.scales = { 0.5f, 1.0f, 1.5f };
    params.rollingMean = { 0.0f, 0.0f, 0.0f };
    params.rollingVar = { 1.0f, 1.0f, 1.0f };

    PxCpuVector input{
            // Batch 1
            1.0f, 2.0f, 3.0f,   // Channel 1
            4.0f, 5.0f, 6.0f,   // Channel 2
            7.0f, 8.0f, 9.0f,   // Channel 3
            // Batch 2
            10.0f, 11.0f, 12.0f,  // Channel 1
            13.0f, 14.0f, 15.0f,  // Channel 2
            16.0f, 17.0f, 18.0f   // Channel 3
    };

    PxCpuVector expected{
            1.5f, 2.0f, 2.5f, 3.0f,
            7.0f, 8.0f, 9.0f, 10.0f,
            16.5f, 18.0f, 19.5f, 21.0f,
            7.5f, 8.0f, 8.5f, 9.0f,
            19.0f, 20.0f, 2.0f, 2.0f,
            3.0f, 3.0f, 3.0f, 3.0f
    };


    BNTest(input, expected, params);
}

#ifdef USE_CUDA

class BatchNormCudaTest : public Test
{

protected:
    void SetUp(const BNTestParams& params)
    {
        biases_ = PxCudaTensor<1>({ (size_t) params.channels }, params.biases);
        scales_ = PxCudaTensor<1>({ (size_t) params.channels }, params.scales);
        rollingMean_ = PxCudaTensor<1>({ (size_t) params.channels }, params.rollingMean);
        rollingVar_ = PxCudaTensor<1>({ (size_t) params.channels }, params.rollingVar);
        output_ = PxCudaVector(params.batch * params.channels * params.outHeight * params.outWidth);
        cudnnSetTensor4dDescriptor(dstTens_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, params.batch, params.channels,
                                   params.outHeight, params.outWidth);
        cudnnSetTensor4dDescriptor(normTens_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, params.channels, 1, 1);
    }

    void BNTest(const PxCudaVector& input, const PxCudaVector& expected, const BNTestParams& params)
    {
        SetUp(params);

        BNContext ctxt;
        ctxt.cudnnContext = &cudnnContext_;
        ctxt.inputGpu = &input;
        ctxt.biasesGpu = &biases_;
        ctxt.scalesGpu = &scales_;
        ctxt.rollingMeanGpu = &rollingMean_;
        ctxt.rollingVarGpu = &rollingVar_;
        ctxt.normTens = &normTens_;
        ctxt.dstTens = &dstTens_;

        ctxt.outputGpu = &output_;
        ctxt.batch = params.batch;
        ctxt.channels = params.channels;
        ctxt.outHeight = params.outHeight;
        ctxt.outWidth = params.outWidth;

        batchNormForwardGpu(ctxt);

        EXPECT_THAT(ctxt.outputGpu->asVector(), Pointwise(FloatNear(1e-4), expected.asVector()));
    }

private:
    CudnnContext cudnnContext_;
    PxCudaTensor<1> biases_;
    PxCudaTensor<1> scales_;
    PxCudaTensor<1> rollingMean_;
    PxCudaTensor<1> rollingVar_;
    CudnnTensorDesc dstTens_;
    CudnnTensorDesc normTens_;
    PxCudaVector output_;
};

TEST_F(BatchNormCudaTest, SimpleBatchNorm)
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

    PxCudaVector input{ 1.0f, 2.0f };
    PxCudaVector expected{ 1.5f, 4.0f };

    BNTest(input, expected, params);
}

TEST_F(BatchNormCudaTest, LargerBatchNorm)
{
    BNTestParams params;
    params.batch = 2;
    params.channels = 3;
    params.outHeight = 2;
    params.outWidth = 2;

    params.biases = { 1.0f, 2.0f, 3.0f };
    params.scales = { 0.5f, 1.0f, 1.5f };
    params.rollingMean = { 0.0f, 0.0f, 0.0f };
    params.rollingVar = { 1.0f, 1.0f, 1.0f };

    PxCudaVector input{
            // Batch 1
            1.0f, 2.0f, 3.0f,   // Channel 1
            4.0f, 5.0f, 6.0f,   // Channel 2
            7.0f, 8.0f, 9.0f,   // Channel 3
            // Batch 2
            10.0f, 11.0f, 12.0f,  // Channel 1
            13.0f, 14.0f, 15.0f,  // Channel 2
            16.0f, 17.0f, 18.0f   // Channel 3
    };

    PxCudaVector expected{
            1.5f, 2.0f, 2.5f, 3.0f,
            7.0f, 8.0f, 9.0f, 10.0f,
            16.5f, 18.0f, 19.5f, 21.0f,
            7.5f, 8.0f, 8.5f, 9.0f,
            19.0f, 20.0f, 2.0f, 2.0f,
            3.0f, 3.0f, 3.0f, 3.0f
    };

    BNTest(input, expected, params);
}

#endif