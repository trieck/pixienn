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

#include "ConnAlgo.h"

using namespace px;
using namespace testing;

struct ConnectedTestParams
{
    int batch = 0;
    int inputs = 0;
    int outputs = 0;

    PxCpuVector weights;
};

class ConnectedLayerTest : public Test
{
protected:
    void SetUp(const ConnectedTestParams& params)
    {
        weights_ = PxCpuTensor<2>({ (size_t) params.outputs, (size_t) params.inputs }, params.weights);
        output_ = PxCpuVector(params.batch * params.outputs);
    }

    void ConnectedTest(const PxCpuVector& input, const PxCpuVector& expected, const ConnectedTestParams& params)
    {
        SetUp(params);

        ConnContext ctxt;
        ctxt.input = &input;
        ctxt.weights = &weights_;
        ctxt.output = &output_;
        ctxt.batch = params.batch;
        ctxt.inputs = params.inputs;
        ctxt.outputs = params.outputs;

        connectedForward(ctxt);

        EXPECT_THAT(*ctxt.output, Pointwise(FloatNear(1e-4), expected));
    }

private:
    PxCpuTensor<2> weights_;
    PxCpuTensor<1> biases_;
    PxCpuVector output_;
};

TEST_F(ConnectedLayerTest, SimpleConnectedLayer)
{
    ConnectedTestParams params;
    params.batch = 1;
    params.inputs = 2;
    params.outputs = 1;

    params.weights = { 1.0f, 2.0f };

    PxCpuVector input{ 2.0f, 3.0f };
    PxCpuVector expected{ 1.0f * 2.0f + 2.0f * 3.0f };

    ConnectedTest(input, expected, params);
}

TEST_F(ConnectedLayerTest, LargerConnectedLayer)
{
    ConnectedTestParams params;
    params.batch = 2;
    params.inputs = 3;
    params.outputs = 2;

    params.weights = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };

    PxCpuVector input{
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f
    };

    PxCpuVector expected{14.0, 32.0, 32.0, 77.0};

    ConnectedTest(input, expected, params);
}
#ifdef USE_CUDA

class ConnectedLayerCudaTest : public Test
{
protected:
    void SetUp(const ConnectedTestParams& params)
    {
        weights_ = PxCudaTensor<2>({ (size_t) params.outputs, (size_t) params.inputs }, params.weights);
        output_ = PxCudaVector(params.batch * params.outputs);
    }

    void ConnectedTest(const PxCudaVector& input, const PxCudaVector& expected, const ConnectedTestParams& params)
    {
        SetUp(params);

        ConnContext ctxt;
        ctxt.cublasContext = &cublasContext_;
        ctxt.inputGpu = &input;
        ctxt.weightsGpu = &weights_;
        ctxt.outputGpu = &output_;
        ctxt.batch = params.batch;
        ctxt.inputs = params.inputs;
        ctxt.outputs = params.outputs;

        connectedForwardGpu(ctxt);

        EXPECT_THAT(ctxt.outputGpu->asVector(), Pointwise(FloatNear(1e-4), expected.asVector()));
    }

private:
    PxCudaTensor<2> weights_;
    PxCudaTensor<1> biases_;
    PxCudaVector output_;
    CublasContext cublasContext_;
};

TEST_F(ConnectedLayerCudaTest, SimpleConnectedLayer)
{
    ConnectedTestParams params;
    params.batch = 1;
    params.inputs = 2;
    params.outputs = 1;

    params.weights = { 1.0f, 2.0f };

    PxCudaVector input{ 2.0f, 3.0f };
    PxCudaVector expected{ 1.0f * 2.0f + 2.0f * 3.0f };

    ConnectedTest(input, expected, params);
}

TEST_F(ConnectedLayerCudaTest, LargerConnectedLayer)
{
    ConnectedTestParams params;
    params.batch = 2;
    params.inputs = 3;
    params.outputs = 2;

    params.weights = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };

    PxCudaVector input{
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f
    };

    PxCudaVector expected{14.0, 32.0, 32.0, 77.0};

    ConnectedTest(input, expected, params);
}


#endif // USE_CUDA
