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

#include "ConvAlgo.h"

using namespace px;
using namespace testing;

struct ConvTestParams
{
    int batch;
    int channels;
    int dilation;
    int filters;
    int groups;
    int height;
    int kernel;
    int outHeight;
    int outWidth;
    int padding;
    int stride;
    int width;
};

class ConvTest : public Test
{
protected:
    void SetUp(const ConvTestParams& params)
    {
        column_ = PxCpuTensor<2>({ static_cast<size_t>(params.kernel * params.kernel * params.channels / params.groups),
                                   static_cast<size_t>(params.outHeight * params.outWidth) }, 0.0f);

        output_ = PxCpuVector(params.batch * params.filters * params.outHeight * params.outWidth, 0.0f);
    }

    void convolutionTest(const PxCpuVector& input, const PxCpuTensor<4>& weights, const PxCpuVector& expected,
                         const ConvTestParams& params)
    {
        SetUp(params);

        ConvContext ctxt;
        ctxt.input = &input;
        ctxt.weights = &weights;
        ctxt.column = &column_;
        ctxt.output = &output_;
        ctxt.batch = params.batch;
        ctxt.channels = params.channels;
        ctxt.dilation = params.dilation;
        ctxt.filters = params.filters;
        ctxt.groups = params.groups;
        ctxt.height = params.height;
        ctxt.kernel = params.kernel;
        ctxt.outHeight = params.outHeight;
        ctxt.outWidth = params.outWidth;
        ctxt.padding = params.padding;
        ctxt.stride = params.stride;
        ctxt.width = params.width;

        convolutionalForward(ctxt);

        EXPECT_THAT(*ctxt.output, Pointwise(FloatNear(1e-5), expected));
    }

private:
    PxCpuTensor<2> column_;
    PxCpuVector output_;
};

TEST_F(ConvTest, SimpleConvolution)
{
    ConvTestParams params;
    params.batch = 1;
    params.channels = 1;
    params.dilation = 1;
    params.filters = 1;
    params.groups = 1;
    params.height = 2;
    params.kernel = 2;
    params.outHeight = 1;
    params.outWidth = 1;
    params.padding = 0;
    params.stride = 1;
    params.width = 2;

    PxCpuVector input{ 1.0f, 2.0f, 3.0f, 4.0f };
    PxCpuTensor<4> weights({ 1, 1, 2, 2 }, { 0.5f, 0.5f, 0.5f, 0.5f });
    PxCpuVector expected{ 5.0f };

    convolutionTest(input, weights, expected, params);
}

TEST_F(ConvTest, LargerConvolution)
{
    PxCpuVector input{
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f
    };

    PxCpuTensor<4> weights({ 2, 2, 2, 2 }, {
            0.5f, 0.5f, 0.5f, 0.5f,
            0.5f, 0.5f, 0.5f, 0.5f,
            0.5f, 0.5f, 0.5f, 0.5f,
            0.5f, 0.5f, 0.5f, 0.5f
    });

    PxCpuVector expected{
            7.0f, 9.0f, 11.0f,
            15.0f, 17.0f, 19.0f,
            23.0f, 25.0f, 27.0f
    };

    ConvTestParams params;
    params.batch = 1;
    params.channels = 1;
    params.dilation = 1;
    params.filters = 1;
    params.groups = 1;
    params.height = 4;
    params.kernel = 2;
    params.outHeight = 3;
    params.outWidth = 3;
    params.padding = 0;
    params.stride = 1;
    params.width = 4;

    convolutionTest(input, weights, expected, std::move(params));
}

#ifdef USE_CUDA

class CUDNNTest : public Test
{
protected:
    void SetUp(const ConvTestParams& params)
    {
        auto status = cudnnSetTensor4dDescriptor(xDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, params.batch,
                                                 params.channels, params.height, params.width);
        PX_CHECK_CUDNN(status);

        status = cudnnSetConvolution2dDescriptor(convDesc_, params.padding, params.padding, params.stride,
                                                 params.stride, params.dilation, params.dilation,
                                                 CUDNN_CROSS_CORRELATION,
                                                 CUDNN_DATA_FLOAT);
        PX_CHECK_CUDNN(status);

        status = cudnnSetConvolutionGroupCount(convDesc_, params.groups);
        PX_CHECK_CUDNN(status);

        status = cudnnSetFilter4dDescriptor(wDesc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, params.filters,
                                            params.channels / params.groups, params.kernel, params.kernel);
        PX_CHECK_CUDNN(status);

        int n, c, h, w;
        status = cudnnGetConvolution2dForwardOutputDim(convDesc_, xDesc_, wDesc_, &n, &c, &h, &w);
        PX_CHECK_CUDNN(status);

        PX_CHECK(n == params.batch && c == params.filters && h == params.outHeight && w == params.outWidth,
                 "Output layer dimensions mismatch!");

        status = cudnnSetTensor4dDescriptor(yDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
        PX_CHECK_CUDNN(status);

        auto requestCount = 1, resultCount = 0;
        status = cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnContext_, &requestCount);
        PX_CHECK_CUDNN(status);

        auto results = std::make_unique<cudnnConvolutionFwdAlgoPerf_t[]>(requestCount);

        status = cudnnFindConvolutionForwardAlgorithm(cudnnContext_,
                                                      xDesc_,
                                                      wDesc_,
                                                      convDesc_,
                                                      yDesc_,
                                                      requestCount,
                                                      &resultCount,
                                                      results.get());
        PX_CHECK_CUDNN(status);

        size_t workspaceSize = std::numeric_limits<size_t>::max();
        for (auto i = 0; i < resultCount; ++i) {
            if (results[i].status == CUDNN_STATUS_SUCCESS && results[i].memory < workspaceSize) {
                workspaceSize = results[i].memory;
                bestAlgo_ = results[i].algo;
            }
        }

        workspace_ = PxCudaTensor<1>({ workspaceSize / sizeof(float) });
    }

    void TearDown() override
    {
    }

    void convolutionTestGpu(const PxCudaVector& input, const PxCudaTensor<4>& weights, const PxCudaVector& expected,
                            const ConvTestParams& params)
    {
        SetUp(params);

        ConvContext ctxt;
        ctxt.cudnnContext = &cudnnContext_;
        ctxt.xDesc = &xDesc_;
        ctxt.yDesc = &yDesc_;
        ctxt.wDesc = &wDesc_;
        ctxt.convDesc = &convDesc_;
        ctxt.bestAlgo = bestAlgo_;
        ctxt.workspace = &workspace_;

        ctxt.inputGpu = &input;
        ctxt.weightsGpu = &weights;
        ctxt.batch = params.batch;
        ctxt.channels = params.channels;
        ctxt.dilation = params.dilation;
        ctxt.filters = params.filters;
        ctxt.groups = params.groups;
        ctxt.height = params.height;
        ctxt.kernel = params.kernel;
        ctxt.outHeight = params.outHeight;
        ctxt.outWidth = params.outWidth;
        ctxt.padding = params.padding;
        ctxt.stride = params.stride;
        ctxt.width = params.width;

        PxCudaVector output(ctxt.batch * ctxt.filters * ctxt.outHeight * ctxt.outWidth, 0.0f);
        ctxt.outputGpu = &output;

        convolutionalForwardGpu(ctxt);

        EXPECT_THAT(ctxt.outputGpu->asVector(), Pointwise(FloatNear(1e-5), expected.asVector()));
    }

private:
    CudnnContext cudnnContext_;
    CudnnTensorDesc xDesc_, yDesc_;
    CudnnFilterDesc wDesc_;
    CudnnConvDesc convDesc_;
    cudnnConvolutionFwdAlgo_t bestAlgo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    PxCudaTensor<1> workspace_;
};

TEST_F(CUDNNTest, SimpleConvolutionGpu)
{
    ConvTestParams params;
    params.batch = 1;
    params.channels = 1;
    params.dilation = 1;
    params.filters = 1;
    params.groups = 1;
    params.height = 2;
    params.kernel = 2;
    params.outHeight = 1;
    params.outWidth = 1;
    params.padding = 0;
    params.stride = 1;
    params.width = 2;

    PxCudaVector input{ 1.0f, 2.0f, 3.0f, 4.0f };
    PxCudaTensor<4> weights({ 1, 1, 2, 2 }, { 0.5f, 0.5f, 0.5f, 0.5f });
    PxCudaVector expected{ 5.0f };

    convolutionTestGpu(input, weights, expected, std::move(params));
}

TEST_F(CUDNNTest, LargerConvolutionGpu)
{
    ConvTestParams params;
    params.batch = 1;
    params.channels = 1;
    params.dilation = 1;
    params.filters = 1;
    params.groups = 1;
    params.height = 4;
    params.kernel = 2;
    params.outHeight = 3;
    params.outWidth = 3;
    params.padding = 0;
    params.stride = 1;
    params.width = 4;

    PxCudaVector input{
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f
    };

    PxCudaTensor<4> weights({ 2, 2, 2, 2 }, {
            0.5f, 0.5f, 0.5f, 0.5f,
            0.5f, 0.5f, 0.5f, 0.5f,
            0.5f, 0.5f, 0.5f, 0.5f,
            0.5f, 0.5f, 0.5f, 0.5f
    });

    PxCudaVector expected{
            7.0f, 9.0f, 11.0f,
            15.0f, 17.0f, 19.0f,
            23.0f, 25.0f, 27.0f
    };

    convolutionTestGpu(input, weights, expected, std::move(params));
}

#endif // USE_CUDA
