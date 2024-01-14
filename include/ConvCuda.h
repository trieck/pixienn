/********************************************************************************
* Copyright 2023 Thomas A. Rieck, All Rights Reserved
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

#pragma once

#include "Cudnn.h"

namespace px {

template<>
class CVExtras<Device::CUDA>
{
protected:
    using V = typename Layer<Device::CUDA>::V;

    V fwdWorkspace_, bwdFilterWorkspace_;
    CudnnTensorDesc::Ptr xDesc_, yDesc_, sbmv_;
    CudnnConvDesc::Ptr convDesc_;
    CudnnFilterDesc::Ptr wDesc_;
    cudnnConvolutionFwdAlgo_t fwdAlgo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
};

template<>
inline void ConvLayer<Device::CUDA>::setup()
{
    xDesc_ = std::make_unique<CudnnTensorDesc>();
    yDesc_ = std::make_unique<CudnnTensorDesc>();
    sbmv_ = std::make_unique<CudnnTensorDesc>();
    wDesc_ = std::make_unique<CudnnFilterDesc>();
    convDesc_ = std::make_unique<CudnnConvDesc>();

    auto status = cudnnSetTensor4dDescriptor(*xDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                             this->batch(), this->channels(), this->height(), this->width());
    PX_CHECK_CUDNN(status);

    status = cudnnSetConvolution2dDescriptor(*convDesc_, padding_, padding_, stride_, stride_, dilation_, dilation_,
                                             CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    PX_CHECK_CUDNN(status);

    status = cudnnSetConvolutionGroupCount(*convDesc_, groups_);
    PX_CHECK_CUDNN(status);

    status = cudnnSetFilter4dDescriptor(*wDesc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                        filters_, this->channels() / groups_, kernel_, kernel_);

    PX_CHECK_CUDNN(status);

    int n, c, h, w;
    status = cudnnGetConvolution2dForwardOutputDim(*convDesc_, *xDesc_, *wDesc_, &n, &c, &h, &w);
    PX_CHECK_CUDNN(status);

    PX_CHECK(n == batch() && c == outChannels() && h == outHeight() && w == outWidth(),
             "Output layer dimensions mismatch");

    status = cudnnSetTensor4dDescriptor(*yDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
    PX_CHECK_CUDNN(status);

    status = cudnnSetTensor4dDescriptor(*sbmv_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, c, 1, 1);
    PX_CHECK_CUDNN(status);

    const auto& ctxt = this->cudnnContext();
    int requestCount = 1, returnedCount = 0;

    status = cudnnGetConvolutionForwardAlgorithmMaxCount(ctxt, &requestCount);
    PX_CHECK_CUDNN(status);

    auto fwdResults = std::make_unique<cudnnConvolutionFwdAlgoPerf_t[]>(requestCount);

    status = cudnnFindConvolutionForwardAlgorithm(ctxt, *xDesc_, *wDesc_, *convDesc_, *yDesc_, requestCount,
                                                  &returnedCount, fwdResults.get());
    PX_CHECK_CUDNN(status);

    auto workspaceSize = std::numeric_limits<size_t>::max();
    for (auto i = 0; i < returnedCount; ++i) {
        if (fwdResults[i].status == CUDNN_STATUS_SUCCESS) {
            if (fwdResults[i].memory < workspaceSize) {
                fwdAlgo_ = fwdResults[i].algo;
                workspaceSize = fwdResults[i].memory;
            }
        }
    }

    fwdWorkspace_ = V(workspaceSize / sizeof(float), 0.0f);

    status = cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(ctxt, &requestCount);
    PX_CHECK_CUDNN(status);

    auto bkwdResults = std::make_unique<cudnnConvolutionBwdFilterAlgoPerf_t[]>(requestCount);

    status = cudnnFindConvolutionBackwardFilterAlgorithm(ctxt, *xDesc_, *yDesc_, *convDesc_, *wDesc_, requestCount,
                                                         &returnedCount, bkwdResults.get());
    PX_CHECK_CUDNN(status);

    workspaceSize = std::numeric_limits<size_t>::max();
    for (auto i = 0; i < returnedCount; ++i) {
        if (bkwdResults[i].status == CUDNN_STATUS_SUCCESS) {
            if (bkwdResults[i].memory < workspaceSize) {
                bwdFilterAlgo_ = bkwdResults[i].algo;
                workspaceSize = bkwdResults[i].memory;
            }
        }
    }

    bwdFilterWorkspace_ = V(workspaceSize / sizeof(float), 0.0f);
}

template<>
inline std::streamoff ConvLayer<Device::CUDA>::loadWeights(std::istream& is)
{
    auto start = is.tellg();

    PxCpuVector biases(biases_.size());
    PxCpuVector weights(weights_.size());

    if (batchNorm_) {
        PxCpuVector scales(scales_.size());
        PxCpuVector rollingMean(rollingMean_.size());
        PxCpuVector rollingVar(rollingVar_.size());

        is.read((char*) biases.data(), biases.size() * sizeof(float));
        is.read((char*) scales.data(), scales.size() * sizeof(float));
        is.read((char*) rollingMean.data(), rollingMean.size() * sizeof(float));
        is.read((char*) rollingVar.data(), rollingVar.size() * sizeof(float));

        scales_.copy(scales);
        rollingMean_.copy(rollingMean);
        rollingVar_.copy(rollingVar);
    } else {
        is.read((char*) biases.data(), biases.size() * sizeof(float));
    }

    is.read((char*) weights.data(), weights.size() * sizeof(float));

    biases_.copy(biases);
    weights_.copy(weights);

    PX_CHECK(is.good(), "Could not read weights");

    return is.tellg() - start;
}

template<>
inline void ConvLayer<Device::CUDA>::forward(const V& input)
{
    Layer<Device::CUDA>::forward(input);

    auto alpha = 1.0f;
    auto beta = 0.0f;
    auto expAvgFactor = 0.01f;
    auto epsilon = 0.00001f;

    const auto& ctxt = this->cudnnContext();

    auto status = cudnnConvolutionForward(ctxt, &alpha, *xDesc_, input.data(), *wDesc_, weights_.data(),
                                          *convDesc_, fwdAlgo_, fwdWorkspace_.data(),
                                          fwdWorkspace_.size() * sizeof(float),
                                          &beta, *yDesc_, this->output_.data());
    PX_CHECK_CUDNN(status);

    if (batchNorm_) {
        if (training()) {
            status = cudnnBatchNormalizationForwardTraining(ctxt, CUDNN_BATCHNORM_SPATIAL,
                                                            &alpha, &beta, *yDesc_, this->output_.data(),
                                                            *yDesc_, this->output_.data(), *sbmv_, scales_.data(),
                                                            biases_.data(), expAvgFactor, rollingMean_.data(),
                                                            rollingVar_.data(), epsilon, mean_.data(), var_.data());
        } else {
            status = cudnnBatchNormalizationForwardInference(ctxt, CUDNN_BATCHNORM_SPATIAL,
                                                             &alpha, &beta, *yDesc_, this->output_.data(),
                                                             *yDesc_, this->output_.data(), *sbmv_, scales_.data(),
                                                             biases_.data(), rollingMean_.data(), rollingVar_.data(),
                                                             epsilon);

        }
        PX_CHECK_CUDNN(status);
    } else {
        addBiasGpu(this->output_.data(), biases_.data(), this->batch(), this->outputs(), 1);
    }

    activation_->apply(this->output_);
}

template<>
inline void ConvLayer<Device::CUDA>::backward(const V& input)
{
    Layer<Device::CUDA>::backward(input);

    auto alpha = 1.0f;
    auto beta = 1.0f;

    auto status = cudnnConvolutionBackwardFilter(this->cudnnContext(), &alpha, *xDesc_, input.data(), *yDesc_,
                                                 this->delta_.data(), *convDesc_, bwdFilterAlgo_,
                                                 bwdFilterWorkspace_.data(),
                                                 bwdFilterWorkspace_.size() * sizeof(float),
                                                 &beta, *wDesc_, weightUpdates_.data());
    PX_CHECK_CUDNN(status);
}

}