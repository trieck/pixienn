/********************************************************************************
* Copyright 2020 Thomas A. Rieck, All Rights Reserved
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

#include <cblas.h>

#include "Activation.h"
#include "ConvLayer.h"
#include "Error.h"
#include "Utility.h"

namespace px {

ConvLayer::ConvLayer(const Model& model, const YAML::Node& layerDef) : Layer(model, layerDef),
                                                                       filters_(0), kernel_(0), padding_(0), stride_(0),
                                                                       groups_(0)
{
}

void ConvLayer::setup()
{
    activation_ = property<std::string>("activation", "logistic");
    activationFnc_ = Activation::get(activation_);

    auto batchNormalize = property<bool>("batch_normalize", false);
    dilation_ = property<int>("dilation", 0);
    filters_ = property<int>("filters", 1);
    kernel_ = property<int>("kernel", 1);
    auto pad = property<bool>("pad", 0);
    padding_ = pad ? kernel_ / 2 : 0;
    stride_ = property<int>("stride", 1);
    groups_ = std::max(1, property<int>("groups", 1));

    setOutChannels(filters_);
    setOutHeight((height() + 2 * padding_ - kernel_) / stride_ + 1);
    setOutWidth((width() + 2 * padding_ - kernel_) / stride_ + 1);
    setOutputs(outHeight() * outWidth() * outChannels());

    if (batchNormalize) {
        auto def = layerDef();
        def["type"] = "batchnorm";
        def["inputs"] = outputs();
        def["channels"] = outChannels();
        def["height"] = outHeight();
        def["width"] = outWidth();
        batchNormalize_ = Layer::create(model(), def);
    } else {
        biases_ = PxCpuTensor<1>({ (size_t) filters_ }, 0.f);
    }

    weights_ = random<decltype(weights_)>(
            { (size_t) filters_, (size_t) (channels() / groups_), (size_t) kernel_, (size_t) kernel_ });
    column_ = PxCpuTensor<2>({ (size_t) kernel_ * kernel_ * channels() / groups_, (size_t) outHeight() * outWidth() });
    output_ = PxCpuVector(batch() * outChannels() * outHeight() * outWidth());

#ifdef USE_CUDA
    setup_gpu();
#endif
}

std::ostream& ConvLayer::print(std::ostream& os)
{
    Layer::print(os, "conv", { height(), width(), channels() }, { outHeight(), outWidth(), outChannels() },
                 filters_, std::array<int, 3>{ kernel_, kernel_, stride_ });

    return os;
}

std::streamoff ConvLayer::loadDarknetWeights(std::istream& is)
{
    auto start = is.tellg();

    if (batchNormalize_) {
        batchNormalize_->loadDarknetWeights(is);
    } else {
        is.read((char*) biases_.data(), int(biases_.size() * sizeof(float)));
        PX_CHECK(is.good(), "Could not read biases");
#if USE_CUDA
        if (useGpu()) {
            biasesGpu_.copy(biases_);
        }
#endif
    }

    is.read((char*) weights_.data(), int(sizeof(float) * weights_.size()));
    PX_CHECK(is.good(), "Could not read weights");

#if USE_CUDA
    if (useGpu()) { ;
        weightsGpu_.copy(weights_);
    }
#endif

    return is.tellg() - start;
}

void ConvLayer::forward(const PxCpuVector& input)
{
    int m = filters_ / groups_;
    int n = outWidth() * outHeight();
    int k = kernel_ * kernel_ * channels() / groups_;

    int nweights = weights_.size();
    const auto* pweights = weights_.data();

    const auto* pin = input.data();
    auto* pout = output_.data();

    auto alpha = 1.0f;
    auto beta = 1.0f;

    for (auto i = 0; i < batch(); ++i) {
        for (auto j = 0; j < groups_; ++j) {
            const auto* im = pin + (i * groups_ + j) * channels() / groups_ * height() * width();
            const auto* a = pweights + j * nweights / groups_;
            const auto* b = kernel_ == 1 ? im : column_.data();
            auto* c = pout + (i * groups_ + j) * n * m;

            if (kernel_ != 1) {
                im2col_cpu(im, channels() / groups_, height(), width(), kernel_, stride_, padding_, column_.data());
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, k, b, n, beta, c, n);
        }
    }

    if (batchNormalize_) {
        batchNormalize_->forward(output_);
        output_ = batchNormalize_->output();
    } else {
        add_bias(output_.data(), biases_.data(), batch(), outChannels(), outHeight() * outWidth());
    }

    activationFnc_->apply(output_);
}

#if USE_CUDA

#include <CudaUtils.cuh>

void ConvLayer::setup_gpu()
{
    if (useGpu()) {
        if (!batchNormalize_) {
            biasesGpu_ = PxCudaTensor<1>({ (size_t) filters_ }, 0);
        }

        weightsGpu_ = random<decltype(weightsGpu_)>(
                { (size_t) filters_, (size_t) (channels() / groups_), (size_t) kernel_, (size_t) kernel_ });

        outputGpu_ = PxCudaVector(batch() * outChannels() * outHeight() * outWidth());
        xDesc_ = std::make_unique<CudnnTensorDesc>();
        yDesc_ = std::make_unique<CudnnTensorDesc>();
        wDesc_ = std::make_unique<CudnnFilterDesc>();
        convDesc_ = std::make_unique<CudnnConvDesc>();

        auto status = cudnnSetTensor4dDescriptor(*xDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch(), channels(),
                                                 height(),
                                                 width());
        PX_CHECK_CUDNN(status);

        status = cudnnSetConvolution2dDescriptor(*convDesc_, padding_, padding_, stride_, stride_, dilation_, dilation_,
                                                 CUDNN_CROSS_CORRELATION,
                                                 CUDNN_DATA_FLOAT);
        PX_CHECK_CUDNN(status);

        status = cudnnSetConvolutionGroupCount(*convDesc_, groups_);
        PX_CHECK_CUDNN(status);

        status = cudnnSetFilter4dDescriptor(*wDesc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filters_,
                                            channels() / groups_,
                                            kernel_, kernel_);
        PX_CHECK_CUDNN(status);

        int n, c, h, w;
        status = cudnnGetConvolution2dForwardOutputDim(*convDesc_, *xDesc_, *wDesc_, &n, &c, &h, &w);
        PX_CHECK_CUDNN(status);

        PX_CHECK(n == batch() && c == outChannels() && h == outHeight() && w == outWidth(),
                 "Output layer dimensions mismatch!");

        status = cudnnSetTensor4dDescriptor(*yDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
        PX_CHECK_CUDNN(status);

        const auto& context = cudnnContext();

        int requestCount = 1, resultCount = 0;
        status = cudnnGetConvolutionForwardAlgorithmMaxCount(context, &requestCount);
        PX_CHECK_CUDNN(status);

        auto results = std::make_unique<cudnnConvolutionFwdAlgoPerf_t[]>(requestCount);

        status = cudnnFindConvolutionForwardAlgorithm(context,
                                                      *xDesc_,
                                                      *wDesc_,
                                                      *convDesc_,
                                                      *yDesc_,
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
}

void ConvLayer::forwardGpu(const PxCudaVector& input)
{
    float alpha = 1.f;
    float beta = 1.f;

    const auto& context = cudnnContext();
    auto status = cudnnConvolutionForward(context,
                                          &alpha,
                                          *xDesc_,
                                          input.data(),
                                          *wDesc_,
                                          weightsGpu_.data(),
                                          *convDesc_,
                                          bestAlgo_,
                                          workspace_.data(),
                                          workspace_.size() * sizeof(float),
                                          &beta,
                                          *yDesc_,
                                          outputGpu_.data());
    PX_CHECK_CUDNN(status);

    if (batchNormalize_) {
        batchNormalize_->forwardGpu(outputGpu_);
        outputGpu_ = batchNormalize_->outputGpu();
    } else {
        add_bias_gpu(outputGpu_.data(), biasesGpu_.data(), batch(), outChannels(), outHeight() * outWidth());
    }

    activationFnc_->applyGpu(outputGpu_);
}

#endif  // USE_CUDA

}   // px
