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
#include <xtensor/xadapt.hpp>
#include <xtensor/xrandom.hpp>

#include "Activation.h"
#include "ConvLayer.h"
#include "Error.h"
#include "Utility.h"
#include "SHA1.h"

using namespace xt;

namespace px {

ConvLayer::ConvLayer(const Model& model, const YAML::Node& layerDef) : Layer(model, layerDef)
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
        auto def = layerDef;
        def["type"] = "batchnorm";
        def["channels"] = outChannels();
        def["height"] = outHeight();
        def["width"] = outWidth();
        batchNormalize_ = Layer::create(model, def);
    } else {
        biases_ = zeros<float>({ filters_ });
#ifdef USE_CUDA
        biasesGpu_ = PxDevVector<float>(filters_, 0);
#endif
    }

    weights_ = random::rand<float>({ filters_, channels() / groups_, kernel_, kernel_ });
    column_ = empty<float>({ kernel_ * kernel_ * channels() / groups_, outHeight() * outWidth() });
    output_ = empty<float>({ batch(), outChannels(), outHeight(), outWidth() });

#ifdef USE_CUDA
    weightsGpu_ = PxDevVector<float>::random(filters_ * channels() / groups_ * kernel_ * kernel_);
    columnGpu_ = PxDevVector<float>(kernel_ * kernel_ * channels() / groups_ * outHeight() * outWidth());
    outputGpu_ = PxDevVector<float>(batch() * outChannels() * outHeight() * outWidth());
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
        is.read((char*) biases_.data(), biases_.size() * sizeof(float));
        PX_CHECK(is.good(), "Could not read biases");
#if USE_CUDA
        biasesGpu_.fromHost(biases_);
#endif
    }

    is.read((char*) weights_.data(), sizeof(float) * weights_.size());
    PX_CHECK(is.good(), "Could not read weights");

#if USE_CUDA
    weightsGpu_.fromHost(weights_);
#endif

    return is.tellg() - start;
}

void ConvLayer::forward(const xt::xarray<float>& input)
{
    auto result = sha1(input.data(), input.size());
    std::cout << "sha1[conv](cpu input):  " << result << std::endl;

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

    result = sha1(output_.data(), outputs());
    std::cout << "sha1[conv](cpu output): " << result << std::endl;
}

#if USE_CUDA

#include <CudaUtils.cuh>

constexpr std::uint32_t MEMORY_LIMIT = (1 << 31);

void ConvLayer::setup_gpu()
{
    auto status = cudnnSetTensor4dDescriptor(xDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch(), channels(),
                                             height(),
                                             width());
    PX_CHECK_CUDNN(status);

    status = cudnnSetConvolution2dDescriptor(convDesc_, padding_, padding_, stride_, stride_, dilation_, dilation_,
                                             CUDNN_CROSS_CORRELATION,
                                             CUDNN_DATA_FLOAT);
    PX_CHECK_CUDNN(status);

    status = cudnnSetConvolutionGroupCount(convDesc_, groups_);
    PX_CHECK_CUDNN(status);

    status = cudnnSetFilter4dDescriptor(wDesc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filters_, channels() / groups_,
                                        kernel_, kernel_);
    PX_CHECK_CUDNN(status);

    int n, c, h, w;
    status = cudnnGetConvolution2dForwardOutputDim(convDesc_, xDesc_, wDesc_, &n, &c, &h, &w);
    PX_CHECK_CUDNN(status);

    PX_CHECK(n == batch() && c == outChannels() && h == outHeight() && w == outWidth(),
             "Output layer dimensions mismatch!");

    status = cudnnSetTensor4dDescriptor(yDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
    PX_CHECK_CUDNN(status);

    int count = 0;
    cudnnConvolutionFwdAlgoPerf_t results[2 * CUDNN_CONVOLUTION_FWD_ALGO_COUNT];

    const auto& context = cudnnContext();
    status = cudnnFindConvolutionForwardAlgorithm(context,
                                                  xDesc_,
                                                  wDesc_,
                                                  convDesc_,
                                                  yDesc_,
                                                  CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
                                                  &count,
                                                  results);
    PX_CHECK_CUDNN(status);

    for (auto i = 0; i < count; ++i) {
        if (results[i].memory < MEMORY_LIMIT) {
            bestAlgo_ = results[i].algo;
            break;
        }
    }

    size_t workspaceSize = 0;
    status = cudnnGetConvolutionForwardWorkspaceSize(context,
                                                     xDesc_,
                                                     wDesc_,
                                                     convDesc_,
                                                     yDesc_,
                                                     bestAlgo_,
                                                     &workspaceSize);
    PX_CHECK_CUDNN(status);

    workspace_ = PxDevVector<float>(workspaceSize);
}

void ConvLayer::forwardGpu(const PxDevVector<float>& input)
{
    auto inputCpu = input.asHost();

    assert(inputCpu.size() == inputs());
    auto result = sha1(inputCpu.data(), inputCpu.size());
    std::cout << "sha1[conv](gpu input):  " << result << std::endl;

    float alpha = 1.f;
    float beta = 1.f;

    const auto& context = cudnnContext();
    auto status = cudnnConvolutionForward(context,
                                          &alpha,
                                          xDesc_,
                                          input.data(),
                                          wDesc_,
                                          weightsGpu_.data(),
                                          convDesc_,
                                          bestAlgo_,
                                          workspace_.data(),
                                          workspace_.size(),
                                          &beta,
                                          yDesc_,
                                          outputGpu_.data());
    PX_CHECK_CUDNN(status);

    if (batchNormalize_) {
        batchNormalize_->forwardGpu(outputGpu_);
        outputGpu_ = batchNormalize_->outputGpu();
    } else {
        add_bias_gpu(outputGpu_.data(), biasesGpu_.data(), batch(), outChannels(), outHeight() * outWidth());
    }

    activationFnc_->applyGpu(outputGpu_);

    auto output = outputGpu_.asHost();
    result = sha1(output.data(), outputs());
    std::cout << "sha1[conv](gpu output): " << result << std::endl;
}

#endif  // USE_CUDA

}   // px
