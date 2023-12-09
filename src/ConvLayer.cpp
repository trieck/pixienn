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

#include "Activation.h"
#include "ConvAlgo.h"
#include "ConvLayer.h"
#include "Error.h"

#ifdef USE_CUDA

#include <CudaUtils.cuh>

#endif

namespace px {

ConvLayer::ConvLayer(Model& model, const YAML::Node& layerDef) : Layer(model, layerDef), filters_(0), kernel_(0),
                                                                 padding_(0), stride_(0),
                                                                 groups_(0)
{
}

void ConvLayer::setup()
{
    auto activation = property<std::string>("activation", "logistic");
    activationFnc_ = Activation::get(activation);

    auto batchNormalize = property<bool>("batch_normalize", false);
    dilation_ = property<int>("dilation", 1);
    filters_ = property<int>("filters", 1);
    kernel_ = property<int>("kernel", 1);
    auto pad = property<bool>("pad", false);
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
        biasUpdates_ = PxCpuTensor<1>({ (size_t) filters_ }, 0.f);
    }

    auto scale = std::sqrt(2.0f / (kernel_ * kernel_ * channels() / groups_));
    weights_ = random<decltype(weights_)>({ (size_t) filters_,
                                            (size_t) (channels() / groups_),
                                            (size_t) kernel_,
                                            (size_t) kernel_ }) * scale;
    
    weightUpdates_ = PxCpuTensor<4>(weights_.shape());

#ifdef USE_CUDA
    if (useGpu()) {
        setup_gpu();
    } else {
        column_ = PxCpuTensor<2>(
                { (size_t) kernel_ * kernel_ * channels() / groups_, (size_t) outHeight() * outWidth() });
        output_ = PxCpuVector(batch() * outChannels() * outHeight() * outWidth(), 0.0f);
        delta_ = PxCpuVector(batch() * outChannels() * outHeight() * outWidth(), 0.0f);
    }
#else
    column_ = PxCpuTensor<2>(
            { (size_t) kernel_ * kernel_ * channels() / groups_, (size_t) outHeight() * outWidth() });
    output_ = PxCpuVector(batch() * outChannels() * outHeight() * outWidth(), 0.0f);
    delta_ = PxCpuVector(batch() * outChannels() * outHeight() * outWidth(), 0.0f);
#endif
}

std::ostream& ConvLayer::print(std::ostream& os)
{
    Layer::print(os, "conv", { height(), width(), channels() }, { outHeight(), outWidth(), outChannels() },
                 filters_, std::array<int, 3>{ kernel_, kernel_, stride_ });

    return os;
}

std::streamoff ConvLayer::loadWeights(std::istream& is)
{
    auto start = is.tellg();

    if (batchNormalize_) {
        batchNormalize_->loadWeights(is);
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
    auto ctxt = makeContext(input);

    convolutionalForward(ctxt);

    if (batchNormalize_) {
        batchNormalize_->forward(output_);
        output_.copy(batchNormalize_->output());
    } else {
        addBias(output_.data(), biases_.data(), batch(), filters_, outHeight() * outWidth());
    }

    activationFnc_->apply(output_);
}

void ConvLayer::backward(const PxCpuVector& input)
{
    activationFnc_->gradient(output_, delta_);

    if (batchNormalize_) {
        batchNormalize_->backward(output_);
        output_.copy(batchNormalize_->output());    // TODO: is that right?
    } else {
        backwardBias(biasUpdates_.data(), delta_.data(), batch(), filters_, outHeight() * outWidth());
    }

    auto ctxt = makeContext(input);
    convolutionalBackward(ctxt);
}

ConvContext ConvLayer::makeContext(const PxCpuVector& input)
{
    ConvContext ctxt{};
    ctxt.batch = batch();
    ctxt.channels = channels();
    ctxt.column = &column_;
    ctxt.delta = &delta_;
    ctxt.dilation = dilation_;
    ctxt.filters = filters_;
    ctxt.groups = groups_;
    ctxt.height = height();
    ctxt.input = &input;
    ctxt.kernel = kernel_;
    ctxt.nweights = weights_.size();
    ctxt.outHeight = outHeight();
    ctxt.outWidth = outWidth();
    ctxt.output = &output_;
    ctxt.padding = padding_;
    ctxt.stride = stride_;
    ctxt.weights = &weights_;
    ctxt.weightUpdates = &weightUpdates_;
    ctxt.width = width();

    return ctxt;
}

#if USE_CUDA

void ConvLayer::setup_gpu()
{
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

    if (hasOption("find-best-algo")) {
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
    ConvContext ctxt = makeContext(input);

    convolutionalForwardGpu(ctxt);

    if (batchNormalize_) {
        batchNormalize_->forwardGpu(outputGpu_);
        outputGpu_ = batchNormalize_->outputGpu();
    } else {
        addBiasGpu(outputGpu_.data(), biasesGpu_.data(), batch(), filters_, outHeight() * outWidth());
    }

    activationFnc_->applyGpu(outputGpu_);
}

ConvContext ConvLayer::makeContext(const PxCudaVector& input)
{
    ConvContext ctxt{};

    ctxt.batch = batch();
    ctxt.bestAlgo = bestAlgo_;
    ctxt.channels = channels();
    ctxt.convDesc = convDesc_.get();
    ctxt.cudnnContext = &cudnnContext();
    ctxt.dilation = dilation_;
    ctxt.filters = filters_;
    ctxt.groups = groups_;
    ctxt.height = height();
    ctxt.inputGpu = &input;
    ctxt.kernel = kernel_;
    ctxt.nweights = weightsGpu_.size();
    ctxt.outHeight = outHeight();
    ctxt.outWidth = outWidth();
    ctxt.outputGpu = &outputGpu_;
    ctxt.padding = padding_;
    ctxt.stride = stride_;
    ctxt.wDesc = wDesc_.get();
    ctxt.weightsGpu = &weightsGpu_;
    ctxt.width = width();
    ctxt.workspace = &workspace_;
    ctxt.xDesc = xDesc_.get();
    ctxt.yDesc = yDesc_.get();

    return ctxt;
}

#endif  // USE_CUDA

}   // px
