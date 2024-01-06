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

#include <cblas.h>

#include "Activation.h"
#include "ConvAlgo.h"
#include "ConvLayer.h"
#include "Error.h"
#include "Model.h"
#include "SHA1.h"

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
    activationFnc_ = Activations::get(activation);

    batchNormalize_ = property<bool>("batch_normalize", false);
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

    if (batchNormalize_) {
        scales_ = PxCpuTensor<1>({ (size_t) filters_ }, 1.0f);
        scaleUpdates_ = PxCpuTensor<1>({ (size_t) filters_ }, 0.0f);
        rollingMean_ = PxCpuTensor<1>({ (size_t) filters_ }, 0.f);
        rollingVar_ = PxCpuTensor<1>({ (size_t) filters_ }, 0.f);
        x_ = PxCpuVector(batch() * outputs(), 0.0f);
        xNorm_ = PxCpuVector(batch() * outputs(), 0.0f);
        mean_ = PxCpuTensor<1>({ (size_t) filters_ }, 0.f);
        meanDelta_ = PxCpuTensor<1>({ (size_t) filters_ }, 0.f);
        var_ = PxCpuTensor<1>({ (size_t) filters_ }, 0.f);
        varDelta_ = PxCpuTensor<1>({ (size_t) filters_ }, 0.f);
    }

    biases_ = PxCpuTensor<1>({ (size_t) filters_ }, 0.0f);
    biasUpdates_ = PxCpuTensor<1>({ (size_t) filters_ }, 0.0f);

    auto scale = std::sqrt(1.0f / (kernel_ * kernel_ * (channels() / groups_)));
    weights_ = random<PxCpuTensor<4>>({ (size_t) filters_,
                                        (size_t) (channels() / groups_),
                                        (size_t) kernel_,
                                        (size_t) kernel_ }, -1.0f, 1.0f) * scale;

    weightUpdates_ = PxCpuTensor<4>(weights_.shape(), 0.0f);

#ifdef USE_CUDA
    if (useGpu()) {
        setup_gpu();
    } else {
        column_ = PxCpuTensor<2>(
                { (size_t) kernel_ * kernel_ * channels() / groups_, (size_t) outHeight() * outWidth() }, 0.0f);
        output_ = PxCpuVector(batch() * outputs(), 0.0f);
        delta_ = PxCpuVector(batch() * outputs(), 0.0f);
    }
#else
    column_ = PxCpuTensor<2>(
            { (size_t) kernel_ * kernel_ * channels() / groups_, (size_t) outHeight() * outWidth() });
    output_ = PxCpuVector(batch() * outputs(), 0.0f);
    delta_ = PxCpuVector(batch() * outputs(), 0.0f);
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
        is.read((char*) biases_.data(), int(sizeof(float) * biases_.size()));
        is.read((char*) scales_.data(), int(sizeof(float) * scales_.size()));
        is.read((char*) rollingMean_.data(), int(sizeof(float) * rollingMean_.size()));
        is.read((char*) rollingVar_.data(), int(sizeof(float) * rollingVar_.size()));
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

std::streamoff ConvLayer::saveWeights(std::ostream& os)
{
    auto start = os.tellp();

    if (batchNormalize_) {
        os.write((char*) biases_.data(), int(sizeof(float) * biases_.size()));
        os.write((char*) scales_.data(), int(sizeof(float) * scales_.size()));
        os.write((char*) rollingMean_.data(), int(sizeof(float) * rollingMean_.size()));
        os.write((char*) rollingVar_.data(), int(sizeof(float) * rollingVar_.size()));
    } else {
        os.write((char*) biases_.data(), int(biases_.size() * sizeof(float)));
        PX_CHECK(os.good(), "Could not write biases");
    }

    os.write((char*) weights_.data(), int(sizeof(float) * weights_.size()));
    PX_CHECK(os.good(), "Could not write weights");

    return os.tellp() - start;
}

void ConvLayer::forward(const PxCpuVector& input)
{
    Layer::forward(input);

    auto ctxt = makeContext(input);
    convolutionalForward(ctxt);

    if (batchNormalize_) {
        auto bnContext = makeBNContext(output_);
        batchNormForward(bnContext);
    } else {
        addBias(output_.data(), biases_.data(), batch(), filters_, outHeight() * outWidth());
    }

    activationFnc_->apply(output_);
}

void ConvLayer::backward(const PxCpuVector& input)
{
    Layer::backward(input);

    activationFnc_->gradient(output_, delta_);

    if (batchNormalize_) {
        auto bnContext = makeBNContext(output_);
        batchNormBackward(bnContext);
    } else {
        backwardBias(biasUpdates_.data(), delta_.data(), batch(), filters_, outHeight() * outWidth());
    }

    auto ctxt = makeContext(input);
    convolutionalBackward(ctxt);
}

void ConvLayer::update()
{
    const auto& net = model();
    auto learningRate = net.learningRate();
    auto momentum = net.momentum();
    auto decay = net.decay();

    cblas_saxpy(weights_.size(), -decay * batch(), weights_.data(), 1, weightUpdates_.data(), 1);
    cblas_saxpy(weights_.size(), learningRate / batch(), weightUpdates_.data(), 1, weights_.data(), 1);
    cblas_sscal(weights_.size(), momentum, weightUpdates_.data(), 1);

    cblas_saxpy(filters_, learningRate / batch(), biasUpdates_.data(), 1, biases_.data(), 1);
    cblas_sscal(filters_, momentum, biasUpdates_.data(), 1);

    if (scales_.size()) {
        cblas_saxpy(filters_, learningRate / batch(), scaleUpdates_.data(), 1, scales_.data(), 1);
        cblas_sscal(filters_, momentum, scaleUpdates_.data(), 1);
    }
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
    ctxt.netDelta = model().delta();
    ctxt.nweights = weights_.size();
    ctxt.outHeight = outHeight();
    ctxt.outWidth = outWidth();
    ctxt.output = &output_;
    ctxt.padding = padding_;
    ctxt.stride = stride_;
    ctxt.weightUpdates = &weightUpdates_;
    ctxt.weights = &weights_;
    ctxt.width = width();

    return ctxt;
}

BNContext ConvLayer::makeBNContext(const PxCpuVector& input)
{
    BNContext ctxt;

    ctxt.input = &input;
    ctxt.output = &output_;
    ctxt.x = &x_;
    ctxt.xNorm = &xNorm_;
    ctxt.biases = &biases_;
    ctxt.biasUpdates = &biasUpdates_;
    ctxt.delta = &delta_;
    ctxt.meanDelta = &meanDelta_;
    ctxt.scales = &scales_;
    ctxt.scaleUpdates = &scaleUpdates_;
    ctxt.mean = &mean_;
    ctxt.var = &var_;
    ctxt.varDelta = &varDelta_;
    ctxt.rollingMean = &rollingMean_;
    ctxt.rollingVar = &rollingVar_;

    ctxt.batch = batch();
    ctxt.channels = outChannels();
    ctxt.outHeight = outHeight();
    ctxt.outWidth = outWidth();

    ctxt.training = training();

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
