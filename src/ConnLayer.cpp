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

#include "ConnLayer.h"
#include "Utility.h"

#if USE_CUDA

#include "BiasKernels.cuh"

#endif

namespace px {

ConnLayer::ConnLayer(const Model& model, const YAML::Node& layerDef) : Layer(model, layerDef), scales_(0),
                                                                       rollingMean_(0), rollingVar_(0)
{
}

void ConnLayer::setup()
{
    activation_ = property<std::string>("activation", "logistic");
    activationFnc_ = Activation::get(activation_);

    auto batchNormalize = property<bool>("batch_normalize", false);

    setChannels(inputs());
    setHeight(1);
    setWidth(1);

    setOutputs(property<int>("output", 1));
    setOutHeight(1);
    setOutWidth(1);
    setOutChannels(outputs());

    if (batchNormalize) {
        auto def = layerDef();
        def["type"] = "batchnorm";
        def["channels"] = outChannels();
        def["height"] = outHeight();
        def["width"] = outWidth();
        batchNormalize_ = Layer::create(model(), def);
    } else {
        biases_ = PxCpuTensor<1>({ (size_t) outputs() });
    }

    weights_ = random<decltype(weights_)>({ (size_t) inputs(), (size_t) outputs() });
    output_ = PxCpuVector(batch() * outputs());

#ifdef USE_CUDA
    setupGpu();
#endif
}

#ifdef USE_CUDA // USE_CUDA

void ConnLayer::setupGpu()
{
    if (useGpu()) {
        if (!batchNormalize_) {
            biasesGpu_ = PxCudaTensor<1>({ (size_t) outputs() }, 0.f);
        }

        weightsGpu_ = random<decltype(weightsGpu_)>({ (size_t) inputs(), (size_t) outputs() });
        outputGpu_ = PxCudaVector(batch() * outputs(), 0.f);
    }
}

#endif // USE_CUDA

std::ostream& ConnLayer::print(std::ostream& os)
{
    Layer::print(os, "connected", { height(), width(), channels() }, { outHeight(), outWidth(), outChannels() });

    return os;
}

std::streamoff ConnLayer::loadDarknetWeights(std::istream& is)
{
    auto start = is.tellg();

    is.read((char*) biases_.data(), biases_.size() * sizeof(float));
    PX_CHECK(is.good(), "Could not read biases");

    is.read((char*) weights_.data(), sizeof(float) * weights_.size());
    PX_CHECK(is.good(), "Could not read weights");

#ifdef USE_CUDA
    if (useGpu()) {
        biasesGpu_.copy(biases_);
        weightsGpu_.copy(weights_);
    }
#endif

    if (batchNormalize_) {
        is.read((char*) &scales_, sizeof(float));
        is.read((char*) &rollingMean_, sizeof(float));
        is.read((char*) &rollingVar_, sizeof(float));
        PX_CHECK(is.good(), "Could not read batch_normalize parameters");
    }

    return is.tellg() - start;
}

void ConnLayer::forward(const PxCpuVector& input)
{
    auto m = batch();
    auto n = outputs();
    auto k = inputs();
    auto* a = input.data();
    auto* b = weights_.data();
    auto* c = output_.data();

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f, a, k, b, k, 1.0f, c, n);

    if (batchNormalize_) {
        batchNormalize_->forward(output_);
        output_ = batchNormalize_->output();
    } else {
        addBias(c, biases_.data(), m, outChannels(), outHeight() * outWidth());
    }

    activationFnc_->apply(output_);
}

#if USE_CUDA

void ConnLayer::forwardGpu(const PxCudaVector& input)
{
    auto m = outputs();
    auto n = batch();
    auto k = inputs();
    auto* a = weightsGpu_.data();
    auto* b = input.data();
    auto* c = outputGpu_.data();

    float alpha = 1.0f, beta = 1.0f;

    const auto& context = cublasContext();

    auto status = cublasSgemm(context,
                              CUBLAS_OP_T,  /* transpose A */
                              CUBLAS_OP_N,  /* transpose B */
                              m,            /* M */
                              n,            /* N */
                              k,            /* K */
                              &alpha,       /* alpha */
                              a,            /* A */
                              k,            /* lda */
                              b,            /* B */
                              k,            /* ldb */
                              &beta,        /* beta */
                              c,            /* C */
                              m             /* ldc */
    );

    PX_CHECK_CUBLAS(status);

    if (batchNormalize_) {
        batchNormalize_->forwardGpu(outputGpu_);
        outputGpu_ = batchNormalize_->outputGpu();
    } else {
        add_bias_gpu(c, biasesGpu_.data(), n, m, 1);
    }

    activationFnc_->applyGpu(outputGpu_);
}

#endif  // USE_CUDA

} // px
