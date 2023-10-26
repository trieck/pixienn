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

#include "ConnLayer.h"

#if USE_CUDA

#include "BiasKernels.cuh"

#endif

namespace px {

ConnLayer::ConnLayer(const Model& model, const YAML::Node& layerDef) : Layer(model, layerDef)
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
        auto def = layerDef;
        def["type"] = "batchnorm";
        def["channels"] = outChannels();
        def["height"] = outHeight();
        def["width"] = outWidth();
        batchNormalize_ = Layer::create(model, def);
    } else {
#ifdef USE_CUDA
        biases_ = PxDevVector<float>(outputs(), 0.f);
#else
        biases_ = zeros<float>({ outputs() });
#endif
    }

#ifdef USE_CUDA
    weights_ = PxDevVector<float>::random(inputs() * outputs(), -1.f, 1.f);
    output_ = PxDevVector<float>(batch() * outputs(), 0.f);
#else
    weights_ = random::rand<float>({ inputs(), outputs() });
    output_ = zeros<float>({ batch() * outputs() });
#endif
}

std::ostream& ConnLayer::print(std::ostream& os)
{
    Layer::print(os, "connected", { height(), width(), channels() }, { outHeight(), outWidth(), outChannels() });

    return os;
}

std::streamoff ConnLayer::loadDarknetWeights(std::istream& is)
{
    auto start = is.tellg();

#ifdef USE_CUDA
    std::vector<float> biases(biases_.size());
    std::vector<float> weights(weights_.size());

    is.read((char*) biases.data(), biases_.size() * sizeof(float));
    PX_CHECK(is.good(), "Could not read biases");
    biases_.hostCopy(biases);

    is.read((char*) weights.data(), weights_.size() * sizeof(float));
    PX_CHECK(is.good(), "Could not read weights");
    weights_.hostCopy(weights);
#else
    is.read((char*) biases_.data(), biases_.size() * sizeof(float));
    PX_CHECK(is.good(), "Could not read biases");

    is.read((char*) weights_.data(), sizeof(float) * weights_.size());
    PX_CHECK(is.good(), "Could not read weights");
#endif

    if (batchNormalize_) {
        is.read((char*) &scales_, sizeof(float));
        is.read((char*) &rollingMean_, sizeof(float));
        is.read((char*) &rollingVar_, sizeof(float));
        PX_CHECK(is.good(), "Could not read batch_normalize parameters");
    }

    return is.tellg() - start;
}

void ConnLayer::forward(const PxDevVector<float>& input)
{
#ifdef USE_CUDA
    forward_gpu(input);
#else
    output_.fill(0);

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
        add_bias(c, biases_.data(), m, outChannels(), outHeight() * outWidth());


    }

    activationFnc_->apply(output_);
#endif
}

#if USE_CUDA

void ConnLayer::forward_gpu(const PxDevVector<float>& input)
{
    output_.fill(0);

    auto m = outputs();
    auto n = batch();
    auto k = inputs();
    auto* a = weights_.data();
    auto* b = input.data();
    auto* c = output_.data();

    assert(input.size() == k * n);
    assert(weights_.size() == k * m);
    assert(output_.size() == m * n);

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
        batchNormalize_->forward(output_);
        output_ = batchNormalize_->output();
    } else {
        add_bias_gpu(c, biases_.data(), n, m, 1);
    }

    activationFnc_->apply(output_);
}

#endif  // USE_CUDA

} // px
