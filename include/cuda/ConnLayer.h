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

#pragma once

#include "Adam.h"
#include "BiasKernels.cuh"
#include "Gemm.h"

namespace px {

template<>
class FCExtras<Device::CUDA>
{
protected:
    using V = typename Layer<Device::CUDA>::V;

    CudnnTensorDesc::Ptr yDesc_, sbmv_;
    V m_, v_;
    V bias_m_, bias_v_;
    V scale_m_, scale_v_;
    V preActivation_;
};

template<>
inline void ConnLayer<Device::CUDA>::setup()
{
    this->yDesc_ = std::make_unique<CudnnTensorDesc>();
    this->sbmv_ = std::make_unique<CudnnTensorDesc>();

    auto status = cudnnSetTensor4dDescriptor(*this->yDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                             this->batch(), this->outChannels(), this->outHeight(),
                                             this->outWidth());
    PX_CHECK_CUDNN(status);

    status = cudnnSetTensor4dDescriptor(*this->sbmv_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                        1, this->outChannels(), 1, 1);
    PX_CHECK_CUDNN(status);

    if (this->model().adamEnabled()) {
        m_ = V(weights_.size(), 0.0f);
        v_ = V(weights_.size(), 0.0f);
        bias_m_ = V(biases_.size(), 0.0f);
        bias_v_ = V(biases_.size(), 0.0f);
        scale_m_ = V(scales_.size(), 0.0f);
        scale_v_ = V(scales_.size(), 0.0f);
    }
}

template<>
inline std::streamoff ConnLayer<Device::CUDA>::loadWeights(std::istream& is)
{
    auto start = is.tellg();

    PxCpuVector biases(biases_.size());
    PxCpuVector weights(weights_.size());

    is.read((char*) biases.data(), biases.size() * sizeof(float));
    is.read((char*) weights.data(), weights.size() * sizeof(float));

    biases_.copy(biases);
    weights_.copy(weights);

    if (batchNorm_) {
        PxCpuVector scales(scales_.size());
        PxCpuVector rollingMean(rollingMean_.size());
        PxCpuVector rollingVar(rollingVar_.size());

        is.read((char*) scales.data(), scales.size() * sizeof(float));
        is.read((char*) rollingMean.data(), rollingMean.size() * sizeof(float));
        is.read((char*) rollingVar.data(), rollingVar.size() * sizeof(float));

        scales_.copy(scales);
        rollingMean_.copy(rollingMean);
        rollingVar_.copy(rollingVar);
    }

    PX_CHECK(is.good(), "Could not read weights");

    return is.tellg() - start;
}

template<>
inline std::streamoff ConnLayer<Device::CUDA>::saveWeights(std::ostream& os)
{
    auto start = os.tellp();

    PxCpuVector biases(biases_.size());
    PxCpuVector weights(weights_.size());

    biases.copyDevice(biases_.data(), biases_.size());
    weights.copyDevice(weights_.data(), weights_.size());

    os.write((char*) biases.data(), int(sizeof(float) * biases.size()));
    PX_CHECK(os.good(), "Could not write biases");

    os.write((char*) weights.data(), int(sizeof(float) * weights.size()));
    PX_CHECK(os.good(), "Could not write weights");

    if (batchNorm_) {
        PxCpuVector scales(scales_.size());
        PxCpuVector rollingMean(rollingMean_.size());
        PxCpuVector rollingVar(rollingVar_.size());

        scales.copyDevice(scales_.data(), scales_.size());
        rollingMean.copyDevice(rollingMean_.data(), rollingMean_.size());
        rollingVar.copyDevice(rollingVar_.data(), rollingVar_.size());

        os.write((char*) scales.data(), int(sizeof(float) * scales.size()));
        os.write((char*) rollingMean.data(), int(sizeof(float) * rollingMean.size()));
        os.write((char*) rollingVar.data(), int(sizeof(float) * rollingVar.size()));
    }

    PX_CHECK(os.good(), "Could not write connected layer parameters");

    return os.tellp() - start;
}

template<>
inline void ConnLayer<Device::CUDA>::forward(const V& input)
{
    Layer<Device::CUDA>::forward(input);

    auto m = this->batch();
    auto n = this->outputs();
    auto k = this->inputs();
    auto* a = input.data();
    auto* b = weights_.data();
    auto* c = output_.data();

    float alpha = 1.0f, beta = 1.0f;

    const auto& ctxt = this->cublasContext();
    cublasGemm(ctxt, false, true, m, n, k, alpha, a, k, b, k, beta, c, n);

    if (batchNorm_) {
        auto expAvgFactor = 0.01f;
        auto epsilon = 0.00001f;
        beta = 0.0f;

        const auto& cudnnContext = this->cudnnContext();

        cudnnStatus_t nstatus;
        if (training()) {
            x_.copy(output_);

            nstatus = cudnnBatchNormalizationForwardTraining(cudnnContext, CUDNN_BATCHNORM_SPATIAL,
                                                             &alpha, &beta, *yDesc_, x_.data(),
                                                             *yDesc_, this->output_.data(), *sbmv_, scales_.data(),
                                                             biases_.data(), expAvgFactor, rollingMean_.data(),
                                                             rollingVar_.data(), epsilon, mean_.data(),
                                                             var_.data());
        } else {
            nstatus = cudnnBatchNormalizationForwardInference(cudnnContext, CUDNN_BATCHNORM_SPATIAL,
                                                              &alpha, &beta, *yDesc_, this->output_.data(),
                                                              *yDesc_, this->output_.data(), *sbmv_, scales_.data(),
                                                              biases_.data(), rollingMean_.data(),
                                                              rollingVar_.data(),
                                                              epsilon);
        }
        PX_CHECK_CUDNN(nstatus);
    } else {
        addBiasGpu(this->output_.data(), biases_.data(), this->batch(), this->outputs(), 1);
    }

    this->preActivation_ = this->output_;
    activation_->apply(this->output_);
}

template<>
inline void ConnLayer<Device::CUDA>::backward(const V& input, V* grad)
{
    Layer<Device::CUDA>::backward(input, grad);

    activation_->gradient(this->preActivation_, this->delta_);

    if (batchNorm_) {
        float alpha = 1.0f, beta = 0.0f;
        auto expAvgFactor = 0.01f;
        auto epsilon = 0.00001f;
        const auto& ctxt = this->cudnnContext();

        auto status = cudnnBatchNormalizationBackward(ctxt, CUDNN_BATCHNORM_SPATIAL,
                                                      &alpha, &beta, &alpha, &beta, *yDesc_, this->x_.data(),
                                                      *yDesc_, this->delta_.data(), *yDesc_, this->xNorm_.data(),
                                                      *sbmv_, scales_.data(), scaleUpdates_.data(),
                                                      biasUpdates_.data(), epsilon, mean_.data(), var_.data());
        PX_CHECK_CUDNN(status);
    } else {
        backwardBiasGpu(biasUpdates_.data(), this->delta_.data(), this->batch(), this->outputs(), 1);
    }

    auto m = this->outputs();
    auto n = this->inputs();
    auto k = this->batch();
    auto* a = this->delta_.data();
    auto* b = input.data();
    auto* c = this->weightUpdates_.data();

    float alpha = 1.0f, beta = 0.0f;

    const auto& ctxt = this->cublasContext();
    cublasGemm(ctxt, true, false, m, n, k, alpha, a, m, b, n, beta, c, n);

    m = this->batch();
    n = this->inputs();
    k = this->outputs();
    a = this->delta_.data();
    b = this->weights_.data();
    c = grad == nullptr ? nullptr : grad->data();

    if (c) {
        cublasGemm(ctxt, false, false, m, n, k, alpha, a, k, b, n, beta, c, n);
    }
}

template<>
inline void ConnLayer<Device::CUDA>::update()
{
    const auto& net = this->model();
    auto learningRate = net.learningRate();
    auto momentum = net.momentum();
    auto decay = net.decay();
    auto batch = this->batch();

    Layer<Device::CUDA>::update();

    const auto& ctxt = this->cublasContext();

    if (net.adamEnabled()) {
        auto beta1 = net.adamBeta1();
        auto beta2 = net.adamBeta2();
        auto epsilon = net.adamEpsilon();
        auto t = net.seen();

        Adam<Device::CUDA> adam(ctxt, batch, t, learningRate, beta1, beta2, epsilon, decay);
        adam.update(weights_, weightUpdates_, m_, v_);
        adam.update(biases_, biasUpdates_, bias_m_, bias_v_);

        if (scales_.size()) {
            adam.update(scales_, scaleUpdates_, scale_m_, scale_v_);
        }

        return;
    }

    // update biases
    auto alpha = learningRate / batch;
    auto status = cublasSaxpy(ctxt, this->outputs(), &alpha, biasUpdates_.data(), 1, biases_.data(), 1);
    PX_CHECK_CUBLAS(status);

    status = cublasSscal(ctxt, this->outputs(), &momentum, biasUpdates_.data(), 1);
    PX_CHECK_CUBLAS(status);

    // update scales (if batch normalized)
    if (batchNorm_) {
        alpha = learningRate / batch;
        status = cublasSaxpy(ctxt, this->outputs(), &alpha, scaleUpdates_.data(), 1, scales_.data(), 1);
        PX_CHECK_CUBLAS(status);

        status = cublasSscal(ctxt, this->outputs(), &momentum, scaleUpdates_.data(), 1);
        PX_CHECK_CUBLAS(status);
    }

    // update weights with weight decay
    auto size = this->inputs() * this->outputs();

    alpha = -decay * batch;
    status = cublasSaxpy(ctxt, size, &alpha, weights_.data(), 1, weightUpdates_.data(), 1);
    PX_CHECK_CUBLAS(status);

    alpha = learningRate / batch;
    status = cublasSaxpy(ctxt, size, &alpha, weightUpdates_.data(), 1, weights_.data(), 1);
    PX_CHECK_CUBLAS(status);

    status = cublasSscal(ctxt, size, &momentum, weightUpdates_.data(), 1);
    PX_CHECK_CUBLAS(status);
}

template<>
inline void ConnLayer<Device::CUDA>::scaleGradients()
{
    Layer<Device::CUDA>::scaleGradients();

    this->scaleTensor(weightUpdates_);
    this->scaleTensor(biasUpdates_);
    this->scaleTensor(scaleUpdates_);
}

template<>
inline void ConnLayer<Device::CUDA>::clipGradients()
{
    Layer<Device::CUDA>::clipGradients();

    constrainGpu(weightUpdates_.size(), gradientClipValue_, weightUpdates_.data());
    constrainGpu(biasUpdates_.size(), gradientClipValue_, biasUpdates_.data());
    constrainGpu(scaleUpdates_.size(), gradientClipValue_, scaleUpdates_.data());
}

}   // px
