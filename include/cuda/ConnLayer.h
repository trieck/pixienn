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

namespace px {

template<>
class FCExtras<Device::CUDA>
{
protected:
    CudnnTensorDesc::Ptr yDesc_, sbmv_;
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
inline void ConnLayer<Device::CUDA>::forward(const V& input)
{
    auto m = this->outputs();
    auto n = this->batch();
    auto k = this->inputs();
    auto* a = weights_.data();
    auto* b = input.data();
    auto* c = output_.data();

    float alpha = 1.0f, beta = 1.0f;
    auto expAvgFactor = 0.01f;
    auto epsilon = 0.00001f;
    const auto& ctxt = this->cublasContext();

    auto status = cublasSgemm(ctxt,
                              CUBLAS_OP_T, /* transpose A */
                              CUBLAS_OP_N, /* transpose B */
                              m, /* M */
                              n, /* N */
                              k, /* K */
                              &alpha, /* alpha */
                              a, /* A */
                              k, /* lda */
                              b, /* B */
                              k, /* ldb */
                              &beta, /* beta */
                              c, /* C */
                              m /* ldc */
    );

    PX_CHECK_CUBLAS(status);

    if (batchNorm_) {
        beta = 0.0f;

        const auto& cudnnContext = this->cudnnContext();

        cudnnStatus_t nstatus;
        if (training()) {
            nstatus = cudnnBatchNormalizationForwardTraining(cudnnContext, CUDNN_BATCHNORM_SPATIAL,
                                                             &alpha, &beta, *yDesc_, this->output_.data(),
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

    activation_->apply(this->output_);
}

template<>
inline void ConnLayer<Device::CUDA>::backward(const V& input)
{
    // TODO: implement
}

}