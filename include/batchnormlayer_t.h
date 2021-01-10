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

#ifndef PIXIENN_BATCHNORMLAYER_T_H
#define PIXIENN_BATCHNORMLAYER_T_H

namespace px {

template<typename T = cpu_array>
class batchnormlayer_t : public layer_t<T>
{
public:
    using base_type = layer_t<T>;
    using value_type = typename T::value_type;
    using tensor_type = typename layer_t<T>::tensor_type;

    virtual ~batchnormlayer_t() = default;

    std::ostream& print(std::ostream& os) override;

    std::streamoff loadDarknetWeights(std::istream& is) override;

    void forward(const tensor_type& input) override;

protected:
    batchnormlayer_t(const model_t <T>& model, const YAML::Node& layerDef);

private:
    friend layer_factory<T>;

    template<std::size_t N = 1>

    using tensor_t = typename T::template tensor_type<N>;

    tensor_t<1> biases_, scales_, rollingMean_, rollingVar_;
};

template<typename T>
batchnormlayer_t<T>::batchnormlayer_t(const model_t <T>& model, const YAML::Node& layerDef)
        : layer_t<T>(model, layerDef)
{
    // FIXME: these are h->d copies...
    biases_ = xt::zeros<float>({ base_type::channels() });
    scales_ = xt::ones<float>({ base_type::channels() });
    rollingMean_ = xt::zeros<float>({ base_type::channels() });
    rollingVar_ = xt::zeros<float>({ base_type::channels() });

    base_type::setOutChannels(base_type::channels());
    base_type::setOutHeight(base_type::height());
    base_type::setOutWidth(base_type::width());
    base_type::setOutputs(base_type::outHeight() * base_type::outWidth() * base_type::outChannels());

    base_type::output_ = xt::empty<float>(
            { base_type::batch(), base_type::outChannels(), base_type::outHeight(), base_type::outWidth() });
}

template<typename T>
std::ostream& batchnormlayer_t<T>::print(std::ostream& os)
{
    return os;
}

template<typename T>
std::streamoff batchnormlayer_t<T>::loadDarknetWeights(std::istream& is)
{
    auto start = is.tellg();

    cpu_tensor<1> biases(biases_), scales(scales_), rollingMean(rollingMean_), rollingVar(rollingVar_);

    is.read((char*) biases.data(), sizeof(float) * biases.size());
    is.read((char*) scales.data(), sizeof(float) * scales.size());
    is.read((char*) rollingMean.data(), sizeof(float) * rollingMean.size());
    is.read((char*) rollingVar.data(), sizeof(float) * rollingVar.size());
    PX_CHECK(is.good(), "Could not read batch_normalize parameters");

    // host to device
    biases_ = biases;
    scales_ = scales;
    rollingMean_ = rollingMean;
    rollingVar_ = rollingVar;

    return is.tellg() - start;
}

template<typename T>
inline void batchnormlayer_t<T>::forward(const tensor_type& input)
{
// output_ = input;
//
//    auto b = batch();
//    auto c = outChannels();
//    auto size = outHeight() * outWidth();
//
//    normalize_cpu(output_.data(), rollingMean_.data(), rollingVar_.data(), b, c, size);
//
//    scale_bias(output_.data(), scales_.data(), b, c, size);
//    add_bias(output_.data(), biases_.data(), b, c, size);

    abort();
}

template<>
inline void batchnormlayer_t<cuda_array>::forward(const tensor_type& input)
{
    CudnnContext context;

    int n = input.shape(0);
    int c = input.shape(1);
    int h = input.shape(2);
    int w = input.shape(3);

    CudnnTensorDesc xDesc;
    auto status = cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
    PX_CHECK_CUDNN(status);

    n = base_type::output_.shape(0);
    c = base_type::output_.shape(1);
    h = base_type::output_.shape(2);
    w = base_type::output_.shape(3);

    CudnnTensorDesc yDesc;
    status = cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
    PX_CHECK_CUDNN(status);

    c = base_type::channels();

    CudnnTensorDesc biasDesc;
    status = cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, c, 1, 1);
    PX_CHECK_CUDNN(status);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    status = cudnnBatchNormalizationForwardInference(context,
                                                     CUDNN_BATCHNORM_SPATIAL,
                                                     &alpha,
                                                     &beta,
                                                     xDesc,
                                                     input.data().get(),
                                                     yDesc,
                                                     base_type::output_.data().get(),
                                                     biasDesc,
                                                     scales_.data().get(),
                                                     biases_.data().get(),
                                                     rollingMean_.data().get(),
                                                     rollingVar_.data().get(),
                                                     CUDNN_BN_MIN_EPSILON);

    PX_CHECK_CUDNN(status);


}

}   // namespace px

#endif // PIXIENN_BATCHNORMLAYER_T_H
