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

#ifndef PIXIENN_CONVLAYER_T_H
#define PIXIENN_CONVLAYER_T_H

namespace px {

template<typename T = cpu_array>
class convlayer_t : public layer_t<T>
{
public:
    using base_type = layer_t<T>;
    using value_type = typename T::value_type;
    using tensor_type = typename layer_t<T>::tensor_type;

    virtual ~convlayer_t() = default;

    std::ostream& print(std::ostream& os) override;

    std::streamoff loadDarknetWeights(std::istream& is) override;

    void forward(const tensor_type& input) override;

protected:
    convlayer_t(const model_t <T>& model, const YAML::Node& layerDef);

private:
    friend layer_factory<T>;

    template<std::size_t N = 1>
    using tensor_t = typename T::template tensor_type<N>;

    tensor_t<4> weights_;
    tensor_t<1> biases_;
    tensor_t<2> column_;

    int dilation_ = 0, filters_, kernel_, padding_, stride_, groups_;
    std::string activation_;

    typename layer_t<T>::Ptr batchNormalize_;
    typename activation_t<T>::Ptr activationFnc_;
};

template<typename T>
convlayer_t<T>::convlayer_t(const model_t <T>& model, const YAML::Node& layerDef) : layer_t<T>(model, layerDef)
{
    activation_ = base_type::template property<std::string>("activation", "logistic");
    activationFnc_ = activation_t<T>::get(activation_);

    auto batchNormalize = base_type::template property<bool>("batch_normalize", false);
    dilation_ = base_type::template property<int>("dilation", 0);
    filters_ = base_type::template property<int>("filters", 1);
    kernel_ = base_type::template property<int>("kernel", 1);
    auto pad = base_type::template property<bool>("pad", 0);
    padding_ = pad ? kernel_ / 2 : 0;
    stride_ = base_type::template property<int>("stride", 1);
    groups_ = std::max(1, base_type::template property<int>("groups", 1));

    base_type::setOutChannels(filters_);
    base_type::setOutHeight((base_type::height() + 2 * padding_ - kernel_) / stride_ + 1);
    base_type::setOutWidth((base_type::width() + 2 * padding_ - kernel_) / stride_ + 1);
    base_type::setOutputs(base_type::outHeight() * base_type::outWidth() * base_type::outChannels());

    if (batchNormalize) {
        auto def = layerDef;
        def["type"] = "batchnorm";
        def["channels"] = base_type::outChannels();
        def["height"] = base_type::outHeight();
        def["width"] = base_type::outWidth();
        batchNormalize_ = layer_t<T>::create(model, def);
    } else { ;
        biases_ = xt::zeros<value_type>({ filters_ });    // FIXME: this is an h->d copy...
    }

    // FIXME: these are h->d copies...
    weights_ = xt::random::rand<value_type>({ filters_, base_type::channels() / groups_, kernel_, kernel_ });
    column_ = xt::empty<value_type>({ kernel_ * kernel_ * base_type::channels() / groups_,
                                      base_type::outHeight() * base_type::outWidth() });
    base_type::output_ = xt::empty<value_type>({ base_type::batch(), base_type::outChannels(), base_type::outHeight(),
                                                 base_type::outWidth() });
}

template<typename T>
std::ostream& convlayer_t<T>::print(std::ostream& os)
{
    return os;
}

template<typename T>
std::streamoff convlayer_t<T>::loadDarknetWeights(std::istream& is)
{
    auto start = is.tellg();

    if (batchNormalize_) {
        batchNormalize_->loadDarknetWeights(is);
    } else {
        cpu_tensor<1> biases = biases_; // device to host
        is.read((char*) biases.data(), biases.size() * sizeof(float));
        PX_CHECK(is.good(), "Could not read biases");
        biases_ = biases;   // host to device
    }

    cpu_tensor<4> weights = weights_;   // device to host
    is.read((char*) weights.data(), sizeof(float) * weights.size());
    PX_CHECK(is.good(), "Could not read weights");
    weights_ = weights; // host to device

    return is.tellg() - start;
}

template<typename T>
inline void convlayer_t<T>::forward(const tensor_type& input)
{
    abort();
}

template<>
inline void convlayer_t<cuda_array>::forward(const tensor_type& input)
{
    CudnnContext context;

    int n = input.shape(0);
    int c = input.shape(1);
    int h = input.shape(2);
    int w = input.shape(3);

    CudnnTensorDesc xDesc;
    auto status = cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
    PX_CHECK_CUDNN(status);

    n = weights_.shape(0);
    c = weights_.shape(1);
    h = weights_.shape(2);
    w = weights_.shape(3);

    CudnnFilterDesc wDesc;
    status = cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w);
    PX_CHECK_CUDNN(status);

    CudnnConvDesc convDesc;
    status = cudnnSetConvolution2dDescriptor(convDesc,
                                             padding_,
                                             padding_,
                                             stride_,
                                             stride_,
                                             dilation_,
                                             dilation_,
                                             CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    PX_CHECK_CUDNN(status);

    status = cudnnGetConvolution2dForwardOutputDim(convDesc, xDesc, wDesc, &n, &c, &h, &w);
    PX_CHECK_CUDNN(status);

    CudnnTensorDesc yDesc;
    status = cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
    PX_CHECK_CUDNN(status);

    int retCount;
    cudnnConvolutionFwdAlgoPerf_t fwdAlgoPerf;

    status = cudnnGetConvolutionForwardAlgorithm_v7(context, xDesc, wDesc, convDesc, yDesc, 1, &retCount, &fwdAlgoPerf);
    PX_CHECK_CUDNN(status);
    PX_CHECK_CUDNN(fwdAlgoPerf.status);

    cuda_tensor_t<uint8_t, 1> ws = decltype(ws)::from_shape({ fwdAlgoPerf.memory });

    float one = 1;

    status = cudnnConvolutionForward(context,
                                     &one,
                                     xDesc,
                                     input.data().get(),
                                     wDesc,
                                     weights_.data().get(),
                                     convDesc,
                                     fwdAlgoPerf.algo,
                                     ws.data().get(),
                                     fwdAlgoPerf.memory,
                                     &one,
                                     yDesc,
                                     output_.data().get());

    PX_CHECK_CUDNN(status);

    if (batchNormalize_) {
        batchNormalize_->forward(output_);
        output_ = batchNormalize_->output();
    } else {
        // FIXME: add_bias(output_.data(), biases_.data(), batch(), outChannels(), outHeight() * outWidth());
    }

    activationFnc_->apply(output_);
}

}   // namespace px

#endif // PIXIENN_CONVLAYER_T_H
