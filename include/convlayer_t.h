
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

//#include "Activation.h"

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
    convlayer_t(const Model& model, const YAML::Node& layerDef);

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

    //Activation::Ptr activationFnc_;
};

template<typename T>
convlayer_t<T>::convlayer_t(const Model& model, const YAML::Node& layerDef) : layer_t<T>(model, layerDef)
{
    activation_ = base_type::template property<std::string>("batch", "logistic");
//    activationFnc_ = Activation::get(activation_);

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
        //batchNormalize_ = Layer::create(model, def);
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
        is.read((char*) biases_.data(), biases_.size() * sizeof(float));
        PX_CHECK(is.good(), "Could not read biases");
    }

    is.read((char*) weights_.data(), sizeof(float) * weights_.size());
    PX_CHECK(is.good(), "Could not read weights");

    return is.tellg() - start;
}

template<typename T>
void convlayer_t<T>::forward(const tensor_type& input)
{
    extern tensor_type convolve_gpu(const tensor_type& input, const tensor_t<4>& weights, int padding,
                                    int stride, int dilation);

    base_type::output_ = convolve_gpu(input, weights_, padding_, stride_, dilation_);
}

#endif // PIXIENN_CONVLAYER_T_H
