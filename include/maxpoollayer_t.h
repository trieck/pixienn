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

#ifndef PIXIENN_MAXPOOLLAYER_T_H
#define PIXIENN_MAXPOOLLAYER_T_H

namespace px {

template<typename T = cpu_array>
class maxpoollayer_t : public layer_t<T>
{
public:
    using base_type = layer_t<T>;
    using value_type = typename T::value_type;
    using tensor_type = typename layer_t<T>::tensor_type;

    virtual ~maxpoollayer_t() = default;

    std::ostream& print(std::ostream& os) override;

    void forward(const tensor_type& input) override;

protected:
    maxpoollayer_t(const model_t <T>& model, const YAML::Node& layerDef);

private:
    friend layer_factory<T>;

    int kernel_ = 0, stride_ = 0, padding_;
};

template<typename T>
std::ostream& maxpoollayer_t<T>::print(std::ostream& os)
{
    return os;
}

template<typename T>
void maxpoollayer_t<T>::forward(const maxpoollayer_t::tensor_type& input)
{
    // FIXME:
}

template<typename T>
maxpoollayer_t<T>::maxpoollayer_t(const model_t <T>& model, const YAML::Node& layerDef) : layer_t<T>(model, layerDef)
{
    kernel_ = base_type::template property<int>("kernel", 1);
    stride_ = base_type::template property<int>("stride", 1);
    padding_ = base_type::template property<int>("padding", std::max(0, kernel_ - 1));

    base_type::setOutChannels(base_type::channels());
    base_type::setOutHeight((base_type::height() + padding_ - kernel_) / stride_ + 1);
    base_type::setOutWidth((base_type::width() + padding_ - kernel_) / stride_ + 1);

    base_type::setOutputs(base_type::outHeight() * base_type::outWidth() * base_type::outChannels());

    base_type::output_ = xt::empty<value_type>({ base_type::batch(), base_type::outChannels(), base_type::outHeight(),
                                                 base_type::outWidth() });
}

}   // namespace px

#endif // PIXIENN_MAXPOOLLAYER_T_H
