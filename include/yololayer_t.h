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

#ifndef PIXIENN_YOLOLAYER_T_H
#define PIXIENN_YOLOLAYER_T_H

namespace px {

template<typename T = cpu_array>
class yololayer_t : public layer_t<T>
{
public:
    using base_type = layer_t<T>;
    using value_type = typename T::value_type;
    using tensor_type = typename layer_t<T>::tensor_type;

    virtual ~yololayer_t() = default;

    std::ostream& print(std::ostream& os) override;

    void forward(const tensor_type& input) override;

protected:
    yololayer_t(const model_t <T>& model, const YAML::Node& layerDef);

private:
    friend layer_factory<T>;

};

template<typename T>
std::ostream& yololayer_t<T>::print(std::ostream& os)
{
    return os;
}

template<typename T>
void yololayer_t<T>::forward(const yololayer_t::tensor_type& input)
{
    // FIXME:
}

template<typename T>
yololayer_t<T>::yololayer_t(const model_t <T>& model, const YAML::Node& layerDef) : layer_t<T>(model, layerDef)
{

}

}   // namespace px

#endif // PIXIENN_YOLOLAYER_T_H
