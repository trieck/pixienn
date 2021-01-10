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

#ifndef PIXIENN_LAYERFACTORY_T_H
#define PIXIENN_LAYERFACTORY_T_H

namespace px {

template<typename T>
class model_t;

template<typename T>
class layer_t;

template<typename T>
class layer_factory : public Singleton<layer_factory<T>>
{
public:
    layer_factory();

    template<typename U>
    void registerFactory(const char* type);

    using Ptr = typename layer_t<T>::Ptr;
    Ptr create(const model_t<T>& model, const YAML::Node& layerDef);

private:
    using factory = std::function<Ptr(const model_t<T>& model, const YAML::Node& layerDef)>;
    std::unordered_map<std::string, factory> factories_;
};

} // namespace px

#include "batchnormlayer_t.h"
#include "convlayer_t.h"
#include "maxpoollayer_t.h"
#include "routelayer_t.h"
#include "upsamplelayer_t.h"
#include "yololayer_t.h"

namespace px {

template<typename T>
layer_factory<T>::layer_factory()
{
    registerFactory<batchnormlayer_t<T>>("batchnorm");
    registerFactory<convlayer_t<T>>("conv");
    registerFactory<maxpoollayer_t<T>>("maxpool");
    registerFactory<routelayer_t<T>>("route");
    registerFactory<upsamplelayer_t<T>>("upsample");
    registerFactory<yololayer_t<T>>("yolo");
}

template<typename T>
template<typename U>
void layer_factory<T>::registerFactory(const char* type)
{
    factories_[type] = [](const model_t<T>& model, const YAML::Node& layerDef) {
        return Ptr(new U(model, layerDef));
    };
}

template<typename T>
auto layer_factory<T>::create(const model_t<T>& model, const YAML::Node& layerDef) -> Ptr
{
    PX_CHECK(layerDef.IsMap(), "Layer definition is not a map.");

    const auto type = layerDef["type"].as<std::string>();

    const auto it = factories_.find(type);
    if (it == std::end(factories_)) {
        PX_ERROR_THROW("Unable to find a layer factory for layer type \"%s\".", type.c_str());
    }

    return (it->second)(model, layerDef);
}

}   // namespace px

#endif // PIXIENN_LAYERFACTORY_T_H
