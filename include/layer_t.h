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

#ifndef PIXIENN_LAYER_T_H
#define PIXIENN_LAYER_T_H

#include "Error.h"
#include "Singleton.h"
#include "Tensor.h"
#include <yaml-cpp/yaml.h>

#include <xtensor/xrandom.hpp>

namespace px {

class Model;

template<typename T=cpu_array>
class layer_t
{
public:
    using self_type = layer_t<T>;
    using tensor_type = T;

    virtual ~layer_t() = 0;
    using Ptr = std::shared_ptr<self_type>;

    static Ptr create(const Model& model, const YAML::Node& layerDef);

    int inputs() const;
    int index() const;
    int batch() const;
    int channels() const;
    int height() const;
    int width() const;

    int outChannels() const;
    int outHeight() const;
    int outWidth() const;
    int outputs() const;

    virtual std::ostream& print(std::ostream& os) = 0;
    virtual void forward(const tensor_type& input) = 0;
    virtual std::streamoff loadDarknetWeights(std::istream& is);

    const tensor_type& output() const noexcept;

protected:
    layer_t(const Model& model, const YAML::Node& layerDef);

    const Model& model() const noexcept;

    template<typename U>
    U property(const std::string& prop) const;

    template<typename U>
    U property(const std::string& prop, const U& def) const;

    void setInputs(int inputs);
    void setChannels(int channels);
    void setHeight(int height);
    void setWidth(int width);

    void setOutputs(int outputs);
    void setOutChannels(int channels);
    void setOutHeight(int height);
    void setOutWidth(int width);

    void print(std::ostream& os, const std::string& name,
               std::array<int, 3>&& input,
               std::array<int, 3>&& output,
               std::optional<int>&& filters = std::nullopt,
               std::optional<std::array<int, 3>>&& size = std::nullopt);

    tensor_type output_;

private:
    const Model& model_;
    YAML::Node layerDef_;
    int batch_, channels_, height_, width_;
    int outChannels_, outHeight_, outWidth_, inputs_, index_, outputs_;
};

template<typename T>
class convlayer_t;

template<typename T>
class layer_factory : public Singleton<layer_factory<T>>
{
public:
    layer_factory();

    template<typename U>
    void registerFactory(const char* type);

    template<typename U>
    void foobar()
    {

    }

    using Ptr = typename layer_t<T>::Ptr;
    Ptr create(const Model& model, const YAML::Node& layerDef);

private:
    using factory = std::function<Ptr(const Model& model, const YAML::Node& layerDef)>;
    std::unordered_map<std::string, factory> factories_;
};

#include "convlayer_t.h"

template<typename T>
layer_factory<T>::layer_factory()
{
    registerFactory<convlayer_t<T>>("conv");
}

template<typename T>
template<typename U>
void layer_factory<T>::registerFactory(const char* type)
{
    factories_[type] = [](const Model& model, const YAML::Node& layerDef) {
        return Ptr(new U(model, layerDef));
    };
}

template<typename T>
auto layer_factory<T>::create(const Model& model, const YAML::Node& layerDef) -> Ptr
{
    PX_CHECK(layerDef.IsMap(), "Layer definition is not a map.");

    const auto type = layerDef["type"].as<std::string>();

    const auto it = factories_.find(type);
    if (it == std::end(factories_)) {
        PX_ERROR_THROW("Unable to find a layer factory for layer type \"%s\".", type.c_str());
    }

    return (it->second)(model, layerDef);
}

template<typename T>
layer_t<T>::~layer_t<T>()
{
}

template<typename T>
auto layer_t<T>::create(const Model& model, const YAML::Node& layerDef) -> Ptr
{
    return layer_factory<T>::instance().create(model, layerDef);
}

template<typename T>
int layer_t<T>::inputs() const
{
    return inputs_;
}

template<typename T>
int layer_t<T>::index() const
{
    return index_;
}

template<typename T>
int layer_t<T>::batch() const
{
    return batch_;
}

template<typename T>
int layer_t<T>::channels() const
{
    return channels_;
}

template<typename T>
int layer_t<T>::height() const
{
    return height_;
}

template<typename T>
int layer_t<T>::width() const
{
    return width_;
}

template<typename T>
int layer_t<T>::outChannels() const
{
    return outChannels_;
}

template<typename T>
int layer_t<T>::outHeight() const
{
    return outHeight_;
}

template<typename T>
int layer_t<T>::outWidth() const
{
    return outWidth_;
}

template<typename T>
int layer_t<T>::outputs() const
{
    return outputs_;
}

template<typename T>
auto layer_t<T>::output() const noexcept -> const tensor_type&
{
    return output_;
}

template<typename T>
layer_t<T>::layer_t(const Model& model, const YAML::Node& layerDef) :  model_(model), layerDef_(layerDef)
{
    batch_ = property<int>("batch");
    channels_ = property<int>("channels");
    height_ = property<int>("height");
    index_ = property<int>("index");
    inputs_ = property<int>("inputs");
    width_ = property<int>("width");

    outChannels_ = outHeight_ = outWidth_ = outputs_ = 0;
}

template<typename T>
const Model& layer_t<T>::model() const noexcept
{
    return model_;
}

template<typename T>
template<typename U>
U layer_t<T>::property(const std::string& prop, const U& def) const
{
    const auto node = layerDef_[prop];
    if (!node.IsDefined() || node.IsNull()) {
        return def;
    }

    return node.as<U>();
}

template<typename T>
template<typename U>
U layer_t<T>::property(const std::string& prop) const
{
    const auto node = layerDef_[prop];

    PX_CHECK(node.IsDefined() && !node.IsNull(), "Layer has no property named \"%s\".", prop.c_str());

    return node.as<U>();
}

template<typename T>
void layer_t<T>::setInputs(int inputs)
{
    inputs_ = inputs;
}

template<typename T>
void layer_t<T>::setChannels(int channels)
{
    channels_ = channels;
}

template<typename T>
void layer_t<T>::setHeight(int height)
{
    height_ = height;
}

template<typename T>
void layer_t<T>::setWidth(int width)
{
    width_ = width;
}

template<typename T>
void layer_t<T>::setOutputs(int outputs)
{
    outputs_ = outputs;
}

template<typename T>
void layer_t<T>::setOutChannels(int channels)
{
    outChannels_ = channels;
}

template<typename T>
void layer_t<T>::setOutHeight(int height)
{
    outHeight_ = height;
}

template<typename T>
void layer_t<T>::setOutWidth(int width)
{
    outWidth_ = width;
}

template<typename T>
void layer_t<T>::print(std::ostream& os, const std::string& name, std::array<int, 3>&& input,
                       std::array<int, 3>&& output, std::optional<int>&& filters,
                       std::optional<std::array<int, 3>>&& size)
{

}

template<typename T>
std::streamoff layer_t<T>::loadDarknetWeights(std::istream& is)
{
    return 0;
}

}   // namespace px

#endif // PIXIENN_LAYER_T_H
