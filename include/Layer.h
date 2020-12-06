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

#ifndef PIXIENN_LAYER_H
#define PIXIENN_LAYER_H

#include "Error.h"
#include <xtensor/xarray.hpp>
#include <yaml-cpp/yaml.h>

namespace px {

class Model;
class LayerFactories;

class Layer
{
protected:
    Layer(const Model& model, const YAML::Node& layerDef);

public:
    virtual ~Layer() = 0;
    using Ptr = std::shared_ptr<Layer>;

    static Layer::Ptr create(const Model& model, const YAML::Node& layerDef);

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

    virtual inline std::streamoff loadDarknetWeights(std::istream& is)
    {
        return 0;
    }

    virtual void forward(const xt::xarray<float>& input) = 0;

    const xt::xarray<float>& output() const noexcept;

protected:
    const Model& model() const noexcept;

    template<typename T>
    T property(const std::string& prop) const;

    template<typename T>
    T property(const std::string& prop, const T& def) const;

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

    xt::xarray<float> output_;
private:
    const Model& model_;
    YAML::Node layerDef_;
    int batch_, channels_, height_, width_;
    int outChannels_, outHeight_, outWidth_, inputs_, index_, outputs_;
};

template<typename T>
T Layer::property(const std::string& prop) const
{
    const auto node = layerDef_[prop];

    PX_CHECK(node.IsDefined() && !node.IsNull(), "Layer has no property named \"%s\".", prop.c_str());

    return node.as<T>();
}

template<typename T>
T Layer::property(const std::string& prop, const T& def) const
{
    const auto node = layerDef_[prop];
    if (!node.IsDefined() || node.IsNull()) {
        return def;
    }

    return node.as<T>();
}

} // px

#endif // PIXIENN_LAYER_H
