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

#include "BatchNormLayer.h"
#include "Common.h"
#include "ConnLayer.h"
#include "ConvLayer.h"
#include "DetectLayer.h"
#include "Error.h"
#include "Layer.h"
#include "MaxPoolLayer.h"
#include "Model.h"
#include "RouteLayer.h"
#include "Singleton.h"
#include "UpsampleLayer.h"
#include "YoloLayer.h"

namespace px {

class LayerFactories : public Singleton<LayerFactories>
{
public:
    LayerFactories();

    template<typename T>
    void registerFactory(const char* type);

    Layer::Ptr create(const Model& model, const YAML::Node& layerDef);

private:
    using LayerFactory = std::function<Layer::Ptr(const Model& model, const YAML::Node& layerDef)>;
    std::unordered_map<std::string, LayerFactory> factories_;
};

LayerFactories::LayerFactories()
{
    registerFactory<BatchNormLayer>("batchnorm");
    registerFactory<ConnLayer>("connected");
    registerFactory<ConvLayer>("conv");
    registerFactory<DetectLayer>("detection");
    registerFactory<MaxPoolLayer>("maxpool");
    registerFactory<RouteLayer>("route");
    registerFactory<UpsampleLayer>("upsample");
    registerFactory<YoloLayer>("yolo");
}

template<typename T>
void LayerFactories::registerFactory(const char* type)
{
    factories_[type] = [](const Model& model, const YAML::Node& layerDef) {
        return Layer::Ptr(new T(model, layerDef));
    };
}

Layer::Ptr LayerFactories::create(const Model& model, const YAML::Node& layerDef)
{
    PX_CHECK(layerDef.IsMap(), "Layer definition is not a map.");

    const auto type = layerDef["type"].as<std::string>();

    const auto it = factories_.find(type);
    if (it == std::end(factories_)) {
        PX_ERROR_THROW("Unable to find a layer factory for layer type \"%s\".", type.c_str());
    }

    return (it->second)(model, layerDef);
}

Layer::Layer(const Model& model, const YAML::Node& layerDef) : model_(model), layerDef_(layerDef)
{
    batch_ = property<int>("batch");
    channels_ = property<int>("channels");
    height_ = property<int>("height");
    index_ = property<int>("index");
    inputs_ = property<int>("inputs");
    width_ = property<int>("width");

    outChannels_ = outHeight_ = outWidth_ = outputs_ = 0;
}

Layer::~Layer()
{
}

Layer::Ptr Layer::create(const Model& model, const YAML::Node& layerDef)
{
    return LayerFactories::instance().create(model, layerDef);
}

const int Layer::batch() const
{
    return batch_;
}

const int Layer::channels() const
{
    return channels_;
}

const int Layer::height() const
{
    return height_;
}

const int Layer::width() const
{
    return width_;
}

const int Layer::outChannels() const
{
    return outChannels_;
}

const int Layer::outHeight() const
{
    return outHeight_;
}

const int Layer::outWidth() const
{
    return outWidth_;
}

const int Layer::outputs() const
{
    return outputs_;
}

void Layer::setOutChannels(int channels)
{
    outChannels_ = channels;
}

void Layer::setOutHeight(int height)
{
    outHeight_ = height;
}

void Layer::setOutWidth(int width)
{
    outWidth_ = width;
}

void Layer::setOutputs(int outputs)
{
    outputs_ = outputs;
}

const int Layer::inputs() const
{
    return inputs_;
}

void Layer::setInputs(int inputs)
{
    inputs_ = inputs;
}

void Layer::setChannels(int channels)
{
    channels_ = channels;
}

void Layer::setHeight(int height)
{
    height_ = height;
}

void Layer::setWidth(int width)
{
    width_ = width;
}

const xt::xarray<float>& Layer::output() const noexcept
{
    return output_;
}

const Model& Layer::model() const noexcept
{
    return model_;
}

const int Layer::index() const
{
    return index_;
}

} // px
