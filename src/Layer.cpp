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

#include "ConvLayer.h"
#include "Layer.h"
#include "MaxPoolLayer.h"
#include "Singleton.h"
#include "common.h"
#include <Error.h>

PX_BEGIN

class LayerFactories : public Singleton<LayerFactories>
{
public:
    LayerFactories();

    template<typename T>
    void registerFactory(const char* type);

    Layer::Ptr create(const YAML::Node& layerDef);

private:
    using LayerFactory = std::function<Layer::Ptr(const YAML::Node& layerDef)>;
    std::unordered_map<std::string, LayerFactory> factories_;
};

LayerFactories::LayerFactories()
{
    registerFactory<ConvLayer>("conv");
    registerFactory<MaxPoolLayer>("maxpool");
}

template<typename T>
void LayerFactories::registerFactory(const char* type)
{
    factories_[type] = [](const YAML::Node& layerDef) {
        return Layer::Ptr(new T(layerDef));
    };
}

Layer::Ptr LayerFactories::create(const YAML::Node& layerDef)
{
    PX_CHECK(layerDef.IsMap(), "Layer definition is not a map.");

    const auto type = layerDef["type"].as<std::string>();

    const auto it = factories_.find(type);
    if (it == std::end(factories_)) {
        PX_ERROR_THROW("Unable to find a layer factory for layer type \"%s\".", type.c_str());
    }

    return (it->second)(layerDef);
}

Layer::Layer(const YAML::Node& layerDef) : layerDef_(layerDef)
{
    batch_ = property<int>("batch");
    channels_ = property<int>("channels");
    height_ = property<int>("height");
    width_ = property<int>("width");

    outChannels_ = outHeight_ = outWidth_ = 0;
}

Layer::~Layer()
{
}

Layer::Ptr Layer::create(const YAML::Node& layerDef)
{
    return LayerFactories::instance().create(layerDef);
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

PX_END
