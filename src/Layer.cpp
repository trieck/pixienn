/********************************************************************************
* Copyright 2020-2023 Thomas A. Rieck, All Rights Reserved
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

#include "Common.h"

#include "BatchNormLayer.h"
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

    auto ptr = (it->second)(model, layerDef);

    ptr->setup();

    return ptr;
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

Layer::~Layer() = default;

Layer::Ptr Layer::create(const Model& model, const YAML::Node& layerDef)
{
    return LayerFactories::instance().create(model, layerDef);
}

int Layer::batch() const
{
    return batch_;
}

int Layer::channels() const
{
    return channels_;
}

int Layer::height() const
{
    return height_;
}

int Layer::width() const
{
    return width_;
}

int Layer::outChannels() const
{
    return outChannels_;
}

int Layer::outHeight() const
{
    return outHeight_;
}

int Layer::outWidth() const
{
    return outWidth_;
}

int Layer::outputs() const
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

int Layer::inputs() const
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

const PxCpuVector& Layer::output() const noexcept
{
    return output_;
}

#ifdef USE_CUDA
const PxCudaVector& Layer::outputGpu() const noexcept
{
    return outputGpu_;
}
#endif // USE_CUDA

const Model& Layer::model() const noexcept
{
    return model_;
}

const YAML::Node& Layer::layerDef() const noexcept
{
    return layerDef_;
}

int Layer::index() const
{
    return index_;
}

void Layer::print(std::ostream& os, const std::string& name, std::array<int, 3>&& input,
                  std::array<int, 3>&& output, std::optional<int>&& filters,
                  std::optional<std::array<int, 3>>&& size)
{
    std::cout << std::setfill(' ') << std::setw(5) << std::right << index() << ' ';

    os << std::setfill('.');

    if (filters.has_value()) {
        os << std::setw(15) << std::left << name;
        os << std::setw(10) << std::left << filters.value();
    } else {
        os << std::setw(25) << std::left << name;
    }

    if (size.has_value()) {
        const auto& value = size.value();
        os << std::setw(20) << std::left << std::string(
                std::to_string(value[0]) + " x " + std::to_string(value[1]) + " / " + std::to_string(value[2]));
    } else {
        os << std::setw(20) << std::left << "";
    }

    os << std::setw(20) << std::left << std::string(
            std::to_string(input[0]) + " x " + std::to_string(input[1]) + " x " + std::to_string(input[2]));

    os << std::setw(20) << std::left << std::string(
            std::to_string(output[0]) + " x " + std::to_string(output[1]) + " x " + std::to_string(output[2]));

    os << std::endl << std::flush;
}

#ifdef USE_CUDA

const CublasContext& Layer::cublasContext() const noexcept
{
    return model_.cublasContext();
}

const CudnnContext& Layer::cudnnContext() const noexcept
{
    return model_.cudnnContext();
}

bool Layer::useGpu() const
{
    return model_.useGpu();
}

#endif // USE_CUDA

} // px
