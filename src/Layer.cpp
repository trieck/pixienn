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

#include <cblas.h>
#include "BatchNormLayer.h"
#include "Common.h"
#include "ConnLayer.h"
#include "ConvLayer.h"
#include "DetectLayer.h"
#include "Error.h"
#include "Layer.h"
#include "MaxPoolLayer.h"
#include "Model.h"
#include "RegionLayer.h"
#include "RouteLayer.h"
#include "ShortcutLayer.h"
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

    Layer::Ptr create(Model& model, const YAML::Node& layerDef);

private:
    using LayerFactory = std::function<Layer::Ptr(Model& model, const YAML::Node& layerDef)>;
    std::unordered_map<std::string, LayerFactory> factories_;
};

LayerFactories::LayerFactories()
{
    registerFactory<BatchNormLayer>("batchnorm");
    registerFactory<ConnLayer>("connected");
    registerFactory<ConvLayer>("conv");
    registerFactory<DetectLayer>("detection");
    registerFactory<MaxPoolLayer>("maxpool");
    registerFactory<RegionLayer>("region");
    registerFactory<RouteLayer>("route");
    registerFactory<ShortcutLayer>("shortcut");
    registerFactory<UpsampleLayer>("upsample");
    registerFactory<YoloLayer>("yolo");
}

template<typename T>
void LayerFactories::registerFactory(const char* type)
{
    factories_[type] = [](Model& model, const YAML::Node& layerDef) {
        return Layer::Ptr(new T(model, layerDef));
    };
}

Layer::Ptr LayerFactories::create(Model& model, const YAML::Node& layerDef)
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

Layer::Layer(Model& model, const YAML::Node& layerDef) : model_(model), layerDef_(layerDef)
{
    batch_ = property<int>("batch");
    channels_ = property<int>("channels");
    height_ = property<int>("height");
    index_ = property<int>("index");
    inputs_ = property<int>("inputs");
    width_ = property<int>("width");

    gradientRescaling_ = model.gradRescaling();
    gradientThreshold_ = model.gradThreshold();

    gradientClipping_ = model.gradClipping();
    gradientClipValue_ = model.gradClipValue();

    outChannels_ = outHeight_ = outWidth_ = outputs_ = 0;
}

Layer::~Layer() = default;

Layer::Ptr Layer::create(Model& model, const YAML::Node& layerDef)
{
    return LayerFactories::instance().create(model, layerDef);
}

int Layer::batch() const noexcept
{
    return batch_;
}

int Layer::channels() const noexcept
{
    return channels_;
}

int Layer::height() const noexcept
{
    return height_;
}

int Layer::width() const noexcept
{
    return width_;
}

int Layer::outChannels() const noexcept
{
    return outChannels_;
}

int Layer::outHeight() const noexcept
{
    return outHeight_;
}

int Layer::outWidth() const noexcept
{
    return outWidth_;
}

int Layer::outputs() const noexcept
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

int Layer::inputs() const noexcept
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

PxCpuVector* Layer::delta() noexcept
{
    return &delta_;
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

Model& Layer::model() noexcept
{
    return model_;
}

const YAML::Node& Layer::layerDef() const noexcept
{
    return layerDef_;
}

int Layer::index() const noexcept
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

bool Layer::hasOption(const std::string& option) const
{
    return model_.hasOption(option);
}

bool Layer::training() const
{
    return model_.training();
}

bool Layer::inferring() const
{
    return model_.inferring();
}

float Layer::cost() const noexcept
{
    return cost_;
}

uint32_t Layer::classes() const noexcept
{
    return model_.classes();
}

const TrainBatch& Layer::trainingBatch() const noexcept
{
    return model_.trainingBatch();
}

std::streamoff Layer::loadWeights(std::istream& is)
{
    return 0;
}

std::streamoff Layer::saveWeights(std::ostream& is)
{
    return 0;
}

bool Layer::hasCost() const noexcept
{
    return false;
}

void Layer::update()
{
}

void Layer::forward(const PxCpuVector& input)
{
    delta_.fill(0);
    output_.fill(0);
    cost_ = 0;
}

void Layer::backward(const PxCpuVector& input)
{
    if (gradientRescaling_) {
        scaleGradients();
    }

    if (gradientClipping_) {
        clipGradients();
    }
}

void Layer::scaleGradients()
{
    auto norm = magArray(delta_.data(), delta_.size());
    if (norm > gradientThreshold_) {
        float scale = gradientThreshold_ / norm;
        cblas_sscal(delta_.size(), scale, delta_.data(), 1);
    }
}

const GroundTruths& Layer::groundTruth() const noexcept
{
    return trainingBatch().groundTruth();
}

const GroundTruthVec& Layer::groundTruth(uint32_t batch)
{
    return trainingBatch().groundTruth(batch);
}

void Layer::clipGradients()
{
    constrain(delta_.size(), gradientClipValue_, delta_.data(), 1);
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
