/********************************************************************************
* Copyright 2023 Thomas A. Rieck, All Rights Reserved
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

#pragma once

#include <cblas.h>
#include <yaml-cpp/yaml.h>

#include "DeviceTraits.h"
#include "Error.h"
#include "PxTensor.h"
#include "RecordWriter.h"
#include "TrainBatch.h"

#ifdef USE_CUDA

#include "Cublas.h"
#include "Cudnn.h"

#endif

namespace px {

///////////////////////////////////////////////////////////////////////////////
template<Device D>
class Model;

///////////////////////////////////////////////////////////////////////////////
// Layer
template<Device D = Device::CPU>
class Layer
{
public:
    using V = typename Model<D>::V;
    using Ptr = std::shared_ptr<Layer<D>>;

    Layer(Model<D>& model, YAML::Node layerDef);
    virtual ~Layer() = default;

    virtual void forward(const V& input);
    virtual void backward(const V& input, V* grad);
    virtual void update();

    virtual std::ostream& print(std::ostream& os) = 0;

    virtual std::streamoff loadWeights(std::istream& is);
    virtual std::streamoff saveWeights(std::ostream& os);
    virtual bool hasCost() const noexcept;
    float cost() const noexcept;

    int batch() const noexcept;
    int channels() const noexcept;
    int height() const noexcept;
    int index() const noexcept;
    int inputs() const noexcept;
    int outChannels() const noexcept;
    int outHeight() const noexcept;
    int outWidth() const noexcept;
    int outputs() const noexcept;
    int width() const noexcept;

    const V& output() const noexcept;
    V& delta() noexcept;

    std::size_t seen() const noexcept;

protected:
    void setInputs(int inputs);
    void setChannels(int channels);
    void setHeight(int height);
    void setWidth(int width);

    void setOutputs(int outputs);
    void setOutChannels(int channels);
    void setOutHeight(int height);
    void setOutWidth(int width);

    template<typename T>
    T property(const std::string& prop) const;

    template<typename T>
    T property(const std::string& prop, const T& def) const;

    void print(std::ostream& os, const std::string& name,
               std::array<int, 3>&& input,
               std::array<int, 3>&& output,
               std::optional<int>&& filters = std::nullopt,
               std::optional<std::array<int, 3>>&& size = std::nullopt);

    const Model<D>& model() const noexcept;
    Model<D>& model() noexcept;

    bool inferring() const noexcept;
    bool training() const noexcept;
    int classes() const noexcept;

    const TrainBatch& trainingBatch() const noexcept;

    const GroundTruths& groundTruth() const noexcept;
    const GroundTruthVec& groundTruth(uint32_t batch) const noexcept;

    virtual void scaleGradients();
    virtual void clipGradients();

    void scaleTensor(V& tensor);

    RecordWriter& recordWriter() const;

    bool gradientRescaling_ = false;
    float gradientThreshold_ = 0.0f;

    bool gradientClipping_ = false;
    float gradientClipValue_ = 0.0f;

#ifdef USE_CUDA
    template<Device D_ = D, typename = EnableIfCuda<D_>>
    const CublasContext& cublasContext() const noexcept;

    template<Device D_ = D, typename = EnableIfCuda<D_>>
    const CudnnContext& cudnnContext() const noexcept;
#endif

    V output_, delta_;
    float cost_ = 0.0f;

private:
    Model<D>& model_;
    YAML::Node layerDef_;

    int batch_, channels_, height_, width_;
    int outChannels_, outHeight_, outWidth_, inputs_, index_, outputs_;
};

template<Device D>
Layer<D>::Layer(Model<D>& model, YAML::Node layerDef)
        : model_(model), layerDef_(std::move(layerDef)),
          batch_(0), channels_(0), height_(0), width_(0),
          outChannels_(0), outHeight_(0), outWidth_(0), inputs_(0), index_(0), outputs_(0)
{
    batch_ = property<int>("batch", 1);
    channels_ = property<int>("channels", 3);
    height_ = property<int>("height", 0);
    index_ = property<int>("index", 0);
    inputs_ = property<int>("inputs", 0);
    width_ = property<int>("width", 0);

    gradientRescaling_ = model.gradRescaling();
    gradientThreshold_ = model.gradThreshold();

    gradientClipping_ = model.gradClipping();
    gradientClipValue_ = model.gradClipValue();
}

template<Device D>
void Layer<D>::print(std::ostream& os, const std::string& name, std::array<int, 3>&& input, std::array<int, 3>&& output,
                     std::optional<int>&& filters, std::optional<std::array<int, 3>>&& size)
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

template<Device D>
std::streamoff Layer<D>::loadWeights(std::istream& is)
{
    return 0;
}

template<Device D>
std::streamoff Layer<D>::saveWeights(std::ostream& is)
{
    return 0;
}

template<Device D>
bool Layer<D>::hasCost() const noexcept
{
    return false;
}

template<Device D>
float Layer<D>::cost() const noexcept
{
    return cost_;
}

template<Device D>
template<typename T>
T Layer<D>::property(const std::string& prop) const
{
    const auto node = layerDef_[prop];

    PX_CHECK(node.IsDefined() && !node.IsNull(), "Layer has no property named \"%s\".", prop.c_str());

    return node.as<T>();
}

template<Device D>
template<typename T>
T Layer<D>::property(const std::string& prop, const T& def) const
{
    const auto node = layerDef_[prop];
    if (!node.IsDefined() || node.IsNull()) {
        return def;
    }

    return node.as<T>();
}

template<Device D>
int Layer<D>::batch() const noexcept
{
    return batch_;
}

template<Device D>
int Layer<D>::channels() const noexcept
{
    return channels_;
}

template<Device D>
int Layer<D>::height() const noexcept
{
    return height_;
}

template<Device D>
int Layer<D>::width() const noexcept
{
    return width_;
}

template<Device D>
int Layer<D>::outChannels() const noexcept
{
    return outChannels_;
}

template<Device D>
int Layer<D>::outHeight() const noexcept
{
    return outHeight_;
}

template<Device D>
int Layer<D>::outWidth() const noexcept
{
    return outWidth_;
}

template<Device D>
int Layer<D>::outputs() const noexcept
{
    return outputs_;
}

template<Device D>
int Layer<D>::inputs() const noexcept
{
    return inputs_;
}

template<Device D>
int Layer<D>::index() const noexcept
{
    return index_;
}

template<Device D>
void Layer<D>::setInputs(int inputs)
{
    inputs_ = inputs;
}

template<Device D>
void Layer<D>::setChannels(int channels)
{
    channels_ = channels;
}

template<Device D>
void Layer<D>::setHeight(int height)
{
    height_ = height;
}

template<Device D>
void Layer<D>::setWidth(int width)
{
    width_ = width;
}

template<Device D>
void Layer<D>::setOutputs(int outputs)
{
    outputs_ = outputs;
}

template<Device D>
void Layer<D>::setOutChannels(int channels)
{
    outChannels_ = channels;
}

template<Device D>
void Layer<D>::setOutHeight(int height)
{
    outHeight_ = height;
}

template<Device D>
void Layer<D>::setOutWidth(int width)
{
    outWidth_ = width;
}

template<Device D>
auto Layer<D>::output() const noexcept -> const V&
{
    return output_;
}

template<Device D>
void Layer<D>::forward(const V& input)
{
    delta_.fill(0);
    output_.fill(0);
    cost_ = 0;
}

template<Device D>
void Layer<D>::backward(const V& input, V* grad)
{
    if (this->gradientRescaling_) {
        this->scaleGradients();
    }

    if (this->gradientClipping_) {
        this->clipGradients();
    }
}

template<Device D>
void Layer<D>::update()
{
}

template<Device D>
const Model<D>& Layer<D>::model() const noexcept
{
    return model_;
}

template<Device D>
Model<D>& Layer<D>::model() noexcept
{
    return model_;
}

template<Device D>
bool Layer<D>::training() const noexcept
{
    return model_.training();
}

template<Device D>
bool Layer<D>::inferring() const noexcept
{
    return !model_.training();
}

template<Device D>
int Layer<D>::classes() const noexcept
{
    return model_.classes();
}

template<Device D>
auto Layer<D>::delta() noexcept -> V&
{
    return delta_;
}

template<Device D>
const TrainBatch& Layer<D>::trainingBatch() const noexcept
{
    return model_.trainingBatch();
}

template<Device D>
const GroundTruths& Layer<D>::groundTruth() const noexcept
{
    return trainingBatch().groundTruth();
}

template<Device D>
const GroundTruthVec& Layer<D>::groundTruth(uint32_t batch) const noexcept
{
    return trainingBatch().groundTruth(batch);
}

template<Device D>
void Layer<D>::scaleGradients()
{
    scaleTensor(delta_);
}

template<Device D>
void Layer<D>::scaleTensor(V& tensor)
{
    auto norm = magArray(tensor.data(), tensor.size());
    if (norm > 0 && norm > gradientThreshold_) {
        auto scale = gradientThreshold_ / norm;
        cblas_sscal(tensor.size(), scale, tensor.data(), 1);
    }
}

template<Device D>
void Layer<D>::clipGradients()
{
    constrain(delta_.size(), gradientClipValue_, delta_.data(), 1);
}

template<Device D>
std::size_t Layer<D>::seen() const noexcept
{
    return model_.seen();
}

template<Device D>
RecordWriter& Layer<D>::recordWriter() const
{
    return model_.recordWriter();
}

#ifdef USE_CUDA

template<Device D>
template<Device D_, typename>
const CudnnContext& Layer<D>::cudnnContext() const noexcept
{
    return model_.cudnnContext();
}

template<Device D>
template<Device D_, typename>
const CublasContext& Layer<D>::cublasContext() const noexcept
{
    return model_.cublasContext();
}

#endif  // USE_CUDA

}   // px

