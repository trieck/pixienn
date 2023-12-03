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

#ifndef PIXIENN_LAYER_H
#define PIXIENN_LAYER_H

#include <yaml-cpp/yaml.h>

#ifdef USE_CUDA

#include "Cublas.h"
#include "Cudnn.h"

#endif

#include "Error.h"
#include "PxTensor.h"

namespace px {

class Model;
class LayerFactories;

class Layer
{
protected:
    Layer(Model& model, const YAML::Node& layerDef);

public:
    virtual ~Layer() = 0;
    using Ptr = std::shared_ptr<Layer>;

    static Layer::Ptr create(Model& model, const YAML::Node& layerDef);

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
    int truth() const noexcept;
    int truths() const noexcept;
    int width() const noexcept;
    uint32_t classes() const noexcept;

    virtual std::ostream& print(std::ostream& os) = 0;

    virtual inline std::streamoff loadWeights(std::istream& is)
    {
        return 0;
    }

    virtual void forward(const PxCpuVector& input) = 0;
    virtual void backward(const PxCpuVector& input) = 0;
    const PxCpuVector& output() const noexcept;
    PxCpuVector::pointer delta() noexcept;

#ifdef USE_CUDA
    virtual void forwardGpu(const PxCudaVector& input) = 0;
    const PxCudaVector& outputGpu() const noexcept;
    const CublasContext& cublasContext() const noexcept;
    const CudnnContext& cudnnContext() const noexcept;
    bool useGpu() const;
#endif

protected:
    const Model& model() const noexcept;
    Model& model() noexcept;

    const YAML::Node& layerDef() const noexcept;
    bool hasOption(const std::string& option) const;

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
    void setTruths(int truths);
    void setCost(float cost);

    void print(std::ostream& os, const std::string& name,
               std::array<int, 3>&& input,
               std::array<int, 3>&& output,
               std::optional<int>&& filters = std::nullopt,
               std::optional<std::array<int, 3>>&& size = std::nullopt);

    bool training() const;
    bool inferring() const;

#ifdef USE_CUDA
    PxCudaVector outputGpu_;
#endif

    PxCpuVector output_, delta_;

private:
    friend LayerFactories;

    virtual void setup() = 0;

    Model& model_;
    YAML::Node layerDef_;
    int batch_, channels_, height_, width_;
    int outChannels_, outHeight_, outWidth_, inputs_, index_, outputs_;

    // training parameters
    int truth_ = 0, truths_ = 0;
    float cost_ = 0.0f;
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
