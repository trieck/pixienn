#pragma once

#include <yaml-cpp/yaml.h>

#include "Error.h"

#ifdef USE_CUDA

#include "Cublas.h"
#include "DeviceTraits.h"
#include "Cudnn.h"
#include "PxTensor.h"

#endif  // USE_CUDA

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
    virtual void backward(const V& input);
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

    Model<D>& model() noexcept;

    bool inferring() const noexcept;
    bool training() const noexcept;
    int classes() const noexcept;

    template<Device D_ = D, typename = EnableIfCuda<D_>>
    const CublasContext& cublasContext() const noexcept;

    template<Device D_ = D, typename = EnableIfCuda<D_>>
    const CudnnContext& cudnnContext() const noexcept;

    V output_, delta_;
private:
    Model<D>& model_;
    YAML::Node layerDef_;

    float cost_ = 0.0f;

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
}

template<Device D>
void Layer<D>::backward(const V& input)
{
}

template<Device D>
void Layer<D>::update()
{
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

}   // px

