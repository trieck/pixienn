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

#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include "Activation.cuh"

#endif

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

#include "ActivationTypes.h"
#include "Common.h"
#include "DeviceTraits.h"
#include "PxTensor.h"
#include "Singleton.h"

namespace px {

class Leaky
{
public:
    static constexpr ActivationType type = ActivationType::LEAKY;

    Leaky(float alpha = 0.1f) : alpha_(alpha)
    {
    }

    CUDA_CALLABLE float apply(float x) const
    {
        return x > 0.0f ? x : alpha_ * x;
    }

    CUDA_CALLABLE float gradient(float x) const
    {
        return x > 0.0f ? 1.0f : alpha_;
    }
private:
    float alpha_;
};

class Linear
{
public:
    static constexpr ActivationType type = ActivationType::LINEAR;

    CUDA_CALLABLE float apply(float x) const
    {
        return x;
    }

    CUDA_CALLABLE float gradient(float x) const
    {
        return 1.0f;
    }
};

class Loggy
{
public:
    static constexpr ActivationType type = ActivationType::LOGGY;

    CUDA_CALLABLE float apply(float x) const
    {
        return 2.0f / (1.0f + std::exp(-x)) - 1.0f;
    }

    CUDA_CALLABLE float gradient(float x) const
    {
        auto y = (x + 1.0f) / 2.0f;
        return 2.0f * (1 - y) * y;
    }
};

class Logistic
{
public:
    static constexpr ActivationType type = ActivationType::LOGISTIC;

    CUDA_CALLABLE float apply(float x) const
    {
        return 1.0f / (1.0f + std::exp(-x));
    }

    CUDA_CALLABLE float gradient(float x) const
    {
        auto y = apply(x);
        return y * (1.0f - y);
    }
};

class Softplus
{
public:
    static constexpr ActivationType type = ActivationType::SOFTPLUS;

    Softplus(float threshold = 20) : threshold_(threshold)
    {}

    CUDA_CALLABLE float apply(float x) const
    {
        if (x > threshold_) {
            return x;
        } else if (x < -threshold_) {
            return std::exp(x);
        } else {
            return std::log(std::exp(x) + 1);
        }
    }

    CUDA_CALLABLE float gradient(float x) const
    {
        return 1. / (1. + std::exp(-x));
    }

private:
    float threshold_ = 20.0f;
};

class Mish
{
public:
    static constexpr ActivationType type = ActivationType::MISH;

    CUDA_CALLABLE float apply(float x) const
    {
        auto spx = softplus_.apply(x);
        auto tanhSpx = std::tanh(spx);

        return x * tanhSpx;
    }

    CUDA_CALLABLE float gradient(float x) const
    {
        auto sp = softplus_.apply(x);
        auto gradSp = 1 - std::exp(-sp);
        auto tsp = std::tanh(sp);
        auto gradTsp = (1 - tsp * tsp) * gradSp;
        auto grad = x * gradTsp + tsp;
        return grad;
    }

private:
    Softplus softplus_;
};

class ReLU
{
public:
    static constexpr ActivationType type = ActivationType::RELU;

    CUDA_CALLABLE float apply(float x) const
    {
        return x > 0.0f ? x : 0.0f;
    }

    CUDA_CALLABLE float gradient(float x) const
    {
        return x > 0.0f ? 1.0f : 0.0f;
    }
};

class Swish
{
public:
    static constexpr ActivationType type = ActivationType::SWISH;

    Swish(float beta = 1.0f) : beta_(beta)
    {}

    CUDA_CALLABLE float apply(float x) const
    {
        return x / (1 + std::exp(-beta_ * x));
    }

    CUDA_CALLABLE float gradient(float x) const
    {
        auto val = 1 / (1 + std::exp(-x));
        return x * val + beta_ * val * (1 - x * val);
    }

private:
    float beta_;
};

class Tanh
{
public:
    static constexpr ActivationType type = ActivationType::TANH;

    CUDA_CALLABLE float apply(float x) const
    {
        return std::tanh(x);
    }

    CUDA_CALLABLE float gradient(float x) const
    {
        auto y = apply(x);
        return 1.0f - y * y;
    }
};

template<Device D = Device::CPU>
class IActivation;

template<>
class IActivation<Device::CPU>
{
public:
    using V = typename DeviceTraits<Device::CPU>::VectorType;
    using T = typename DeviceTraits<Device::CPU>::ValueType;

    virtual T apply(T x) const = 0;
    virtual void apply(T* begin, T* end) const = 0;
    virtual void apply(V& container) const = 0;

    virtual T gradient(T x) const = 0;
    virtual void gradient(T* dbegin, T* dend, const T* x) const = 0;
    virtual void gradient(const V& container, V& delta) const = 0;
};


#ifdef USE_CUDA
template<>
class IActivation<Device::CUDA>
{
public:
    using V = typename DeviceTraits<Device::CUDA>::VectorType;
    using T = typename DeviceTraits<Device::CUDA>::ValueType;

    virtual void apply(T* begin, T* end) const = 0;
    virtual void apply(V& container) const = 0;

    virtual void gradient(const V& container, V& delta) const = 0;
};

#endif // USE_CUDA


template<typename U, Device D = Device::CPU>
class Activation : public IActivation<D>
{
public:
    using V = typename DeviceTraits<D>::VectorType;
    using T = typename DeviceTraits<D>::ValueType;

    T apply(T x) const override;
    void apply(T* begin, T* end) const override;
    void apply(V& container) const override;

    T gradient(T x) const override;
    void gradient(T* dbegin, T* dend, const T* x) const override;
    void gradient(const V& container, V& delta) const override;

private:
    U algo_;
};

template<typename U, Device D>
auto Activation<U, D>::apply(T x) const -> T
{
    return algo_.apply(x);
}

template<typename U, Device D>
void Activation<U, D>::apply(T* begin, T* end) const
{
    std::for_each(begin, end, [this](T& x) {
        x = this->apply(x);
    });
}

template<typename U, Device D>
void Activation<U, D>::apply(V& container) const
{
    apply(&(*container.begin()), &(*container.end()));
}

template<typename U, Device D>
auto Activation<U, D>::gradient(T x) const -> T
{
    return algo_.gradient(x);
}

template<typename U, Device D>
void Activation<U, D>::gradient(T* dbegin, T* dend, const T* x) const
{
    std::transform(dbegin, dend, x, dbegin, [this](T d, T x) {
        return d * this->gradient(x);
    });
}

template<typename U, Device D>
void Activation<U, D>::gradient(const V& container, V& delta) const
{
    gradient(&(*delta.begin()), &(*delta.end()), &(*container.begin()));
}

#ifdef USE_CUDA

template<typename U>
class Activation<U, Device::CUDA> : public IActivation<Device::CUDA>
{
public:
    using V = typename DeviceTraits<Device::CUDA>::VectorType;
    using T = typename DeviceTraits<Device::CUDA>::ValueType;

    void apply(T* begin, T* end) const override;
    void apply(V& container) const override;
    void gradient(const V& container, V& delta) const override;

private:
    U algo_;
};

template<typename U>
void Activation<U, Device::CUDA>::gradient(const PxCudaVectorT<float>& container, PxCudaVectorT<float>& delta) const
{
    px::gradient(U::type, container.data(), container.size(), delta.data());
}

template<typename U>
void Activation<U, Device::CUDA>::apply(T* begin, T* end) const
{
    px::activate(U::type, begin, end - begin);
}

template<typename U>
void Activation<U, Device::CUDA>::apply(V& container) const
{
    px::activate(U::type, container.data(), container.size());
}

#endif // USE_CUDA

template<Device D = Device::CPU>
using LeakyActivation = Activation<Leaky, D>;

template<Device D = Device::CPU>
using LinearActivation = Activation<Linear, D>;

template<Device D = Device::CPU>
using LoggyActivation = Activation<Loggy, D>;

template<Device D = Device::CPU>
using LogisticActivation = Activation<Logistic, D>;

template<Device D = Device::CPU>
using ReLUActivation = Activation<ReLU, D>;

template<Device D = Device::CPU>
using SoftplusActivation = Activation<Softplus, D>;

template<Device D = Device::CPU>
using SwishActivation = Activation<Swish, D>;

template<Device D = Device::CPU>
using TanhActivation = Activation<Tanh, D>;

template<Device D = Device::CPU>
using MishActivation = Activation<Mish, D>;

template<Device D = Device::CPU>
class Activations : public Singleton<Activations<D>>
{
public:
    using T = typename DeviceTraits<D>::ValueType;
    using Ptr = std::shared_ptr<IActivation<D>>;

    static Ptr get(const std::string& name);

    Ptr at(const std::string& s) const;
    bool hasActivation(const std::string& s) const;

private:
    const std::unordered_map<std::string, Ptr> activations_ = {
            { "leaky",    std::make_shared<LeakyActivation<D>>() },
            { "linear",   std::make_shared<LinearActivation<D>>() },
            { "loggy",    std::make_shared<LoggyActivation<D>>() },
            { "logistic", std::make_shared<LogisticActivation<D>>() },
            { "mish",     std::make_shared<MishActivation<D>>() },
            { "relu",     std::make_shared<ReLUActivation<D>>() },
            { "softplus", std::make_shared<SoftplusActivation<D>>() },
            { "swish",    std::make_shared<SwishActivation<D>>() },
            { "tanh",     std::make_shared<TanhActivation<D>>() }
    };
};

template<Device D>
bool Activations<D>::hasActivation(const std::string& s) const
{
    return activations_.find(s) != activations_.end();
}

template<Device D>
Activations<D>::Ptr Activations<D>::at(const std::string& s) const
{
    return activations_.at(s);
}

template<Device D>
Activations<D>::Ptr Activations<D>::get(const std::string& name)
{
    const auto& instance = Activations<D>::instance();

    PX_CHECK(instance.hasActivation(name), "Cannot find activation type \"%s\".", name.c_str());

    return instance.at(name);
}

}   // px
