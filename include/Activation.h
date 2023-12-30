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

#ifndef PIXIENN_ACTIVATION_H
#define PIXIENN_ACTIVATION_H

#include "Common.h"
#include "PxTensor.h"
#include "Singleton.h"

namespace px {

template<typename T>
class Algorithm
{
public:
    using Type = T;

    virtual T apply(T x) const = 0;
    virtual T gradient(T x) const = 0;
};

template<typename T>
class IActivation
{
public:
    virtual T apply(T x) const = 0;
    virtual void apply(T* begin, T* end) const = 0;
    virtual void apply(PxCpuVectorT<T>& container) const = 0;

    virtual T gradient(T x) const = 0;
    virtual void gradient(T* dbegin, T* dend, const T* x) const = 0;
    virtual void gradient(const PxCpuVectorT<T>& container, PxCpuVectorT<T>& delta) const = 0;
};

template<typename U>
class Activation : public IActivation<typename U::Type>
{
public:
    using T = typename U::Type;

    T apply(T x) const override;
    void apply(T* begin, T* end) const override;
    void apply(PxCpuVectorT<T>& container) const override;

    T gradient(T x) const override;
    void gradient(T* dbegin, T* dend, const T* x) const override;
    void gradient(const PxCpuVectorT<T>& container, PxCpuVectorT<T>& delta) const override;

private:
    U algo_;
};

template<typename U>
auto Activation<U>::apply(T x) const -> T
{
    return algo_.apply(x);
}

template<typename U>
void Activation<U>::apply(T* begin, T* end) const
{
    std::for_each(begin, end, [this](T& x) {
        x = this->apply(x);
    });
}

template<typename U>
void Activation<U>::apply(PxCpuVectorT<T>& container) const
{
    apply(&(*container.begin()), &(*container.end()));
}

template<typename U>
auto Activation<U>::gradient(T x) const -> T
{
    return algo_.gradient(x);
}

template<typename U>
void Activation<U>::gradient(T* dbegin, T* dend, const T* x) const
{
    std::transform(dbegin, dend, x, dbegin, [this](T d, T x) {
        return d * this->gradient(x);
    });
}

template<typename U>
void Activation<U>::gradient(const PxCpuVectorT<T>& container, PxCpuVectorT<T>& delta) const
{
    gradient(&(*delta.begin()), &(*delta.end()), &(*container.begin()));
}

template<typename T>
class Linear : public Algorithm<T>
{
public:
    T apply(T x) const override
    {
        return x;
    }

    T gradient(T x) const override
    {
        return 1;
    }
};

template<typename T>
class LeakyReLU : public Algorithm<T>
{
public:
    LeakyReLU(T alpha = 0.1) : alpha_(alpha)
    {
    }

    T apply(T x) const override
    {
        return (x > 0) ? x : alpha_ * x;
    }

    T gradient(T x) const override
    {
        return (x > 0) ? 1 : alpha_;
    }
private:
    T alpha_;
};

template<typename T>
class Loggy : public Algorithm<T>
{
public:
    T apply(T x) const override
    {
        return 2. / (1. + std::exp(-x)) - 1;
    }

    T gradient(T x) const override
    {
        auto y = (x + 1.) / 2.;
        return 2 * (1 - y) * y;
    }
};

template<typename T>
class Logistic : public Algorithm<T>
{
public:
    T apply(T x) const override
    {
        return 1. / (1. + std::exp(-x));
    }

    T gradient(T x) const override
    {
        auto y = apply(x);
        return y * (1 - y);
    }
};

template<typename T>
class ReLU : public Algorithm<T>
{
public:
    T apply(T x) const override
    {
        return (x > 0) ? x : 0;
    }

    T gradient(T x) const override
    {
        return (x > 0);
    }
};

template<typename T>
class Softplus : public Algorithm<T>
{
public:
    Softplus(T threshold = 20) : threshold_(threshold)
    {}

    T apply(T x) const override
    {
        if (x > threshold_) {
            return x;
        } else if (x < -threshold_) {
            return std::exp(x);
        } else {
            return std::log(std::exp(x) + 1);
        }
    }

    T gradient(T x) const override
    {
        return 1. / (1. + std::exp(-x));
    }

private:
    T threshold_;
};

template<typename T>
class Swish : public Algorithm<T>
{
public:
    Swish(T beta = 1) : beta_(beta)
    {}

    T apply(T x) const override
    {
        return x / (1 + std::exp(-beta_ * x));
    }

    T gradient(T x) const override
    {
        auto val = 1 / (1 + std::exp(-x));
        return x * val + beta_ * val * (1 - x * val);
    }

private:
    T beta_;
};

template<typename T>
class Mish : public Algorithm<T>
{
public:
    T apply(T x) const override
    {
        auto spx = softplus_.apply(x);
        auto tanhSpx = std::tanh(spx);

        return x * tanhSpx;
    }

    T gradient(T x) const override
    {
        auto sp = softplus_.apply(x);
        auto gradSp = 1 - std::exp(-sp);
        auto tsp = std::tanh(sp);
        auto gradTsp = (1 - tsp * tsp) * gradSp;
        auto grad = x * gradTsp + tsp;
        return grad;
    }

private:
    Softplus<T> softplus_;
};

template<typename T>
class Tanh : public Algorithm<T>
{
public:
    T apply(T x) const override
    {
        return std::tanh(x);
    }

    T gradient(T x) const override
    {
        return 1 - std::pow(std::tanh(x), 2);
    }
};

class Activations : public Singleton<Activations>
{
public:
    using Type = float;
    using Ptr = std::shared_ptr<IActivation<Type>>;

    static Ptr get(const std::string& name);

    Ptr at(const std::string& s) const;
    bool hasActivation(const std::string& s) const;
};

using LeakyActivation = Activation<LeakyReLU<Activations::Type>>;
using LinearActivation = Activation<Linear<Activations::Type>>;
using LoggyActivation = Activation<Loggy<Activations::Type>>;
using LogisticActivation = Activation<Logistic<Activations::Type>>;
using MishActivation = Activation<Mish<Activations::Type>>;
using ReLUActivation = Activation<ReLU<Activations::Type>>;
using SoftplusActivation = Activation<Softplus<Activations::Type>>;
using SwishActivation = Activation<Swish<Activations::Type>>;
using TanhActivation = Activation<Tanh<Activations::Type>>;

}   // px

#endif // PIXIENN_ACTIVATION_H
