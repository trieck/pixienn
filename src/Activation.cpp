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

#include <cmath>

#include "Activation.h"
#include "Singleton.h"
#include "Error.h"

#ifdef USE_CUDA

#include "ActivationKernels.cuh"

#endif // USE_CUDA

namespace px {

using namespace px;

class Activations : public Singleton<Activations>
{
public:
    Activations();

    bool hasActivation(const std::string& s) const;
    Activation::Ptr at(const std::string& s) const;

private:
    std::unordered_map<std::string, Activation::Ptr> activations_;
};

class LinearActivation : public Activation
{
public:
    float apply(float x) const override
    {
        return x;
    }

    float gradient(float x) const override
    {
        return x;
    }

    void apply(float* begin, float* end) const override
    {
    }

    void gradient(float *dbegin, float* dend, const float* x) const override
    {
    }

#ifdef USE_CUDA
    void applyGpu(float* x, std::size_t n) const override
    {
        linearActivateGpu(x, n);
    }
#endif
};

class LeakyActivation : public Activation
{
public:
    float apply(float x) const override
    {
        return (x > 0) ? x : .1f * x;
    }

    float gradient(float x) const override
    {
        return (x > 0) ? 1.0f : 0.1f;
    }

    void apply(float* begin, float* end) const override
    {
        std::for_each(begin, end, [this](float& x) {
            x = apply(x);
        });
    }

    void gradient(float *dbegin, float* dend, const float* x) const override
    {
        std::transform(dbegin, dend, x, dbegin, [this](float delta, float x) {
            return delta * gradient(x);
        });
    }

#ifdef USE_CUDA
    void applyGpu(float* x, std::size_t n) const override
    {
        leakyActivateGpu(x, n);
    }
#endif // USE_CUDA
};

class LoggyActivation : public Activation
{
public:
    float apply(float x) const override
    {
        return 2.0f / (1.0f + std::exp(-x)) - 1;
    }

    float gradient(float x) const override
    {
        auto y = (x + 1.0f) / 2.0f;
        return 2 * (1 - y) * y;
    }

    void apply(float* begin, float* end) const override
    {
        std::for_each(begin, end, [this](float& x) {
            x = apply(x);
        });
    }

    void gradient(float *dbegin, float* dend, const float* x) const override
    {
        std::transform(dbegin, dend, x, dbegin, [this](float delta, float x) {
            return delta * gradient(x);
        });
    }

#ifdef USE_CUDA
    void applyGpu(float* x, std::size_t n) const override
    {
        loggyActivateGpu(x, n);
    }
#endif // USE_CUDA
};

class LogisticActivation : public Activation
{
public:
    float apply(float x) const override
    {
        return 1.f / (1.f + std::exp(-x));
    }

    float gradient(float x) const override
    {
        return (1 - x) * x;
    }

    void apply(float* begin, float* end) const override
    {
        std::for_each(begin, end, [this](float& x) {
            x = apply(x);
        });
    }

    void gradient(float* dbegin, float* dend, const float* x) const override
    {
        std::transform(dbegin, dend, x, dbegin, [this](float delta, float x) {
            return delta * gradient(x);
        });
    }

#ifdef USE_CUDA
    void applyGpu(float* x, std::size_t n) const override
    {
        logisticActivateGpu(x, n);
    }
#endif // USE_CUDA
};

class ReluActivation : public Activation
{
public:
    float apply(float x) const override
    {
        return x * float(x > 0);
    }

    float gradient(float x) const override
    {
        return float(x > 0);
    }

    void apply(float* begin, float* end) const override
    {
        std::for_each(begin, end, [this](float& x) {
            x = apply(x);
        });
    }

    void gradient(float* dbegin, float* dend, const float* x) const override
    {
        std::transform(dbegin, dend, x, dbegin, [this](float delta, float x) {
            return delta * gradient(x);
        });
    }

#ifdef USE_CUDA
    void applyGpu(float* x, std::size_t n) const override
    {
        reluActivateGpu(x, n);
    }
#endif
};

Activations::Activations()
{
    activations_ = {
            { "leaky",    std::make_shared<LeakyActivation>() },
            { "linear",   std::make_shared<LinearActivation>() },
            { "loggy",    std::make_shared<LoggyActivation>() },
            { "logistic", std::make_shared<LogisticActivation>() },
            { "relu",     std::make_shared<ReluActivation>() },
    };
}

bool Activations::hasActivation(const std::string& s) const
{
    return activations_.find(s) != activations_.end();
}

Activation::Ptr Activations::at(const std::string& s) const
{
    return activations_.at(s);
}

Activation::Ptr Activation::get(const std::string& s)
{
    auto& activations = Activations::instance();

    PX_CHECK(activations.hasActivation(s), "Cannot find activation type \"%s\".", s.c_str());

    return activations.at(s);
}

void Activation::apply(PxCpuVector& container) const
{
    apply(&(*container.begin()), &(*container.end()));
}

void Activation::gradient(const PxCpuVector& container, PxCpuVector& delta) const
{
    gradient(&(*delta.begin()), &(*delta.end()), &(*container.begin()));
}

float Activation::operator()(float x) const
{
    return apply(x);
}

#ifdef USE_CUDA

void Activation::applyGpu(PxCudaVector& container) const
{
    applyGpu(container.data(), container.size());
}

#endif

}   // px
