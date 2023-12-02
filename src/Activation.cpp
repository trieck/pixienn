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
    void apply(float* begin, float* end) const override
    {
    }

    void gradient(float* begin, float* end, float* delta) const override
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
    void apply(float* begin, float* end) const override
    {
        std::for_each(begin, end, [](float& x) {
            x = (x > 0) ? x : .1f * x;
        });
    }

    void gradient(float* begin, float* end, float* dbegin) const override
    {
        std::transform(begin, end, dbegin, begin, [](float x, float delta) {
            return x * ((delta > 0) ? 1.0f : 0.1f);
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
    void apply(float* begin, float* end) const override
    {
        std::for_each(begin, end, [](float& x) {
            x = 2.0f / (1.0f + std::exp(-x)) - 1;
        });
    }

    void gradient(float* begin, float* end, float* dbegin) const override
    {
        std::transform(begin, end, dbegin, begin, [](float x, float delta) {
            auto y = (delta + 1.0f) / 2.0f;
            return x * 2 * (1 - y) * y;
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
    void apply(float* begin, float* end) const override
    {
        std::for_each(begin, end, [](float& x) {
            x = 1.f / (1.f + std::exp(-x));
        });
    }

    void gradient(float* begin, float* end, float* dbegin) const override
    {
        std::transform(begin, end, dbegin, begin, [](float x, float delta) {
            return x * ((1 - delta) / delta);
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
    void apply(float* begin, float* end) const override
    {
        std::for_each(begin, end, [](float& x) {
            x = x * float(x > 0);
        });
    }

    void gradient(float* begin, float* end, float* dbegin) const override
    {
        std::transform(begin, end, dbegin, begin, [](float x, float delta) {
            return x * float(delta > 0);
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

void Activation::gradient(PxCpuVector& container, PxCpuVector& delta) const
{
    gradient(&(*container.begin()), &(*container.end()), &(*delta.begin()));
}

#ifdef USE_CUDA

void Activation::applyGpu(PxCudaVector& container) const
{
    applyGpu(container.data(), container.size());
}

#endif

}   // px
