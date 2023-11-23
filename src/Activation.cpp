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
        std::for_each(begin, end, [](float& x) {});
    }
#ifdef USE_CUDA
    void applyGpu(float* x, std::size_t n) const override
    {
        linear_activate_gpu(x, n);
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
#ifdef USE_CUDA
    void applyGpu(float* x, std::size_t n) const override
    {
        leaky_activate_gpu(x, n);
    }
#endif // USE_CUDA
};

class LoggyActivation : public Activation
{
public:
    void apply(float* begin, float* end) const override
    {
        std::for_each(begin, end, [](float& x) {
            x = 2.f / (1.f + std::exp(-x)) - 1;
        });
    }
#ifdef USE_CUDA
    void applyGpu(float* x, std::size_t n) const override
    {
        loggy_activate_gpu(x, n);
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
#ifdef USE_CUDA
    void applyGpu(float* x, std::size_t n) const override
    {
        logistic_activate_gpu(x, n);
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
#ifdef USE_CUDA
    void applyGpu(float* x, std::size_t n) const override
    {
        relu_activate_gpu(x, n);
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
    apply(container.begin().base(), container.end().base());
}

#ifdef USE_CUDA

void Activation::applyGpu(PxCudaVector& container) const
{
    applyGpu(container.data(), container.size());
}

#endif

}   // px
