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

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include "Activation.cuh"
#include "Activation.h"
#include "CudaError.h"
#include "CudaUtils.cuh"
#include "DeviceTraits.h"
#include "PxTensor.h"
#include "Singleton.h"

namespace px {

class CudaActivation
{
public:
    using Ptr = std::shared_ptr<CudaActivation>;

    virtual void apply(float* x, std::size_t n) const = 0;
    virtual void gradient(const float* x, std::size_t n, float* delta) const = 0;
};

template<typename Functor>
struct Apply
{
    Apply(Functor functor) : functor_(functor)
    {
    }

    __device__ float operator()(float x) const
    {
        return functor_.apply(x);
    }

    Functor functor_;
};

template<typename Functor>
struct Gradient
{
    Gradient(Functor functor) : functor_(functor)
    {
    }

    __device__ float operator()(const thrust::tuple<float, float>& tup) const
    {
        auto x = thrust::get<0>(tup);
        auto d = thrust::get<1>(tup);
        return d * functor_.gradient(x);
    }

    Functor functor_;
};

template<typename ActivationFunc>
class ActivationImpl : public CudaActivation
{
public:
    ActivationImpl() = default;

    void apply(float* x, std::size_t n) const override
    {
        thrust::device_ptr<float> xPtr(x);
        thrust::transform(xPtr, xPtr + n, xPtr, Apply(activation_));
    }

    void gradient(const float* x, std::size_t n, float* delta) const override
    {
        thrust::device_ptr<const float> xPtr(x);
        thrust::device_ptr<float> deltaPtr(delta);

        auto zipBegin = thrust::make_zip_iterator(thrust::make_tuple(xPtr, deltaPtr));

        thrust::transform(zipBegin, zipBegin + n, deltaPtr, Gradient(activation_));
    }

private:
    ActivationFunc activation_;
};

class ActivationFactory : public Singleton<ActivationFactory>
{
public:
    using ActivationMap = std::unordered_map<ActivationType, CudaActivation::Ptr>;

    CudaActivation::Ptr activation(ActivationType type) const
    {
        auto it = activationMap_.find(type);

        PX_CHECK(it != activationMap_.end(), "Unknown activation type: %d", type);

        return it->second;
    }

private:
    ActivationMap activationMap_ = {
            { LEAKY,    std::make_shared<ActivationImpl<Leaky>>() },
            { LINEAR,   std::make_shared<ActivationImpl<Linear>>() },
            { LOGGY,    std::make_shared<ActivationImpl<Loggy>>() },
            { LOGISTIC, std::make_shared<ActivationImpl<Logistic>>() },
            { SOFTPLUS, std::make_shared<ActivationImpl<Softplus>>() },
            { MISH,     std::make_shared<ActivationImpl<Mish>>() },
            { RELU,     std::make_shared<ActivationImpl<ReLU>>() },
            { SWISH,    std::make_shared<ActivationImpl<Swish>>() },
            { TANH,     std::make_shared<ActivationImpl<Tanh>>() }
    };
};

void activate(ActivationType type, float* x, std::size_t n)
{
    auto factory = ActivationFactory::instance();

    auto functor = factory.activation(type);

    functor->apply(x, n);
}

void gradient(ActivationType type, const float* x, std::size_t n, float* delta)
{
    auto factory = ActivationFactory::instance();

    auto functor = factory.activation(type);

    functor->gradient(x, n, delta);
}

}   // px
