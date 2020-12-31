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

#ifndef PIXIENN_ACTIVATION_T_H
#define PIXIENN_ACTIVATION_T_H

#include "Error.h"
#include "Singleton.h"

#include <thrust/for_each.h>

namespace px {

template<typename T=cpu_array>
class activation_t
{
public:
    using Ptr = std::shared_ptr<activation_t<T>>;

    static activation_t<T>::Ptr get(const std::string& s);

    using It = typename T::iterator;

    virtual void apply(It begin, It end) const = 0;

    void apply(T&) const;

    virtual __host__ __device__ void operator()(typename T::value_type& x) const = 0;
};

template<typename T=cpu_array>
class activations_t : public Singleton<activations_t<T>>
{
public:
    activations_t();

    bool hasActivation(const std::string& s) const;

    typename activation_t<T>::Ptr at(const std::string& s) const;

private:
    std::unordered_map<std::string, typename activation_t<T>::Ptr> activations_;
};

template<typename T=cpu_array>
class leaky_activation_t : public activation_t<T>
{
public:
    using It = typename T::iterator;

    __host__ __device__ void operator()(typename T::value_type& x) const override
    {
        x = (x > 0) ? x : .1f * x;
    };

    void apply(It begin, It end) const override
    {
        thrust::for_each(begin, end, *this);
    }
};

template<typename T=cpu_array>
class linear_activation_t : public activation_t<T>
{
public:
    using It = typename T::iterator;

    __host__ __device__ void operator()(typename T::value_type& x) const override
    {

    }

    void apply(It begin, It end) const override
    {
    }
};

template<typename T=cpu_array>
class loggy_activation_t : public activation_t<T>
{
public:
    using It = typename T::iterator;

    __host__ __device__ void operator()(typename T::value_type& x) const override
    {
        x = 2.f / (1.f + exp(-x)) - 1;
    }

    void apply(It begin, It end) const override
    {
        thrust::for_each(begin, end, *this);
    }
};

template<typename T=cpu_array>
class logistic_activation_t : public activation_t<T>
{
public:
    using It = typename T::iterator;

    __host__ __device__ void operator()(typename T::value_type& x) const override
    {
        x = 1.f / (1.f + std::exp(-x));
    }

    void apply(It begin, It end) const override
    {
        thrust::for_each(begin, end, *this);
    }
};

template<typename T=cpu_array>
class relu_activation_t : public activation_t<T>
{
public:
    using It = typename T::iterator;

    __host__ __device__ void operator()(typename T::value_type& x) const override
    {
        x = x * (x > 0);
    }

    void apply(It begin, It end) const override
    {
        thrust::for_each(begin, end, *this);
    }
};

template<typename T>
activations_t<T>::activations_t()
{
    activations_ = {
            { "leaky",    std::make_shared<leaky_activation_t<T>>() },
            { "linear",   std::make_shared<linear_activation_t<T>>() },
            { "loggy",    std::make_shared<loggy_activation_t<T>>() },
            { "logistic", std::make_shared<logistic_activation_t<T>>() },
            { "relu",     std::make_shared<relu_activation_t<T>>() },
    };
}

template<typename T>
bool activations_t<T>::hasActivation(const std::string& s) const
{
    return activations_.find(s) != activations_.end();
}

template<typename T>
auto activations_t<T>::at(const std::string& s) const -> typename activation_t<T>::Ptr
{
    return activations_.at(s);
}

template<typename T>
auto activation_t<T>::get(const std::string& s) -> Ptr
{
    auto& activations = activations_t<T>::instance();

    PX_CHECK(activations.hasActivation(s), "Cannot find activation type \"%s\".", s.c_str());

    return activations.at(s);
}

template<typename T>
void activation_t<T>::apply(T& container) const
{
    apply(std::begin(container), std::end(container));
}

}   // namespace px

#endif // PIXIENN_ACTIVATION_T_H
