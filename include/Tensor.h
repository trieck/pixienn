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

#ifndef PIXIENN_TENSOR_H
#define PIXIENN_TENSOR_H

#include "Common.h"
#include <random>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xlayout.hpp>
#include <xtensor/xtensor.hpp>

namespace px {

template<typename T>
using xt_cpu_array = xt::xarray_container<thrust::host_vector<T>>;

template<typename T, std::size_t N>
using xt_cpu_tensor = xt::xtensor_container<thrust::host_vector<T>, N>;

template<typename T>
using xt_cuda_array = xt::xarray_container<thrust::device_vector<T>>;

template<typename T, std::size_t N>
using xt_cuda_tensor = xt::xtensor_container<thrust::device_vector<T>, N>;

template<typename T, typename D = thrust::random::uniform_real_distribution<T>, typename E = thrust::default_random_engine>
struct random_functor
{
    D dist_;
    E rng_;

    __host__ __device__

    random_functor(std::uint32_t seed, T lower = 0, T upper = 1) : rng_(seed)
    {
        dist_ = D(lower, upper);
    }

    __host__ __device__

    T operator()(int n)
    {
        rng_.discard(n);
        return dist_(rng_);
    }
};

enum class Device
{
    CPU, CUDA
};

template<typename T = float, typename Base = xt_cpu_array<T>>
class tensor : public Base
{
public:
    using self_type = tensor<T, Base>;
    using base_type = Base;
    using storage_type = typename base_type::storage_type;
    using array_type = tensor<T, xt::xarray_container<storage_type>>;

    template<std::size_t N = 1>
    using tensor_type = tensor<T, xt::xtensor_container<storage_type, N>>;

    using pointer = typename Base::pointer;
    using const_pointer = typename Base::const_pointer;

    tensor() = default;
    tensor(const tensor& rhs) = default;
    tensor(tensor&& rhs) = default;
    tensor(const Base& rhs);
    tensor(Base&& rhs);

    template<typename DC>
    tensor(const xt::xexpression<DC>& rhs);

    tensor& operator=(tensor&& rhs) = default;
    tensor& operator=(const tensor& rhs) = default;
    tensor& operator=(Base&& rhs);
    tensor& operator=(const Base& rhs);

    template<typename DC>
    tensor& operator=(const xt::xexpression<DC>& rhs);

    const_pointer data() const noexcept;
    pointer data() noexcept;

    template<typename I, std::size_t N>
    static tensor random(const I (& shape)[N], T lower = 0, T upper = 1);

    template<typename I, std::size_t N>
    static tensor fill(const I (& shape)[N], T value);
};

template<typename T, typename Base>
tensor<T, Base>::tensor(const Base& rhs) : Base(rhs)
{
}

template<typename T, typename Base>
tensor<T, Base>::tensor(Base&& rhs) : Base(std::move(rhs))
{
}

template<typename T, typename Base>
template<typename DC>
tensor<T, Base>::tensor(const xt::xexpression<DC>& rhs)
{
    *this = rhs;
}

template<typename T, typename Base>
auto tensor<T, Base>::operator=(Base&& rhs) -> tensor&
{
    Base::operator=(std::move(rhs));

    return *this;
}

template<typename T, typename Base>
auto tensor<T, Base>::operator=(const Base& rhs) -> tensor&
{
    Base::operator=(rhs);

    return *this;
}

template<typename T, typename Base>
template<typename DC>
auto tensor<T, Base>::operator=(const xt::xexpression<DC>& rhs) -> tensor&
{
    const DC& derived = rhs.derived_cast();

    Base::resize(derived.shape());

    thrust::copy(derived.begin(), derived.end(), Base::begin());

    return *this;
}

template<typename T, typename B>
template<typename I, std::size_t N>
auto tensor<T, B>::random(const I (& shape)[N], T lower, T upper) -> tensor
{
    auto output = B::from_shape(shape);

    std::random_device r;
    std::seed_seq seq{r(), r(), r(), r(), r(), r(), r(), r()};

    std::vector<std::uint32_t> seeds(1);
    seq.generate(seeds.begin(), seeds.end());

    thrust::counting_iterator<int> sequence(0);
    thrust::transform(sequence, sequence + output.size(), output.begin(), random_functor<T>(seeds[0], lower, upper));

    return output;
}

template<typename T, typename B>
template<typename I, std::size_t N>
auto tensor<T, B>::fill(const I (& shape)[N], T value) -> tensor
{
    auto output = B::from_shape(shape);

    thrust::fill(std::begin(output), std::end(output), value);

    return output;
}

template<typename T, typename Base>
auto tensor<T, Base>::data() const noexcept -> const_pointer
{
    return Base::storage().data();
}

template<typename T, typename Base>
auto tensor<T, Base>::data() noexcept -> pointer
{
    return Base::storage().data();
}

template<typename T = float>
using cpu_array_t = tensor<T, xt_cpu_array<T>>;

using cpu_array = cpu_array_t<>;

template<typename T = float>
using cuda_array_t = tensor<T, xt_cuda_array<T>>;

using cuda_array = cuda_array_t<>;

template<typename T = float, std::size_t N = 1>
using cpu_tensor_t = tensor<T, xt_cpu_tensor<T, N>>;

template<std::size_t N = 1>
using cpu_tensor = cpu_tensor_t<float, N>;

template<typename T = float, std::size_t N = 1>
using cuda_tensor_t = tensor<T, xt_cuda_tensor<T, N>>;

template<std::size_t N = 1>
using cuda_tensor = cuda_tensor_t<float, N>;

}   // px

#endif // PIXIENN_TENSOR_H
