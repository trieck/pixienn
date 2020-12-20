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

#ifndef PIXIENN_TENSOR_CUH
#define PIXIENN_TENSOR_CUH

#include <xtensor/xlayout.hpp>
#include "Common.h"
#include "CudaTensor.cuh"

namespace px {

enum class Device
{
    CPU, CUDA
};

template<typename T>
using cpu_array = xt::xarray<T>;

template<typename T = float, Device D = Device::CPU,
        typename Base = std::conditional_t<D == Device::CPU, cpu_array<T>, cuda_array<T>>>
class tensor : public Base
{
public:
    tensor() = default;
    tensor(const tensor& rhs) = default;
    tensor(tensor&& rhs) = default;
    tensor(const Base& rhs);
    tensor(Base&& rhs);

    tensor& operator=(tensor&& rhs) = default;
    tensor& operator=(const tensor& rhs) = default;
    tensor& operator=(Base&& rhs);
    tensor& operator=(const Base& rhs);

    Device device() const noexcept;
    bool is_cuda() const noexcept;

    template<typename I, std::size_t N>
    static tensor random(const I (& shape)[N], T lower = 0, T upper = 1);

    template<typename I, std::size_t N>
    static tensor fill(const I (& shape)[N], T value);

private:
    using this_type = tensor<T, D, Base>;

    Device dev_ = D;
};

template<typename T, Device D, typename B>
Device tensor<T, D, B>::device() const noexcept
{
    return dev_;
}

template<typename T, Device D, typename B>
bool tensor<T, D, B>::is_cuda() const noexcept
{
    return dev_ == Device::CUDA;
}

template<typename T, Device D, typename Base>
tensor<T, D, Base>::tensor(const Base& rhs)
{
    *this = rhs;
}

template<typename T, Device D, typename Base>
tensor<T, D, Base>::tensor(Base&& rhs)
{
    *this = std::move(rhs);
}

template<typename T, Device D, typename Base>
auto tensor<T, D, Base>::operator=(Base&& rhs) -> tensor&
{
    Base::operator=(std::move(rhs));

    return *this;
}

template<typename T, Device D, typename Base>
auto tensor<T, D, Base>::operator=(const Base& rhs) -> tensor&
{
    Base::operator=(rhs);

    return *this;
}

template<typename T, Device D, typename B>
template<typename I, std::size_t N>
auto tensor<T, D, B>::random(const I (& shape)[N], T lower, T upper) -> tensor
{
    auto output = B::from_shape(shape);

    thrust::counting_iterator<int> sequence(0);
    thrust::transform(sequence, sequence + output.size(), output.begin(), random_functor<T>(lower, upper));

    return output;
}

template<typename T, Device D, typename B>
template<typename I, std::size_t N>
auto tensor<T, D, B>::fill(const I (& shape)[N], T value) -> tensor
{
    auto output = B::from_shape(shape);

    thrust::fill(std::begin(output), std::end(output), value);

    return output;
}

template<typename T = float>
using mycuda_tensor_t = tensor<T, Device::CUDA>;
using mycuda_tensor = mycuda_tensor_t<>;

template<typename T = float>
using mycpu_tensor_t = tensor<T, Device::CPU>;
using mycpu_tensor = mycpu_tensor_t<>;


void foobar();

}   // px

#endif // PIXIENN_TENSOR_CUH
