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

#ifndef PIXIENN_CUDATENSOR_CUH
#define PIXIENN_CUDATENSOR_CUH

#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>

namespace px {

template<typename T>
using cuda_array = xt::xarray_container<thrust::device_vector<T>>;

template<typename T, std::size_t N>
using cuda_tensor = xt::xtensor_container<thrust::device_vector<T>, N>;

template<typename T, typename D = thrust::random::uniform_real_distribution<T>, typename E = thrust::default_random_engine>
struct random_functor
{
    D dist_;
    E rng_;

    __host__ __device__
    random_functor(T lower = 0, T upper = 1)
    {
        dist_ = D(lower, upper);
    };

    __host__ __device__
    T operator()(int n)
    {
        rng_.discard(n);
        return dist_(rng_);
    }
};

template<typename T, typename I, std::size_t N>
inline auto cuda_random(const I (& shape)[N], T lower = 0, T upper = 1) noexcept
{
    auto output = cuda_tensor<T, N>::from_shape(shape);

    thrust::counting_iterator<int> sequence(0);
    thrust::transform(sequence, sequence + output.size(), output.begin(), random_functor<T>(lower, upper));

    return output;
}

template<class T, class I, std::size_t N>
inline auto cuda_fill(const I (& shape)[N], T value) noexcept
{
    auto output = cuda_tensor<T, N>::from_shape(shape);

    thrust::fill(std::begin(output), std::end(output), value);

    return output;
}

template<class T, class I, std::size_t N>
inline auto cuda_ones(const I (& shape)[N]) noexcept
{
    return cuda_fill<T>(shape, 1);
}

template<class T, class I, std::size_t N>
inline auto cuda_empty(const I (& shape)[N]) noexcept
{
    return cuda_tensor<T, N>::from_shape(shape);
}

}   // namespace px

#endif //PIXIENN_CUDATENSOR_CUH
