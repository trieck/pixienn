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

#include "Error.h"
#include "Tensor.h"
#include <boost/chrono.hpp>
#include <boost/thread/thread.hpp>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>

namespace px {

using namespace xt;
using namespace thrust;

template<typename T>
using cuda_array = xarray_container<device_vector<T>>;

template<typename T, std::size_t N>
using cuda_tensor = xtensor_container<device_vector<T>, N>;

template<typename T, typename D = random::uniform_real_distribution<T>, typename E = thrust::default_random_engine>
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
    float operator()(const unsigned int n)
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

template<typename T, typename I, std::size_t N>
void test_random(const I (& shape)[N], T lower = 0, T upper = 1)
{
    const auto r = cuda_random<T>(shape, lower, upper);

    int i = 0;
    for (const auto& value: r) {
        std::cout << ++i << "    " << value << std::endl;
    }
}

template<typename T, typename I, std::size_t N>
void test_fill(const I (& shape)[N], T value)
{
    const auto r = cuda_fill<T>(shape, value);

    int i = 0;
    for (const auto& v: r) {
        std::cout << ++i << "    " << v << std::endl;
    }
}

template<typename T, typename I, std::size_t N>
void test_empty(const I (& shape)[N])
{
    cuda_tensor<T, N> r = cuda_empty<T>(shape);

    int i = 0;
    for (const auto& v: r) {
        std::cout << ++i << "    " << v << std::endl;
    }
}

void sleep(int seconds)
{
    boost::this_thread::sleep_for(boost::chrono::seconds(seconds));
}

class TensorImpl
{
public:
    TensorImpl(Device device) : dev_(device)
    {
        test_random<float>({10, 10, 10}, 0, 100);

        sleep(1);

        test_fill<double>({10, 10, 10}, 18.0123456789);

        sleep(1);

        test_empty<int>({10, 10, 10});
    }

    ~TensorImpl()
    {}

    Device device() const noexcept
    {
        return dev_;
    }

    const std::vector<int64_t> dims() const noexcept
    {
        return dims_;
    }

    int64_t dim(int index) const
    {
        PX_CHECK(index >= 0 && index < dims_.size(), "Index out of range.");
        return dims_[index];
    }

private:
    Device dev_;
    std::vector<int64_t> dims_;
};

Tensor::Tensor(Device dev)
{
    impl_ = std::make_unique<TensorImpl>(dev);
}

Tensor::~Tensor()
{

}

Device Tensor::device() const noexcept
{
    return impl_->device();
}

bool Tensor::isCuda() const noexcept
{
    return impl_->device() == Device::CUDA;
}

}   // px


