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

template<typename T, typename I, std::size_t L>
auto cuda_random(const I (& shape)[L], T lower = 0, T upper = 1)
{

    std::vector<std::size_t> s(&shape[0], &shape[L]);   // FIXME: why can't we forward the shape?
    cuda_array<T> output(std::move(s));

    thrust::counting_iterator<int> sequence(0);
    thrust::transform(sequence, sequence + output.size(), output.begin(), random_functor<T>(lower, upper));

    return output;
}

class TensorImpl
{
public:
    TensorImpl(Device device) : dev_(device)
    {
        auto r = cuda_random<float>({1000, 1000, 10});

        int i = 0;
        for (const auto& value: r) {
            std::cout << ++i << "    " << value << std::endl;
        }
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


