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

#include "CudaVector.h"
#include "Error.h"
#include "Tensor.h"
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>

namespace px {

using namespace xt;

template<typename T>
using cuda_array = xarray_container<cuda_vector<T>>;

template<class T, std::size_t N>
using cuda_tensor = xtensor_container<cuda_vector<T>, N>;

class TensorImpl
{
public:
    TensorImpl(Device device) : dev_(device)
    {
        cuda_array<float> a{ 1, 2, 3 };

        cuda_tensor<float, 2> weights;
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


