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

#include "CudaStorage.h"
#include "Error.h"
#include "Tensor.h"
#include <xtensor/xarray.hpp>

using namespace xt;

namespace px {

template<typename T=float,
        layout_type L = XTENSOR_DEFAULT_LAYOUT,
        class A = CudaAllocator<T>,
        class SA = CudaAllocator<typename std::vector<T, A>::size_type>,
        class UV = CudaUVector<T, A>>
using CudaContainer = xarray_container<UV, L, svector<typename UV::size_type, 4, SA, true>>;

template<class T = float>
using CudaTensorT = CudaContainer<T>;

using CudaTensor = CudaTensorT<>;

class TensorImpl
{
public:
    TensorImpl(Device device) : dev_(device)
    {
        CudaTensor a;
        CudaTensorT<float> b;

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


