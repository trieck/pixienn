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

#ifndef PXTENSOR_H
#define PXTENSOR_H

#include <cuda_runtime.h>
#include <xtensor/xarray.hpp>

#include "Common.h"
#include "CudaUtils.cuh"
#include "CudaError.h"

namespace px {

enum class Device
{
    CPU,
    CUDA
};

///////////////////////////////////////////////////////////////////////////////
template<typename T>
class PxCudaAllocator
{
public:
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;

    using size_type = std::size_t;

    PxCudaAllocator() = default;
    PxCudaAllocator(size_type n);
    ~PxCudaAllocator() = default;

    pointer alloc(size_type n);
    void dealloc(pointer ptr, size_type n);
};

template<typename T>
void PxCudaAllocator<T>::dealloc(PxCudaAllocator::pointer ptr, PxCudaAllocator::size_type /*n*/)
{
    auto result = cudaFree(ptr);
    PX_CUDA_CHECK_ERR(result);
}

template<typename T>
PxCudaAllocator<T>::PxCudaAllocator(PxCudaAllocator::size_type n)
{
    alloc(n);
}

template<typename T>
PxCudaAllocator<T>::pointer PxCudaAllocator<T>::alloc(PxCudaAllocator::size_type n)
{
    pointer ptr;
    auto result = cudaMalloc(&ptr, n * sizeof(T));
    PX_CUDA_CHECK_ERR(result);

    return ptr;
}

///////////////////////////////////////////////////////////////////////////////
template<typename T, Device D, typename Base>
class PxTensor : public Base
{
public:
    using pointer = typename Base::pointer;
    using const_pointer = typename Base::const_pointer;
    using size_type = typename Base::size_type;

    PxTensor();
    PxTensor(std::initializer_list<T> init);

    const_pointer data() const noexcept;
    pointer data() noexcept;

    Device device() const;

private:
    Device dev_ = D;
};

template<typename T, Device D, typename Base>
PxTensor<T, D, Base>::PxTensor(std::initializer_list<T> init)
 : Base(init)
{
}

template<typename T, Device D, typename Base>
PxTensor<T, D, Base>::pointer PxTensor<T, D, Base>::data() noexcept
{
    return nullptr;
}
template<typename T, Device D, typename Base>
PxTensor<T, D, Base>::const_pointer PxTensor<T, D, Base>::data() const noexcept
{
    return nullptr;
}

template<typename T, Device D, typename Base>
PxTensor<T, D, Base>::PxTensor()
{

}

template<typename T, Device D, typename Base>
Device PxTensor<T, D, Base>::device() const
{
    return dev_;
}

///////////////////////////////////////////////////////////////////////////////
template<typename T, typename A = PxCudaAllocator<T>>
class PxCudaVectorT
{
public:
    using allocator_type = A;
    using pointer = typename A::pointer;
    using const_pointer = typename A::const_pointer;
    using reference = typename A::reference;
    using const_reference = typename A::const_reference;
    using size_type = typename A::size_type;

    PxCudaVectorT() = default;
    explicit PxCudaVectorT(size_type count, const allocator_type& alloc = allocator_type());
    PxCudaVectorT(size_type count, const_reference value, const allocator_type& alloc = allocator_type());
    explicit PxCudaVectorT(std::initializer_list<T> init, const allocator_type& alloc = allocator_type());

    PxCudaVectorT(const PxCudaVectorT& rhs);
    PxCudaVectorT(PxCudaVectorT&& rhs);

    PxCudaVectorT& operator=(const PxCudaVectorT& rhs);
    PxCudaVectorT& operator=(PxCudaVectorT&& rhs);
    ~PxCudaVectorT();

    pointer data() noexcept;
    const_pointer data() const noexcept;
    size_type size() const;
    bool empty() const;
    void release();

    template<typename U = std::vector<T>>
    U toHost() const;

private:
    pointer ptr_ = nullptr;
    size_type size_ = 0;
    allocator_type alloc_;
};

template<typename T, typename A>
template<typename U>
U PxCudaVectorT<T, A>::toHost() const
{
    U u(size_);

    auto result = cudaMemcpy(u.data(), ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
    PX_CUDA_CHECK_ERR(result);

    return u;
}

template<typename T, typename A>
PxCudaVectorT<T, A>::PxCudaVectorT(std::initializer_list<T> init, const allocator_type& alloc)
        : alloc_(alloc)
{
    size_ = std::distance(init.begin(), init.end());
    ptr_ = alloc_.alloc(size_);

    auto result = cudaMemcpy(ptr_, init.begin(), size_ * sizeof(T), cudaMemcpyHostToDevice);
    PX_CUDA_CHECK_ERR(result);
}

template<typename T, typename A>
PxCudaVectorT<T, A>::PxCudaVectorT(const PxCudaVectorT& rhs)
{
    *this = rhs;
}

template<typename T, typename A>
PxCudaVectorT<T, A>::PxCudaVectorT(PxCudaVectorT&& rhs)
{
    *this = std::move(rhs);
}

template<typename T, typename A>
PxCudaVectorT<T, A>& PxCudaVectorT<T, A>::operator=(PxCudaVectorT&& rhs)
{
    ptr_ = std::move(rhs.ptr_);
    size_ = rhs.size_;

    rhs.ptr_ = nullptr;
    rhs.size_ = 0;

    return *this;
}

template<typename T, typename A>
PxCudaVectorT<T, A>& PxCudaVectorT<T, A>::operator=(const PxCudaVectorT& rhs)
{
    if (this != &rhs) {
        alloc_.dealloc(ptr_, size_);
        size_ = rhs.size();
        ptr_ = alloc_.alloc(size_);

        auto result = cudaMemcpy(ptr_, rhs.ptr_, size_ * sizeof(T), cudaMemcpyDeviceToDevice);
        PX_CUDA_CHECK_ERR(result);
    }

    return *this;
}

template<typename T, typename A>
void PxCudaVectorT<T, A>::release()
{
    alloc_.dealloc(ptr_, size_);
    ptr_ = nullptr;
    size_ = 0;
}

template<typename T, typename A>
PxCudaVectorT<T, A>::size_type PxCudaVectorT<T, A>::size() const
{
    return size_;
}

template<typename T, typename A>
bool PxCudaVectorT<T, A>::empty() const
{
    return size_ == 0;
}

template<typename T, typename A>
PxCudaVectorT<T, A>::const_pointer PxCudaVectorT<T, A>::data() const noexcept
{
    return ptr_;
}
template<typename T, typename A>
PxCudaVectorT<T, A>::pointer PxCudaVectorT<T, A>::data() noexcept
{
    return ptr_;
}

template<typename T, typename A>
PxCudaVectorT<T, A>::~PxCudaVectorT()
{
    release();
}

template<typename T, typename A>
PxCudaVectorT<T, A>::PxCudaVectorT(size_type count, const allocator_type& alloc) : alloc_(alloc)
{
    if (count != 0) {
        ptr_ = alloc_.alloc(count);
        size_ = count;
    }
}

template<typename T, typename A>
PxCudaVectorT<T, A>::PxCudaVectorT(size_type count, const_reference value, const allocator_type& alloc)
        : PxCudaVectorT(count)
{
    if (count != 0) {
        fill_gpu(ptr_, count, value);
    }
}

///////////////////////////////////////////////////////////////////////////////
template<typename T = float>
using PxCudaTensorT = PxTensor<T, Device::CUDA, PxCudaVectorT<T>>;

using PxCudaTensor = PxCudaTensorT<>;

} // px

#endif // PXTENSOR_H
