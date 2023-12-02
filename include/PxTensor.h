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

#include <numeric>

#if USE_CUDA

#include <cuda_runtime.h>
#include "CudaUtils.cuh"
#include "CudaError.h"

#endif

#include "Common.h"
#include "Error.h"
#include "Strides.h"
#include "CpuUtil.h"

namespace px {

enum class Device
{
    CPU,
    CUDA
};

#ifdef USE_CUDA

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
class PxCudaAllocatorT
{
public:
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;

    using size_type = std::size_t;

    PxCudaAllocatorT() = default;
    PxCudaAllocatorT(size_type n);
    ~PxCudaAllocatorT() = default;

    pointer alloc(size_type n);
    void dealloc(pointer ptr, size_type n);
};

template<typename T>
void PxCudaAllocatorT<T>::dealloc(PxCudaAllocatorT::pointer ptr, PxCudaAllocatorT::size_type /*n*/)
{
    auto result = cudaFree(ptr);
    PX_CUDA_CHECK_ERR(result);
}

template<typename T>
PxCudaAllocatorT<T>::PxCudaAllocatorT(PxCudaAllocatorT::size_type n)
{
    alloc(n);
}

template<typename T>
PxCudaAllocatorT<T>::pointer PxCudaAllocatorT<T>::alloc(PxCudaAllocatorT::size_type n)
{

    pointer ptr;
    auto result = cudaMalloc(&ptr, n * sizeof(T));
    PX_CUDA_CHECK_ERR(result);


    return ptr;
}

#endif // USE_CUDA

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
class PxVector
{
public:
    PxVector() = delete;
    virtual ~PxVector() = default;
    using Ptr = std::unique_ptr<PxVector>;

    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;

    Device device() const;

    virtual std::vector<T> asVector() const = 0;
    virtual bool empty() const = 0;
    virtual size_type size() const noexcept = 0;
    virtual const_pointer data() const noexcept = 0;
    virtual pointer data() noexcept = 0;

protected:
    PxVector(Device dev);

private:
    Device dev_;
};

template<typename T>
PxVector<T>::PxVector(Device dev) : dev_(dev)
{
}

template<typename T>
Device PxVector<T>::device() const
{
    return dev_;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T, typename A = std::allocator<T>>
class PxCpuVectorT : public PxVector<T>
{
public:
    using C = std::vector<T>;
    using allocator_type = A;
    using pointer = typename C::pointer;
    using const_pointer = typename C::const_pointer;
    using value_type = typename C::value_type;
    using reference = typename C::reference;
    using const_reference = typename C::const_reference;
    using size_type = typename C::size_type;
    using iterator = typename C::iterator;
    using const_iterator = typename C::const_iterator;

    PxCpuVectorT();
    explicit PxCpuVectorT(size_type count, const allocator_type& alloc = allocator_type());
    PxCpuVectorT(size_type count, const_reference value, const allocator_type& alloc = allocator_type());
    PxCpuVectorT(std::initializer_list<T>&& init, const allocator_type& alloc = allocator_type());

    PxCpuVectorT(const PxCpuVectorT& rhs);
    PxCpuVectorT(PxCpuVectorT&& rhs);

    PxCpuVectorT& operator=(const PxCpuVectorT& rhs);
    PxCpuVectorT& operator=(PxCpuVectorT&& rhs);
    ~PxCpuVectorT() = default;

    std::vector<T> asVector() const override;
    size_type size() const noexcept override;
    bool empty() const override;
    pointer data() noexcept override;
    const_pointer data() const noexcept override;

    void copy(std::initializer_list<T>&& init);
    void copy(const PxCpuVectorT& rhs);
    void copyHost(const T*, size_type n);
    void release();

    void emplaceBack(T&& value);
    reference operator[](int i);
    const_reference operator[](int i) const;

    PxCpuVectorT operator+(const PxCpuVectorT& rhs) const;
    PxCpuVectorT operator-(const PxCpuVectorT& rhs) const;
    PxCpuVectorT operator+(T value) const;
    PxCpuVectorT operator-(T value) const;

#ifdef USE_CUDA
    void copyDevice(const T*, size_type n);
#endif

    const_iterator begin() const;
    const_iterator end() const;

    iterator begin();
    iterator end();
private:
    C container_;
};

template<typename T, typename A>
void PxCpuVectorT<T, A>::release()
{
    container_.clear();
}

template<typename T, typename A>
void PxCpuVectorT<T, A>::copy(const PxCpuVectorT& rhs)
{
    copyHost(rhs.data(), rhs.size());
}

template<typename T, typename A>
auto PxCpuVectorT<T, A>::operator+(T value) const -> PxCpuVectorT
{
    PxCpuVectorT<T, A> sum(size());
    for (auto i = 0; i < size(); ++i) {
        sum[i] = (*this)[i] + value;
    }

    return sum;
}

template<typename T, typename A>
auto PxCpuVectorT<T, A>::operator-(T value) const -> PxCpuVectorT
{
    PxCpuVectorT<T, A> difference(size());
    for (auto i = 0; i < size(); ++i) {
        difference[i] = (*this)[i] - value;
    }

    return difference;
}

template<typename T, typename A>
auto PxCpuVectorT<T, A>::operator+(const PxCpuVectorT& rhs) const -> PxCpuVectorT
{
    PX_CHECK(size() == rhs.size(), "vectors must be same size.");

    PxCpuVectorT<T, A> sum(size());
    for (auto i = 0; i < size(); ++i) {
        sum[i] = (*this)[i] + rhs[i];
    }

    return sum;
}

template<typename T, typename A>
auto PxCpuVectorT<T, A>::operator-(const PxCpuVectorT& rhs) const -> PxCpuVectorT
{
    PX_CHECK(size() == rhs.size(), "vectors must be same size.");

    PxCpuVectorT<T, A> diff(size());
    for (auto i = 0; i < size(); ++i) {
        diff[i] = (*this)[i] - rhs[i];
    }

    return diff;
}

template<typename T, typename A>
auto PxCpuVectorT<T, A>::operator[](int i) -> reference
{
    return container_[i];
}

template<typename T, typename A>
auto PxCpuVectorT<T, A>::operator[](int i) const -> const_reference
{
    return container_[i];
}

template<typename T, typename A>
void PxCpuVectorT<T, A>::emplaceBack(T&& value)
{
    container_.emplace_back(std::move(value));
}

template<typename T, typename A>
void PxCpuVectorT<T, A>::copyHost(const T* ptr, size_type n)
{
    PX_CHECK(n <= size(), "Element size out of range");

    memcpy(data(), ptr, n * sizeof(T));
}

#ifdef USE_CUDA

template<typename T, typename A>
void PxCpuVectorT<T, A>::copyDevice(const T* ptr, size_type n)
{
    PX_CHECK(n <= size(), "Element size out of range");

    auto result = cudaMemcpy(data(), ptr, n * sizeof(T), cudaMemcpyDeviceToHost);
    PX_CUDA_CHECK_ERR(result);
}

#endif

template<typename T, typename A>
PxCpuVectorT<T, A>::PxCpuVectorT() : PxVector<T>(Device::CPU)
{
}

template<typename T, typename A>
void PxCpuVectorT<T, A>::copy(std::initializer_list<T>&& init)
{
    std::copy(init.begin(), init.end(), container_.begin());
}

template<typename T, typename A>
std::vector<T> PxCpuVectorT<T, A>::asVector() const
{
    return container_;
}

template<typename T, typename A>
auto PxCpuVectorT<T, A>::begin() const -> const_iterator
{
    return container_.begin();
}

template<typename T, typename A>
auto PxCpuVectorT<T, A>::end() const -> const_iterator
{
    return container_.end();
}

template<typename T, typename A>
auto PxCpuVectorT<T, A>::begin() -> iterator
{
    return container_.begin();
}

template<typename T, typename A>
auto PxCpuVectorT<T, A>::end() -> iterator
{
    return container_.end();
}

template<typename T, typename A>
bool PxCpuVectorT<T, A>::empty() const
{
    return container_.empty();
}

template<typename T, typename A>
PxCpuVectorT<T, A>::size_type PxCpuVectorT<T, A>::size() const noexcept
{
    return container_.size();
}

template<typename T, typename A>
PxCpuVectorT<T, A>::const_pointer PxCpuVectorT<T, A>::data() const noexcept
{
    return container_.data();
}

template<typename T, typename A>
PxCpuVectorT<T, A>::pointer PxCpuVectorT<T, A>::data() noexcept
{
    return container_.data();
}

template<typename T, typename A>
PxCpuVectorT<T, A>& PxCpuVectorT<T, A>::operator=(PxCpuVectorT&& rhs)
{
    container_ = std::move(rhs.container_);

    return *this;
}

template<typename T, typename A>
PxCpuVectorT<T, A>& PxCpuVectorT<T, A>::operator=(const PxCpuVectorT& rhs)
{
    if (this != &rhs) {
        container_ = rhs.container_;
    }

    return *this;
}

template<typename T, typename A>
PxCpuVectorT<T, A>::PxCpuVectorT(PxCpuVectorT&& rhs) : PxVector<T>(Device::CPU)
{
    *this = std::move(rhs);
}

template<typename T, typename A>
PxCpuVectorT<T, A>::PxCpuVectorT(const PxCpuVectorT& rhs) : PxVector<T>(Device::CPU)
{
    *this = rhs;
}

template<typename T, typename A>
PxCpuVectorT<T, A>::PxCpuVectorT(std::initializer_list<T>&& init, const allocator_type& alloc)
        : PxVector<T>(Device::CPU), container_(init, alloc)
{
}

template<typename T, typename A>
PxCpuVectorT<T, A>::PxCpuVectorT(size_type count, const_reference value, const allocator_type& alloc)
        : PxVector<T>(Device::CPU), container_(count, value, alloc)
{
}

template<typename T, typename A>
PxCpuVectorT<T, A>::PxCpuVectorT(size_type count, const allocator_type& alloc)
        : PxVector<T>(Device::CPU), container_(count, alloc)
{
}

#ifdef USE_CUDA

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T, typename A = PxCudaAllocatorT<T>>
class PxCudaVectorT : public PxVector<T>
{
public:
    using allocator_type = A;
    using pointer = typename A::pointer;
    using const_pointer = typename A::const_pointer;
    using reference = typename A::reference;
    using const_reference = typename A::const_reference;
    using size_type = typename A::size_type;

    PxCudaVectorT();
    explicit PxCudaVectorT(size_type count, const allocator_type& alloc = allocator_type());
    explicit PxCudaVectorT(std::initializer_list<T>&& init, const allocator_type& alloc = allocator_type());
    PxCudaVectorT(size_type count, const_reference value, const allocator_type& alloc = allocator_type());
    PxCudaVectorT(const T* begin, const T* end, const allocator_type& alloc = allocator_type());
    PxCudaVectorT(const PxCudaVectorT& rhs);
    PxCudaVectorT(PxCudaVectorT&& rhs);

    PxCudaVectorT& operator=(const PxCudaVectorT& rhs);
    PxCudaVectorT& operator=(PxCudaVectorT&& rhs);
    ~PxCudaVectorT();

    pointer data() noexcept override;
    const_pointer data() const noexcept override;
    size_type size() const noexcept override;
    bool empty() const noexcept override;

    std::vector<T> asVector() const override;

    template<typename U = PxCpuVectorT<T>>
    U asContainer () const;

    void copy(std::initializer_list<T>&& init);
    void copy(const T* begin, const T* end);
    void copy(const PxCpuVectorT<T>& rhs);
    void copy(const PxCudaVectorT& rhs);
    void copyHost(const T*, size_type n);
    void copyDevice(const T*, size_type n);

    void release();
private:
    pointer ptr_ = nullptr;
    size_type size_ = 0;
    allocator_type alloc_;
};

template<typename T, typename A>
void PxCudaVectorT<T, A>::copy(const PxCpuVectorT<T>& rhs)
{
    copyHost(rhs.data(), rhs.size());
}

template<typename T, typename A>
void PxCudaVectorT<T, A>::copyHost(const T* ptr, size_type n)
{
    PX_CHECK(n <= size_, "Element size out of range");

    auto result = cudaMemcpy(ptr_, ptr, n * sizeof(T), cudaMemcpyHostToDevice);
    PX_CUDA_CHECK_ERR(result);
}

template<typename T, typename A>
void PxCudaVectorT<T, A>::copyDevice(const T* ptr, size_type n)
{
    PX_CHECK(n <= size_, "Element size out of range");

    auto result = cudaMemcpy(ptr_, ptr, n * sizeof(T), cudaMemcpyDeviceToDevice);
    PX_CUDA_CHECK_ERR(result);
}

template<typename T, typename A>
void PxCudaVectorT<T, A>::copy(const PxCudaVectorT& rhs)
{
    copyDevice(rhs.data(), rhs.size());
}

template<typename T, typename A>
void PxCudaVectorT<T, A>::copy(const T* begin, const T* end)
{
    size_ = std::distance(begin, end);
    ptr_ = alloc_.alloc(size_);

    auto result = cudaMemcpy(ptr_, begin, size_ * sizeof(T), cudaMemcpyHostToDevice);
    PX_CUDA_CHECK_ERR(result);
}

template<typename T, typename A>
void PxCudaVectorT<T, A>::copy(std::initializer_list<T>&& init)
{
    copy(init.begin(), init.end());
}

template<typename T, typename A>
std::vector<T> PxCudaVectorT<T, A>::asVector() const
{
    std::vector<T> v(size_);

    auto result = cudaMemcpy(v.data(), ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
    PX_CUDA_CHECK_ERR(result);

    return v;
}

template<typename T, typename A>
template<typename U>
U PxCudaVectorT<T, A>::asContainer() const
{
    U u(size_);

    auto result = cudaMemcpy(u.data(), ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
    PX_CUDA_CHECK_ERR(result);

    return u;
}

template<typename T, typename A>
PxCudaVectorT<T, A>::PxCudaVectorT(std::initializer_list<T>&& init, const allocator_type& alloc)
        : PxVector<T>(Device::CUDA), alloc_(alloc)
{
    copy(std::move(init));
}

template<typename T, typename A>
PxCudaVectorT<T, A>::PxCudaVectorT(const T* begin, const T* end, const allocator_type& alloc)
        : PxVector<T>(Device::CUDA), alloc_(alloc)
{
    copy(begin, end);
}

template<typename T, typename A>
PxCudaVectorT<T, A>::PxCudaVectorT() : PxVector<T>(Device::CUDA)
{
}

template<typename T, typename A>
PxCudaVectorT<T, A>::PxCudaVectorT(const PxCudaVectorT& rhs) : PxVector<T>(Device::CUDA)
{
    *this = rhs;
}

template<typename T, typename A>
PxCudaVectorT<T, A>::PxCudaVectorT(PxCudaVectorT&& rhs) : PxVector<T>(Device::CUDA)
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
PxCudaVectorT<T, A>::size_type PxCudaVectorT<T, A>::size() const noexcept
{
    return size_;
}

template<typename T, typename A>
bool PxCudaVectorT<T, A>::empty() const noexcept
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
PxCudaVectorT<T, A>::PxCudaVectorT(size_type count, const allocator_type& alloc)
        : PxVector<T>(Device::CUDA), alloc_(alloc)
{
    if (count != 0) {
        ptr_ = alloc_.alloc(count);
        size_ = count;
    }
}

template<typename T, typename A>
PxCudaVectorT<T, A>::PxCudaVectorT(size_type count, const_reference value, const allocator_type& alloc)
        : PxCudaVectorT(count, alloc)
{
    if (count != 0) {
        fill_gpu(ptr_, count, value);
    }
}

#endif // USE_CUDA

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T, std::size_t N>
class PxTensor
{
public:
    PxTensor() = delete;
    virtual ~PxTensor() = default;
    using Ptr = std::unique_ptr<PxTensor>;

    using pointer = T*;
    using const_pointer = const T*;
    using value_type = T;
    using size_type = std::size_t;
    using shape_type = std::array<size_type, N>;

    Device device() const;
    virtual const shape_type& shape() const noexcept;
    virtual const shape_type& strides() const noexcept;

    size_type size() const noexcept;

    virtual void copy(const PxTensor<T, N>& rhs) = 0;
    virtual const_pointer data() const noexcept = 0;
    virtual pointer data() noexcept = 0;
    virtual std::vector<T> asVector() const = 0;

    static size_type dims();
protected:
    PxTensor(Device dev);

    PxTensor(Device dev, const shape_type& shape);

private:
    static constexpr size_type dims_ = N;
    shape_type shape_{};
    shape_type strides_{};
    Device dev_;
};

template<typename T, std::size_t N>
PxTensor<T, N>::PxTensor(Device dev) : dev_(dev)
{
}

template<typename T, std::size_t N>
PxTensor<T, N>::PxTensor(Device dev, const shape_type& shape) : dev_(dev), shape_(shape)
{
    compute_strides(shape, strides_);
}

template<typename T, std::size_t N>
PxTensor<T, N>::size_type PxTensor<T, N>::size() const noexcept
{
    auto size = std::accumulate(shape_.cbegin(), shape_.cend(), size_type(1), std::multiplies<size_type>());

    return size;
}

template<typename T, std::size_t N>
const PxTensor<T, N>::shape_type& PxTensor<T, N>::shape() const noexcept
{
    return shape_;
}

template<typename T, std::size_t N>
const PxTensor<T, N>::shape_type& PxTensor<T, N>::strides() const noexcept
{
    return strides_;
}

template<typename T, std::size_t N>
PxTensor<T, N>::size_type PxTensor<T, N>::dims()
{
    return dims_;
}

template<typename T, std::size_t N>
Device PxTensor<T, N>::device() const
{
    return dev_;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T, Device D, typename C, std::size_t N>
class PxTensorImpl : public PxTensor<T, N>
{
public:
    using self_type = PxTensorImpl<T, D, C, N>;
    using base_type = PxTensor<T, N>;
    using value_type = T;
    using pointer = typename C::pointer;
    using const_pointer = typename C::const_pointer;
    using size_type = typename C::size_type;
    using shape_type = base_type::shape_type;
    using device_type = std::integral_constant<Device, D>;

    PxTensorImpl();
    PxTensorImpl(const shape_type& shape);
    PxTensorImpl(const shape_type& shape, T value);
    PxTensorImpl(const shape_type& shape, std::initializer_list<T>&& init);
    PxTensorImpl(const shape_type& shape, const PxCpuVectorT<T>& init);

    void copy(const PxTensor<T, N>& rhs) override;
    const_pointer data() const noexcept override;
    pointer data() noexcept override;
    std::vector<T> asVector() const override;
    void release();

private:
    C container_;
};

template<typename T, Device D, typename C, std::size_t N>
void PxTensorImpl<T, D, C, N>::release()
{
    container_.release();
}

template<typename T, Device D, typename C, std::size_t N>
void PxTensorImpl<T, D, C, N>::copy(const PxTensorImpl::base_type& rhs)
{
#ifdef USE_CUDA
    if (rhs.device() == Device::CPU) {
        container_.copyHost(rhs.data(), rhs.size());
    } else {
        container_.copyDevice(rhs.data(), rhs.size());
    }
#else
    container_.copyHost(rhs.data(), rhs.size());
#endif
}

template<typename T, Device D, typename C, std::size_t N>
PxTensorImpl<T, D, C, N>::PxTensorImpl() : PxTensor<T, N>(D)
{
}

template<typename T, Device D, typename C, std::size_t N>
PxTensorImpl<T, D, C, N>::PxTensorImpl(const shape_type& shape)
        : PxTensor<T, N>(D, shape), container_(PxTensor<T, N>::size())
{
}

template<typename T, Device D, typename C, std::size_t N>
PxTensorImpl<T, D, C, N>::PxTensorImpl(const shape_type& shape, T value)
        : PxTensor<T, N>(D, shape), container_(PxTensor<T, N>::size(), value)
{
}

template<typename T, Device D, typename C, std::size_t N>
PxTensorImpl<T, D, C, N>::PxTensorImpl(const shape_type& shape, std::initializer_list<T>&& init)
        : PxTensor<T, N>(D, shape), container_(PxTensor<T, N>::size())
{
    auto isize = std::distance(init.begin(), init.end());
    auto size = PxTensor<T, N>::size();
    PX_CHECK(isize == size, "Initializer list size must match shape.");

    container_.copy(std::forward<decltype(init)>(init));
}

template<typename T, Device D, typename C, std::size_t N>
PxTensorImpl<T, D, C, N>::PxTensorImpl(const shape_type& shape, const PxCpuVectorT<T>& init)
        : PxTensor<T, N>(D, shape), container_(PxTensor<T, N>::size())
{
    container_.copy(init);
}

template<typename T, Device D, typename C, std::size_t N>
std::vector<T> PxTensorImpl<T, D, C, N>::asVector() const
{
    return container_.asVector();
}

template<typename T, Device D, typename C, std::size_t N>
PxTensorImpl<T, D, C, N>::pointer PxTensorImpl<T, D, C, N>::data() noexcept
{
    return container_.data();
}

template<typename T, Device D, typename C, std::size_t N>
PxTensorImpl<T, D, C, N>::const_pointer PxTensorImpl<T, D, C, N>::data() const noexcept
{
    return container_.data();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
using PxCpuVector = PxCpuVectorT<float>;

template<typename T, std::size_t N>
using PxCpuTensorT = PxTensorImpl<T, Device::CPU, PxCpuVectorT<T>, N>;

template<std::size_t N>
using PxCpuTensor = PxCpuTensorT<float, N>;

#ifdef USE_CUDA

using PxCudaAllocator = PxCudaAllocatorT<float>;
using PxCudaVector = PxCudaVectorT<float>;

template<typename T, std::size_t N>
using PxCudaTensorT = PxTensorImpl<T, Device::CUDA, PxCudaVectorT<T>, N>;

template<std::size_t N>
using PxCudaTensor = PxCudaTensorT<float, N>;

#endif // USE_CUDA

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T, typename... Args>
PxVector<T>::Ptr makeVector(Device dev, Args&& ...args)
{
    typename PxVector<T>::Ptr p;

#ifdef USE_CUDA
    if (dev == Device::CUDA) {
        p = std::make_unique<PxCudaVectorT<T>>(std::forward<Args>(args)...);
    } else {
        p = std::make_unique<PxCpuVectorT<T>>(std::forward<Args>(args)...);
    }
#else
    PX_CHECK(dev == Device::CPU, "Invalid device.");
    p = std::make_unique<PxCpuVectorT<T>>(std::forward<Args>(args)...);
#endif

    return p;
}

template<typename... Args>
PxVector<float>::Ptr cpuVector(Args&& ...args)
{
    return makeVector<float>(Device::CPU, std::forward<Args>(args)...);
}

#ifdef USE_CUDA

template<typename... Args>
PxVector<float>::Ptr cudaVector(Args&& ...args)
{
    return makeVector<float>(Device::CUDA, std::forward<Args>(args)...);
}

#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T, Device D, std::size_t N, typename... Args>
PxTensor<T, N>::Ptr makeTensor(Args&& ...args)
{
    typename PxTensor<T, N>::Ptr p;

#ifdef USE_CUDA
    if constexpr (D == Device::CUDA) {
        p = std::make_unique<PxCudaTensorT<T, N>>(std::forward<Args>(args)...);
    } else {
        p = std::make_unique<PxCpuTensorT<T, N>>(std::forward<Args>(args)...);
    }
#else
    p = std::make_unique<PxCpuTensorT< T, N>>(std::forward<Args>(args)...);
#endif

    return p;
}

template<std::size_t N, typename... Args>
PxTensor<float, N>::Ptr cpuTensor(Args&& ...args)
{
    return makeTensor<float, Device::CPU, N>(std::forward<Args>(args)...);
}

#ifdef USE_CUDA

template<std::size_t N, typename... Args>
PxTensor<float, N>::Ptr cudaTensor(Args&& ...args)
{
    return makeTensor<float, Device::CUDA, N>(std::forward<Args>(args)...);
}

#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
T random(const typename T::shape_type& shape)
{
    T out(shape);

#ifdef USE_CUDA
    if constexpr (typename T::device_type() == Device::CUDA) {
        random_generate_gpu(out.data(), out.size(), typename T::value_type(0), typename T::value_type(1));
    } else {
        random_generate_cpu(out.data(), out.size(), typename T::value_type(0), typename T::value_type(1));
    }
#else
    random_generate_cpu(out.data(), out.size(), typename T::value_type(0), typename T::value_type(1));
#endif // USE_CUDA

    return out;
}

} // px

#endif // PXTENSOR_H
