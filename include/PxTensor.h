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

///////////////////////////////////////////////////////////////////////////////
template<typename T>
class cuda_ptr_t
{
public:
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    cuda_ptr_t() = default;
    explicit cuda_ptr_t(pointer p);
    cuda_ptr_t(const cuda_ptr_t& rhs) = default;
    cuda_ptr_t(cuda_ptr_t&& rhs) noexcept = default;
    ~cuda_ptr_t() = default;

    cuda_ptr_t& operator=(const cuda_ptr_t& rhs) = default;
    cuda_ptr_t& operator=(cuda_ptr_t&& rhs) noexcept = default;

    [[nodiscard]] const_pointer get() const noexcept;
    pointer get() noexcept;

    T operator*() const;
    cuda_ptr_t& operator++();
    cuda_ptr_t operator+(size_type n) const;

private:
    pointer ptr_ = nullptr;
};

template<typename T>
cuda_ptr_t<T>::cuda_ptr_t(cuda_ptr_t::pointer p) : ptr_(p)
{
}

template<typename T>
cuda_ptr_t<T>::pointer cuda_ptr_t<T>::get() noexcept
{
    return ptr_;
}

template<typename T>
cuda_ptr_t<T>::const_pointer cuda_ptr_t<T>::get() const noexcept
{
    return ptr_;
}

template<typename T>
cuda_ptr_t<T>& cuda_ptr_t<T>::operator++()
{
    ptr_++;
    return *this;
}

template<typename T>
T cuda_ptr_t<T>::operator*() const
{
    T value;

    auto result = cudaMemcpy(&value, ptr_, sizeof(T), cudaMemcpyDeviceToHost);
    PX_CUDA_CHECK_ERR(result);

    return value;
}

template<typename T>
cuda_ptr_t<T> cuda_ptr_t<T>::operator+(cuda_ptr_t::size_type n) const
{
    return cuda_ptr_t(ptr_ + n);
}

template<typename T>
bool operator==(const cuda_ptr_t<T>& lhs, const cuda_ptr_t<T>& rhs)
{
    return lhs.get() == rhs.get();
}

template<typename T>
cuda_ptr_t<T>::difference_type operator-(const cuda_ptr_t<T>& lhs,
                                         const cuda_ptr_t<T>& rhs)
{
    return lhs.get() - rhs.get();
}

///////////////////////////////////////////////////////////////////////////////
template<typename T>
class cuda_allocator_t
{
public:
    using value_type = T;
    using pointer = cuda_ptr_t<T>;
    using const_pointer = const cuda_ptr_t<T>;
    using reference = value_type&;
    using const_reference = const value_type&;

    using size_type = std::size_t;
    using difference_type = typename pointer::difference_type;

    cuda_allocator_t() = default;
    explicit cuda_allocator_t(size_type n);
    cuda_allocator_t(const cuda_allocator_t& rhs) = default;
    cuda_allocator_t(cuda_allocator_t&& rhs) noexcept = default;
    ~cuda_allocator_t() = default;

    cuda_allocator_t& operator=(const cuda_allocator_t& rhs) = default;
    cuda_allocator_t& operator=(cuda_allocator_t&& rhs) noexcept = default;

    pointer alloc(size_type n);
    void dealloc(pointer ptr);
};

template<typename T>
cuda_allocator_t<T>::cuda_allocator_t(size_type n)
{
    alloc(n);
}

template<typename T>
cuda_allocator_t<T>::pointer cuda_allocator_t<T>::alloc(cuda_allocator_t<T>::size_type n)
{
    T* ptr;
    auto result = cudaMalloc(&ptr, n * sizeof(T));
    PX_CUDA_CHECK_ERR(result);

    return cuda_ptr_t<T>(ptr);
}

template<typename T>
void cuda_allocator_t<T>::dealloc(pointer ptr)
{
    cudaFree(ptr.get());
}

///////////////////////////////////////////////////////////////////////////////
template<typename T, typename A=cuda_allocator_t<T>>
class cuda_vector_t
{
public:
    using allocator_type = A;
    using value_type = typename A::value_type;
    using difference_type = typename A::difference_type;

    using pointer = typename A::pointer;
    using const_pointer = typename A::const_pointer;
    using reference = typename A::reference;
    using const_reference = typename A::const_reference;

    using size_type = typename A::size_type;

    cuda_vector_t(const allocator_type& alloc = allocator_type()) noexcept;
    explicit cuda_vector_t(size_type count, const allocator_type& alloc = allocator_type());
    cuda_vector_t(size_type count, const_reference value, const allocator_type& alloc = allocator_type());

    explicit cuda_vector_t(std::initializer_list<T> init, const allocator_type& alloc = allocator_type());
    cuda_vector_t(const cuda_vector_t& rhs);
    cuda_vector_t(const cuda_vector_t& rhs, const allocator_type& alloc);
    ~cuda_vector_t();

    cuda_vector_t& operator=(const cuda_vector_t& rhs);

    [[nodiscard]] bool empty() const noexcept;
    size_type size() const noexcept;
    void resize(size_type size);
    void clear();
    void randomize(T a = 0.f, T b = 1.f);

    using iterator = pointer;
    using const_iterator = const_pointer;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    iterator begin() noexcept;
    iterator end() noexcept;
    [[nodiscard]] const_iterator begin() const noexcept;
    [[nodiscard]] const_iterator end() const noexcept;

    iterator rbegin() noexcept;
    iterator rend() noexcept;
    [[nodiscard]] const_iterator rbegin() const noexcept;
    [[nodiscard]] const_iterator rend() const noexcept;

    [[nodiscard]] const_iterator cbegin() const noexcept;
    [[nodiscard]] const_iterator cend() const noexcept;

    value_type operator[](size_type i);
    pointer data() noexcept;

private:
    using init_iter = std::initializer_list<T>::const_iterator;
    void init_data(init_iter first, init_iter last);
    void to_host(T* out, size_type n, size_type offset);
    void to_device(init_iter first, init_iter last, iterator out);
    void to_device(const T& value, iterator iterator);

    iterator begin_, end_;
    allocator_type a_;
};

template<typename T, typename A>
cuda_vector_t<T, A>::cuda_vector_t(const allocator_type& alloc) noexcept
        : a_(alloc)
{
}

template<typename T, typename A>
void cuda_vector_t<T, A>::randomize(T a, T b)
{
    random_generate(begin_.get(), size(), a, b);
}

template<typename T, typename A>
void cuda_vector_t<T, A>::init_data(cuda_vector_t::init_iter first,
                                    cuda_vector_t::init_iter last)
{
    size_type size = static_cast<size_type>(std::distance(first, last));
    if (size != size_type(0)) {
        begin_ = a_.alloc(size);
        end_ = begin_ + size;
        to_device(first, last, begin_);
    }
}

template<typename T, typename A>
void cuda_vector_t<T, A>::to_device(cuda_vector_t::init_iter first,
                                    cuda_vector_t::init_iter last, iterator out)
{
    for (; first != last; ++first, (void) ++out) {
        to_device(*first, out);
    }
}

template<typename T, typename A>
void cuda_vector_t<T, A>::to_device(const T& value, iterator iterator)
{
    auto result = cudaMemcpy(iterator.get(), &value, sizeof(T), cudaMemcpyHostToDevice);
    PX_CUDA_CHECK_ERR(result);
}

template<typename T, typename A>
cuda_vector_t<T, A>::cuda_vector_t(std::initializer_list<T> init, const allocator_type& alloc) : a_(alloc)
{
    init_data(init.begin(), init.end());
}

template<typename T, typename A>
cuda_vector_t<T, A>::iterator cuda_vector_t<T, A>::begin() noexcept
{
    return begin_;
}

template<typename T, typename A>
cuda_vector_t<T, A>::iterator cuda_vector_t<T, A>::end() noexcept
{
    return end_;
}

template<typename T, typename A>
cuda_vector_t<T, A>::const_iterator cuda_vector_t<T, A>::begin() const noexcept
{
    return begin_;
}

template<typename T, typename A>
cuda_vector_t<T, A>::const_iterator cuda_vector_t<T, A>::end() const noexcept
{
    return end_;
}

template<typename T, typename A>
cuda_vector_t<T, A>::const_iterator cuda_vector_t<T, A>::cbegin() const noexcept
{
    return begin_;
}

template<typename T, typename A>
cuda_vector_t<T, A>::const_iterator cuda_vector_t<T, A>::cend() const noexcept
{
    return end_;
}

template<typename T, typename A>
cuda_vector_t<T, A>::iterator cuda_vector_t<T, A>::rbegin() noexcept
{
    return reverse_iterator(begin);
}

template<typename T, typename A>
cuda_vector_t<T, A>::iterator cuda_vector_t<T, A>::rend() noexcept
{
    return reverse_iterator(end);
}

template<typename T, typename A>
cuda_vector_t<T, A>::const_iterator cuda_vector_t<T, A>::rbegin() const noexcept
{
    return const_reverse_iterator(begin);
}

template<typename T, typename A>
cuda_vector_t<T, A>::const_iterator cuda_vector_t<T, A>::rend() const noexcept
{
    return const_reverse_iterator(end);
}

template<typename T, typename A>
void cuda_vector_t<T, A>::clear()
{
    resize(0);
}

template<typename T, typename A>
void cuda_vector_t<T, A>::resize(size_type new_size)
{
    if (new_size != size()) {
        // does not preserve elements
        a_.dealloc(begin_);
        begin_ = a_.alloc(new_size);
        end_ = begin_ + new_size;
    }
}

template<typename T, typename A>
cuda_vector_t<T, A>::cuda_vector_t(size_type count, const allocator_type& alloc) : a_(alloc)
{
    if (count != 0) {
        begin_ = a_.alloc(count);
        end_ = begin_ + count;
    }
}


template<typename T, typename A>
cuda_vector_t<T, A>::cuda_vector_t(size_type count, const_reference value, const allocator_type& alloc)
        : cuda_vector_t<T, A>(count, alloc)
{
    if (count != 0) {
        fill_gpu(begin_.get(), size(), value);
    }
}

template<typename T, typename A>
cuda_vector_t<T, A>::cuda_vector_t(const cuda_vector_t& rhs)
{
    *this = rhs;
}

template<typename T, typename A>
cuda_vector_t<T, A>::cuda_vector_t(const cuda_vector_t& rhs, const allocator_type& alloc) : a_(alloc)
{
    *this = rhs;
}

template<typename T, typename A>
cuda_vector_t<T, A>& cuda_vector_t<T, A>::operator=(const cuda_vector_t& rhs)
{
    if (this != &rhs) {
        a_.dealloc(begin_);
        a_ = rhs.a_;
        auto n = rhs.size();
        begin_ = a_.alloc(n);
        end_ = begin_ + n;

        auto result = cudaMemcpy(begin_.get(), rhs.begin_.get(), n * sizeof(T), cudaMemcpyDeviceToDevice);
        PX_CUDA_CHECK_ERR(result);
    }

    return *this;
}

template<typename T, typename A>
cuda_vector_t<T, A>::~cuda_vector_t()
{
    a_.dealloc(begin_);
}

template<typename T, typename A>
typename cuda_vector_t<T, A>::size_type cuda_vector_t<T, A>::size() const noexcept
{
    return (end_ - begin_);
}

template<typename T, typename A>
bool cuda_vector_t<T, A>::empty() const noexcept
{
    return size() == size_type(0);
}

template<typename T, typename A>
void cuda_vector_t<T, A>::to_host(T* out, size_type n, size_type offset)
{
    PX_CHECK((offset + (n - 1)) < size(), "Element size out of range");

    auto src = begin_ + offset;

    auto result = cudaMemcpy(out, src.get(), n * sizeof(T), cudaMemcpyDeviceToHost);
    PX_CUDA_CHECK_ERR(result);
}

template<typename T, typename A>
cuda_vector_t<T, A>::value_type cuda_vector_t<T, A>::operator[](size_type i)
{
    T x;
    to_host(&x, 1, i);
    return x;
}

template<typename T, typename A>
cuda_vector_t<T, A>::pointer cuda_vector_t<T, A>::data() noexcept
{
    return begin_.get();
}

using cuda_vector = cuda_vector_t<float>;

enum class Device
{
    CPU,
    CUDA
};

}   // px

#endif // PXTENSOR_H
