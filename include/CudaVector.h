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

#ifndef PIXIENN_CUDAVECTOR_H
#define PIXIENN_CUDAVECTOR_H

#include "Error.h"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/device_malloc_allocator.h>

#include <thrust/fill.h>
#include <xtensor/xstorage.hpp>

namespace px {

template<typename T=float, typename A = thrust::device_malloc_allocator<T>>
class cuda_vector
{
public:
    using allocator_type = A;
    using traits = std::allocator_traits<A>;

    using value_type = typename traits::value_type;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = typename traits::pointer;
    using const_pointer = typename traits::const_pointer;

    using size_type = typename traits::size_type;
    using difference_type = typename traits::difference_type;

    using iterator = pointer;
    using const_iterator = const_pointer;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    cuda_vector() noexcept;
    explicit cuda_vector(const allocator_type& alloc) noexcept;
    explicit cuda_vector(size_type count, const allocator_type& alloc = allocator_type());
    cuda_vector(size_type count, const_reference value, const allocator_type& alloc = allocator_type());

    template<class InputIt, class = xt::detail::require_input_iter<InputIt>>
    cuda_vector(InputIt first, InputIt last, const allocator_type& alloc = allocator_type());

    cuda_vector(std::initializer_list<T> init, const allocator_type& alloc = allocator_type());

    ~cuda_vector();

    cuda_vector(const cuda_vector& rhs);
    cuda_vector(const cuda_vector& rhs, const allocator_type& alloc);
    cuda_vector& operator=(const cuda_vector&);

    cuda_vector(cuda_vector&& rhs) noexcept;
    cuda_vector(cuda_vector&& rhs, const allocator_type& alloc) noexcept;
    cuda_vector& operator=(cuda_vector&& rhs) noexcept;

    allocator_type get_allocator() const noexcept;

    bool empty() const noexcept;
    size_type size() const noexcept;
    void resize(size_type size);
    size_type max_size() const noexcept;
    void reserve(size_type new_cap);
    size_type capacity() const noexcept;
    void shrink_to_fit();
    void clear();

    reference operator[](size_type i);
    const_reference operator[](size_type i) const;

    reference at(size_type i);
    const_reference at(size_type i) const;

    reference front();
    const_reference front() const;

    reference back();
    const_reference back() const;

    pointer data() noexcept;
    const_pointer data() const noexcept;

    iterator begin() noexcept;
    iterator end() noexcept;

    const_iterator begin() const noexcept;
    const_iterator end() const noexcept;

    const_iterator cbegin() const noexcept;
    const_iterator cend() const noexcept;

    reverse_iterator rbegin() noexcept;
    reverse_iterator rend() noexcept;

    const_reverse_iterator rbegin() const noexcept;
    const_reverse_iterator rend() const noexcept;

    const_reverse_iterator crbegin() const noexcept;
    const_reverse_iterator crend() const noexcept;

    void swap(cuda_vector& rhs) noexcept;

private:

    template<class I>
    void init_data(I first, I last);

    void resize_impl(size_type new_size);
    pointer safe_init_allocate(size_type size);
    void safe_destroy_deallocate();

    allocator_type m_allocator;
    pointer p_begin;
    pointer p_end;
};

template<typename T, typename A>
cuda_vector<T, A>::cuda_vector() noexcept : cuda_vector(allocator_type())
{
}

template<typename T, typename A>
cuda_vector<T, A>::cuda_vector(const allocator_type& alloc) noexcept : m_allocator(alloc)
{

}

template<typename T, typename A>
cuda_vector<T, A>::cuda_vector(cuda_vector::size_type count, const allocator_type& alloc) : m_allocator(alloc)
{
    if (count != 0) {
        p_begin = xt::detail::safe_init_allocate(m_allocator, count);
        p_end = p_begin + count;
    }
}

template<typename T, typename A>
cuda_vector<T, A>::cuda_vector(cuda_vector::size_type count, const_reference value, const allocator_type& alloc)
{
    if (count != 0) {
        p_begin = m_allocator.allocate(count);
        p_end = p_begin + count;

        thrust::fill(p_begin, p_end, value);
    }
}

template<typename T, typename A>
template<class InputIt, class>
cuda_vector<T, A>::cuda_vector(InputIt first, InputIt last, const allocator_type& alloc)
{
    init_data(first, last);
}

template<typename T, typename A>
cuda_vector<T, A>::cuda_vector(std::initializer_list<T> init, const allocator_type& alloc)
{
    init_data(init.begin(), init.end());
}

template<typename T, typename A>
cuda_vector<T, A>::~cuda_vector()
{
    safe_destroy_deallocate();
}

template<typename T, typename A>
cuda_vector<T, A>::cuda_vector(const cuda_vector& rhs)
        : m_allocator(std::allocator_traits<allocator_type>::select_on_container_copy_construction(rhs.get_allocator()))
{
    init_data(rhs.p_begin, rhs.p_end);
}

template<typename T, typename A>
cuda_vector<T, A>::cuda_vector(const cuda_vector& rhs, const allocator_type& alloc) : m_allocator(alloc)
{
    init_data(rhs.p_begin, rhs.p_end);
}

template<typename T, typename A>
cuda_vector<T, A>::cuda_vector(cuda_vector&& rhs) noexcept
        : m_allocator(std::move(rhs.m_allocator)), p_begin(rhs.p_begin), p_end(rhs.p_end)
{
    rhs.p_begin = nullptr;
    rhs.p_end = nullptr;
}

template<typename T, typename A>
cuda_vector<T, A>::cuda_vector(cuda_vector&& rhs, const allocator_type& alloc) noexcept
        : m_allocator(alloc), p_begin(rhs.p_begin), p_end(rhs.p_end)
{
    rhs.p_begin = nullptr;
    rhs.p_end = nullptr;
}

template<typename T, typename A>
typename cuda_vector<T, A>::allocator_type cuda_vector<T, A>::get_allocator() const noexcept
{
    return m_allocator;
}

template<typename T, typename A>
cuda_vector<T, A>& cuda_vector<T, A>::operator=(const cuda_vector&)
{
    // No copy and swap idiom here due to performance issues
    // FIXME:
    abort();
//    if (this != &rhs) {
//        m_allocator = std::allocator_traits<allocator_type>::select_on_container_copy_construction(rhs.get_allocator());
//        resize_impl(rhs.size());
//        if (xtrivially_default_constructible<value_type>::value) {
//            std::uninitialized_copy(rhs.p_begin, rhs.p_end, p_begin);
//        } else {
//            std::copy(rhs.p_begin, rhs.p_end, p_begin);
//        }
//    }

    return *this;
}

template<typename T, typename A>
cuda_vector<T, A>& cuda_vector<T, A>::operator=(cuda_vector&& rhs) noexcept
{
    cuda_vector tmp(std::move(rhs));
    std::swap(p_begin, tmp.p_begin);
    std::swap(p_end, tmp.p_end);

    return *this;
}

template<typename T, typename A>
bool cuda_vector<T, A>::empty() const noexcept
{
    return size() == size_type(0);
}

template<typename T, typename A>
typename cuda_vector<T, A>::size_type cuda_vector<T, A>::size() const noexcept
{
    return static_cast<size_type>(p_end - p_begin);
}

template<typename T, typename A>
void cuda_vector<T, A>::resize(cuda_vector::size_type size)
{
    resize_impl(size);
}

template<typename T, typename A>
typename cuda_vector<T, A>::size_type cuda_vector<T, A>::max_size() const noexcept
{
    return m_allocator.max_size();
}

template<typename T, typename A>
void cuda_vector<T, A>::reserve(cuda_vector::size_type)
{
    // FIXME:
}

template<typename T, typename A>
typename cuda_vector<T, A>::size_type cuda_vector<T, A>::capacity() const noexcept
{
    return size();
}

template<typename T, typename A>
void cuda_vector<T, A>::shrink_to_fit()
{
    // FIXME:

}

template<typename T, typename A>
void cuda_vector<T, A>::clear()
{
    resize(size_type(0));
}

template<typename T, typename A>
typename cuda_vector<T, A>::reference cuda_vector<T, A>::operator[](cuda_vector::size_type i)
{
    //PX_ERROR_THROW(i < size(), "Index out of range");

    return p_begin[i];
}

template<typename T, typename A>
typename cuda_vector<T, A>::const_reference cuda_vector<T, A>::operator[](cuda_vector::size_type i) const
{
    //PX_ERROR_THROW(i < size(), "Index out of range");

    return p_begin[i];
}

template<typename T, typename A>
typename cuda_vector<T, A>::reference cuda_vector<T, A>::at(cuda_vector::size_type i)
{
    //PX_ERROR_THROW(i < size(), "Index out of range");

    return operator[](i);
}

template<typename T, typename A>
typename cuda_vector<T, A>::const_reference cuda_vector<T, A>::at(cuda_vector::size_type i) const
{
    // FIXME:
    return operator[](i);
}

template<typename T, typename A>
typename cuda_vector<T, A>::reference cuda_vector<T, A>::front()
{
    return p_begin[0];
}

template<typename T, typename A>
typename cuda_vector<T, A>::const_reference cuda_vector<T, A>::front() const
{
    return p_begin[0];
}

template<typename T, typename A>
typename cuda_vector<T, A>::reference cuda_vector<T, A>::back()
{
    return *(p_end - 1);
}

template<typename T, typename A>
typename cuda_vector<T, A>::const_reference cuda_vector<T, A>::back() const
{
    return *(p_end - 1);
}

template<typename T, typename A>
typename cuda_vector<T, A>::pointer cuda_vector<T, A>::data() noexcept
{
    return nullptr;
}

template<typename T, typename A>
typename cuda_vector<T, A>::const_pointer cuda_vector<T, A>::data() const noexcept
{
    return nullptr;
}

template<typename T, typename A>
typename cuda_vector<T, A>::iterator cuda_vector<T, A>::begin() noexcept
{
    return p_begin;
}

template<typename T, typename A>
typename cuda_vector<T, A>::iterator cuda_vector<T, A>::end() noexcept
{
    return p_end;
}

template<typename T, typename A>
typename cuda_vector<T, A>::const_iterator cuda_vector<T, A>::begin() const noexcept
{
    return p_begin;
}

template<typename T, typename A>
typename cuda_vector<T, A>::const_iterator cuda_vector<T, A>::end() const noexcept
{
    return p_end;
}

template<typename T, typename A>
typename cuda_vector<T, A>::const_iterator cuda_vector<T, A>::cbegin() const noexcept
{
    return p_begin;
}

template<typename T, typename A>
typename cuda_vector<T, A>::const_iterator cuda_vector<T, A>::cend() const noexcept
{
    return p_end;
}

template<typename T, typename A>
typename cuda_vector<T, A>::reverse_iterator cuda_vector<T, A>::rbegin() noexcept
{
    return reverse_iterator(end());
}

template<typename T, typename A>
typename cuda_vector<T, A>::reverse_iterator cuda_vector<T, A>::rend() noexcept
{
    return reverse_iterator(begin());
}

template<typename T, typename A>
typename cuda_vector<T, A>::const_reverse_iterator cuda_vector<T, A>::rbegin() const noexcept
{
    return const_reverse_iterator(end());
}

template<typename T, typename A>
typename cuda_vector<T, A>::const_reverse_iterator cuda_vector<T, A>::rend() const noexcept
{
    return const_reverse_iterator(begin());
}

template<typename T, typename A>
typename cuda_vector<T, A>::const_reverse_iterator cuda_vector<T, A>::crbegin() const noexcept
{
    return rbegin();
}

template<typename T, typename A>
typename cuda_vector<T, A>::const_reverse_iterator cuda_vector<T, A>::crend() const noexcept
{
    return rend();
}

template<typename T, typename A>
void cuda_vector<T, A>::swap(cuda_vector& rhs) noexcept
{

}

template<typename T, typename A>
template<class I>
void cuda_vector<T, A>::init_data(I first, I last)
{
    size_type size = static_cast<size_type>(std::distance(first, last));
    if (size != size_type(0)) {
        p_begin = m_allocator.allocate(size);

        thrust::copy(first, last, p_begin);

        p_end = p_begin + size;
    }
}

template<typename T, typename A>
void cuda_vector<T, A>::resize_impl(cuda_vector::size_type new_size)
{
    size_type old_size = size();

    if (new_size != old_size) {
        safe_destroy_deallocate();
        p_begin = safe_init_allocate(new_size);
        p_end = p_begin + new_size;
    }
}

template<typename T, typename A>
typename cuda_vector<T, A>::pointer cuda_vector<T, A>::safe_init_allocate(size_type size)
{
    pointer res = m_allocator.allocate(size);
    if (!std::is_trivially_default_constructible<value_type>::value) {
        for (pointer p = res; p != res + size; ++p) {
            traits::construct(m_allocator, p.get(), value_type());
        }
    }

    return res;
}

template<typename T, typename A>
void cuda_vector<T, A>::safe_destroy_deallocate()
{
    if (p_begin.get() != nullptr) {
        if (!std::is_trivially_default_constructible<T>::value) {
            for (pointer p = p_begin; p != p_begin + size(); ++p) {
                traits::destroy(m_allocator, p.get());
            }
        }
        traits::deallocate(m_allocator, p_begin, size());
    }
}

}   // px

#endif // PIXIENN_CUDAVECTOR_H
