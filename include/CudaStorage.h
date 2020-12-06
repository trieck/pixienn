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

#ifndef PIXIENN_CUDASTORAGE_H
#define PIXIENN_CUDASTORAGE_H

#include "CudaAllocator.h"
#include <xtensor/xstorage.hpp>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>

namespace px {

template<typename T=float, typename A = CudaAllocator<T>>
class CudaUVector
{
public:
    using allocator_type = A;
    using value_type = typename std::allocator_traits<A>::value_type;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = typename std::allocator_traits<A>::pointer;
    using const_pointer = typename std::allocator_traits<A>::const_pointer;

    using size_type = typename std::allocator_traits<A>::size_type;
    using difference_type = typename std::allocator_traits<A>::difference_type;

    using iterator = pointer;
    using const_iterator = const_pointer;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    CudaUVector() noexcept;
    explicit CudaUVector(const allocator_type& alloc) noexcept;
    explicit CudaUVector(size_type count, const allocator_type& alloc = allocator_type());
    CudaUVector(size_type count, const_reference value, const allocator_type& alloc = allocator_type());

    template<class InputIt, class = xt::detail::require_input_iter<InputIt>>
    CudaUVector(InputIt first, InputIt last, const allocator_type& alloc = allocator_type());

    CudaUVector(std::initializer_list<T> init, const allocator_type& alloc = allocator_type());

    ~CudaUVector();

    CudaUVector(const CudaUVector& rhs);
    CudaUVector(const CudaUVector& rhs, const allocator_type& alloc);
    CudaUVector& operator=(const CudaUVector&);

    CudaUVector(CudaUVector&& rhs) noexcept;
    CudaUVector(CudaUVector&& rhs, const allocator_type& alloc) noexcept;
    CudaUVector& operator=(CudaUVector&& rhs) noexcept;

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

    void swap(CudaUVector& rhs) noexcept;

private:

    template<class I>
    void init_data(I first, I last);

    void resize_impl(size_type new_size);

    allocator_type m_allocator;
    pointer p_begin;
    pointer p_end;
};

template<typename T, typename A>
CudaUVector<T, A>::CudaUVector() noexcept : CudaUVector(allocator_type())
{
}

template<typename T, typename A>
CudaUVector<T, A>::CudaUVector(const allocator_type& alloc) noexcept
        : m_allocator(alloc), p_begin(nullptr), p_end(nullptr)
{

}

template<typename T, typename A>
CudaUVector<T, A>::CudaUVector(CudaUVector::size_type count, const allocator_type& alloc)
        : m_allocator(alloc), p_begin(nullptr), p_end(nullptr)
{
    if (count != 0) {
        p_begin = xt::detail::safe_init_allocate(m_allocator, count);
        p_end = p_begin + count;
    }
}

template<typename T, typename A>
CudaUVector<T, A>::CudaUVector(CudaUVector::size_type count, const_reference value, const allocator_type& alloc)
{
    if (count != 0) {
        p_begin = m_allocator.allocate(count);
        p_end = p_begin + count;

        thrust::device_ptr<T> begin(p_begin), end(p_end);
        thrust::fill(begin, end, value);
    }
}

template<typename T, typename A>
template<class InputIt, class>
CudaUVector<T, A>::CudaUVector(InputIt first, InputIt last, const allocator_type& alloc)
{

}

template<typename T, typename A>
CudaUVector<T, A>::CudaUVector(std::initializer_list<T> init, const allocator_type& alloc)
{

}

template<typename T, typename A>
CudaUVector<T, A>::~CudaUVector()
{

}

template<typename T, typename A>
CudaUVector<T, A>::CudaUVector(const CudaUVector& rhs)
{

}

template<typename T, typename A>
CudaUVector<T, A>::CudaUVector(const CudaUVector& rhs, const allocator_type& alloc)
{

}

template<typename T, typename A>
CudaUVector<T, A>::CudaUVector(CudaUVector&& rhs) noexcept
{

}

template<typename T, typename A>
CudaUVector<T, A>::CudaUVector(CudaUVector&& rhs, const allocator_type& alloc) noexcept
{

}

template<typename T, typename A>
typename CudaUVector<T, A>::allocator_type CudaUVector<T, A>::get_allocator() const noexcept
{
    return m_allocator;
}

template<typename T, typename A>
CudaUVector<T, A>& CudaUVector<T, A>::operator=(const CudaUVector&)
{
    return *this;
}

template<typename T, typename A>
CudaUVector<T, A>& CudaUVector<T, A>::operator=(CudaUVector&& rhs) noexcept
{
    return *this;
}

template<typename T, typename A>
bool CudaUVector<T, A>::empty() const noexcept
{
    return false;
}

template<typename T, typename A>
typename CudaUVector<T, A>::size_type CudaUVector<T, A>::size() const noexcept
{
    return 0;
}

template<typename T, typename A>
void CudaUVector<T, A>::resize(CudaUVector::size_type size)
{

}

template<typename T, typename A>
typename CudaUVector<T, A>::size_type CudaUVector<T, A>::max_size() const noexcept
{
    return 0;
}

template<typename T, typename A>
void CudaUVector<T, A>::reserve(CudaUVector::size_type new_cap)
{

}

template<typename T, typename A>
typename CudaUVector<T, A>::size_type CudaUVector<T, A>::capacity() const noexcept
{
    return 0;
}

template<typename T, typename A>
void CudaUVector<T, A>::shrink_to_fit()
{

}

template<typename T, typename A>
void CudaUVector<T, A>::clear()
{

}
template<typename T, typename A>
typename CudaUVector<T, A>::reference CudaUVector<T, A>::operator[](CudaUVector::size_type i)
{
    return *this;
}

template<typename T, typename A>
typename CudaUVector<T, A>::const_reference CudaUVector<T, A>::operator[](CudaUVector::size_type i) const
{
    return *this;
}

template<typename T, typename A>
typename CudaUVector<T, A>::reference CudaUVector<T, A>::at(CudaUVector::size_type i)
{
    return *this;
}

template<typename T, typename A>
typename CudaUVector<T, A>::const_reference CudaUVector<T, A>::at(CudaUVector::size_type i) const
{
    return *this;
}

template<typename T, typename A>
typename CudaUVector<T, A>::reference CudaUVector<T, A>::front()
{
    return p_begin[0];
}

template<typename T, typename A>
typename CudaUVector<T, A>::const_reference CudaUVector<T, A>::front() const
{
    return p_begin[0];
}

template<typename T, typename A>
typename CudaUVector<T, A>::reference CudaUVector<T, A>::back()
{
    return *(p_end - 1);
}

template<typename T, typename A>
typename CudaUVector<T, A>::const_reference CudaUVector<T, A>::back() const
{
    return *(p_end - 1);
}

template<typename T, typename A>
typename CudaUVector<T, A>::pointer CudaUVector<T, A>::data() noexcept
{
    return nullptr;
}

template<typename T, typename A>
typename CudaUVector<T, A>::const_pointer CudaUVector<T, A>::data() const noexcept
{
    return nullptr;
}

template<typename T, typename A>
typename CudaUVector<T, A>::iterator CudaUVector<T, A>::begin() noexcept
{
    return nullptr;
}

template<typename T, typename A>
typename CudaUVector<T, A>::iterator CudaUVector<T, A>::end() noexcept
{
    return nullptr;
}

template<typename T, typename A>
typename CudaUVector<T, A>::const_iterator CudaUVector<T, A>::begin() const noexcept
{
    return nullptr;
}

template<typename T, typename A>
typename CudaUVector<T, A>::const_iterator CudaUVector<T, A>::end() const noexcept
{
    return nullptr;
}

template<typename T, typename A>
typename CudaUVector<T, A>::const_iterator CudaUVector<T, A>::cbegin() const noexcept
{
    return nullptr;
}

template<typename T, typename A>
typename CudaUVector<T, A>::const_iterator CudaUVector<T, A>::cend() const noexcept
{
    return nullptr;
}

template<typename T, typename A>
typename CudaUVector<T, A>::reverse_iterator CudaUVector<T, A>::rbegin() noexcept
{
    return reverse_iterator(end());
}

template<typename T, typename A>
typename CudaUVector<T, A>::reverse_iterator CudaUVector<T, A>::rend() noexcept
{
    return reverse_iterator(begin());
}

template<typename T, typename A>
typename CudaUVector<T, A>::const_reverse_iterator CudaUVector<T, A>::rbegin() const noexcept
{
    return const_reverse_iterator(end());
}

template<typename T, typename A>
typename CudaUVector<T, A>::const_reverse_iterator CudaUVector<T, A>::rend() const noexcept
{
    return const_reverse_iterator(begin());
}

template<typename T, typename A>
typename CudaUVector<T, A>::const_reverse_iterator CudaUVector<T, A>::crbegin() const noexcept
{
    return rbegin();
}

template<typename T, typename A>
typename CudaUVector<T, A>::const_reverse_iterator CudaUVector<T, A>::crend() const noexcept
{
    return rend();
}

template<typename T, typename A>
void CudaUVector<T, A>::swap(CudaUVector& rhs) noexcept
{

}

template<typename T, typename A>
template<class I>
void CudaUVector<T, A>::init_data(I first, I last)
{
}

template<typename T, typename A>
void CudaUVector<T, A>::resize_impl(CudaUVector::size_type new_size)
{
}

}   // px

#endif // PIXIENN_CUDASTORAGE_H
