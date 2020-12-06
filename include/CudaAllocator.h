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

#ifndef PIXIENN_CUDAALLOCATOR_H
#define PIXIENN_CUDAALLOCATOR_H

#include <memory>
#include <cuda_runtime_api.h>
#include "CudaError.h"

namespace px {

template<typename T>
struct CudaAllocator
{
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using pointer = value_type*;
    using const_pointer = const value_type*;

    CudaAllocator() = default;

    template<typename U>
    constexpr CudaAllocator(const CudaAllocator<U>&) noexcept
    {}

    CudaAllocator(const CudaAllocator&) noexcept
    {}

    ~CudaAllocator() noexcept
    {}

    pointer allocate(std::size_t n, const void* = nullptr)
    {
        void* buffer;
        auto result = cudaMalloc(&buffer, n * sizeof(T));
        PX_CUDA_CHECK_ERR(result);

        result = cudaMemset(buffer, 0, n * sizeof(T));
        PX_CUDA_CHECK_ERR(result);

        return static_cast<pointer>(buffer);
    }

    void deallocate(pointer p, std::size_t)
    {
        PX_CUDA_CHECK_ERR(cudaFree(p));
    }

    std::size_t max_size() const throw()
    {
        return std::numeric_limits<std::size_t>::max() / sizeof(T);
    }
};

template<typename T>
inline bool operator==(const CudaAllocator<T>&, const CudaAllocator<T>&)
{ return true; }

template<typename T>
inline bool operator!=(const CudaAllocator<T>&, const CudaAllocator<T>&)
{ return false; }

}   // px

#endif // PIXIENN_CUDAALLOCATOR_H
