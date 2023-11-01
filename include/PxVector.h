/********************************************************************************
* Copyright 2023 Maxar Technologies Inc.
* Author: Thomas A. Rieck
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
*
* SBIR DATA RIGHTS
* Contract No. HM0476-16-C-0022
* Contractor Name: Radiant Analytic Solutions Inc.
* Contractor Address: 2325 Dulles Corner Blvd. STE 1000, Herndon VA 20171
* Expiration of SBIR Data Rights Period: 2/13/2029
*
* The Government's rights to use, modify, reproduce, release, perform, display,
* or disclose technical data or computer software marked with this legend are
* restricted during the period shown as provided in paragraph (b)(4) of the
* Rights in Noncommercial Technical Data and Computer Software-Small Business
* Innovation Research (SBIR) Program clause contained in the above identified
* contract. No restrictions apply after the expiration date shown above. Any
* reproduction of technical data, computer software, or portions thereof marked
* with this legend must also reproduce the markings.
********************************************************************************/

#ifndef PIXIENN_PXVECTOR_H
#define PIXIENN_PXVECTOR_H

#include "CudaError.h"
#include "CudaUtils.cuh"

#include <cuda_runtime.h>
#include <xtensor/xarray.hpp>

namespace px {

template<typename T = float>
class PxDevVector
{
public:
    PxDevVector() : ptr_(nullptr), N_(0)
    {
    }

    PxDevVector(std::size_t N) : N_(N)
    {
        auto result = cudaMalloc(&ptr_, N * sizeof(T));
        PX_CUDA_CHECK_ERR(result);
    }

    PxDevVector(const T* phost, std::size_t N) : PxDevVector(N)
    {
        fromHost(phost, N);
    }

    PxDevVector(std::size_t N, T value) : PxDevVector(N)
    {
        fill(value);
    }

    PxDevVector(const xt::xarray<T>& xarray) : PxDevVector(xarray.data(), xarray.size())
    {
    }

    PxDevVector(PxDevVector&& rhs)
    {
        *this = std::move(rhs);
    };

    PxDevVector(const PxDevVector& rhs)
    {
        *this = rhs;
    };

    PxDevVector& operator=(PxDevVector&& rhs)
    {
        ptr_ = std::move(rhs.ptr_);
        N_ = rhs.N_;

        rhs.clear();

        return *this;
    };

    PxDevVector& operator=(const PxDevVector& rhs)
    {
        if (this != &rhs) {
            release();
            N_ = rhs.N_;

            auto result = cudaMalloc(&ptr_, N_ * sizeof(T));
            PX_CUDA_CHECK_ERR(result);

            result = cudaMemcpy(ptr_, rhs.ptr_, N_ * sizeof(T), cudaMemcpyDeviceToDevice);
            PX_CUDA_CHECK_ERR(result);
        }

        return *this;
    }

    std::size_t size() const noexcept
    {
        return N_;
    }

    const T* get() const noexcept
    {
        return ptr_;
    }

    T* get() noexcept
    {
        return ptr_;
    }

    const T* data() const noexcept
    {
        return get();
    }

    T* data() noexcept
    {
        return get();
    }

    void clear()
    {
        ptr_ = nullptr;
        N_ = 0;
    }

    void release()
    {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }

        N_ = 0;
    }

    void fromDevice(const T* devp, std::size_t N) const
    {
        PX_CHECK(N <= N_, "Element size out of range");
        auto result = cudaMemcpy(ptr_, devp, N * sizeof(T), cudaMemcpyDeviceToDevice);
        PX_CUDA_CHECK_ERR(result);
    }

    void fromDevice(const PxDevVector<T>& rhs) const
    {
        fromDevice(rhs.get(), rhs.size());
    }

    void fromHost(const T* host, std::size_t N) const
    {
        PX_CHECK(N <= N_, "Element size out of range");
        auto result = cudaMemcpy(ptr_, host, N * sizeof(T), cudaMemcpyHostToDevice);
        PX_CUDA_CHECK_ERR(result);
    }

    void fromHost(const std::vector<T>& source) const
    {
        fromHost(source.data(), source.size());
    }

    void fromHost(const xt::xarray<float>& source) const
    {
        fromHost(source.data(), source.size());
    }

    template<typename U=std::vector<T>>
    U asHost() const
    {
        U u(N_);

        auto result = cudaMemcpy(u.data(), ptr_, N_ * sizeof(T), cudaMemcpyDeviceToHost);
        PX_CUDA_CHECK_ERR(result);

        return u;
    }

    void fill(T value)
    {
        fill_gpu(ptr_, N_, value);
    }

    static PxDevVector<T> random(std::size_t N, T a = 0.f, T b = 1.f)
    {
        PxDevVector<T> vec(N);
        random_generate(vec.ptr_, N);
        return vec;
    }

    virtual ~PxDevVector()
    {
        release();
    }

private:
    T* ptr_ = nullptr;
    size_t N_ = 0;
};

}   // px


#endif //PIXIENN_PXVECTOR_H