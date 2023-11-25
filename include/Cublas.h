/********************************************************************************
* Copyright 2020-2023 Thomas A. Rieck, All Rights Reserved
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

#ifndef PIXIENN_CUBLAS_H
#define PIXIENN_CUBLAS_H

#if USE_CUDA

#include <cublas_v2.h>
#include "Common.h"
#include "Error.h"

#define PX_CHECK_CUBLAS(x) \
    PX_CHECK(x == CUBLAS_STATUS_SUCCESS, "cublas failed with error %d", x);

template<typename T>
using cublas_create = cublasStatus_t  (*)(T**);

template<typename T>
using cublas_destroy = cublasStatus_t  (*)(T*);

template<typename T, cublas_create<T> ctor, cublas_destroy<T> dtor>
class cublas_descriptor
{
public:
    using Type = cublas_descriptor<T, ctor, dtor>;
    using Ptr = std::unique_ptr<Type>;

    cublas_descriptor() : handle_(nullptr)
    {
        auto status = ctor(&handle_);
        PX_CHECK_CUBLAS(status);
    }

    ~cublas_descriptor()
    {
        destroy();
    }

    cublas_descriptor(const cublas_descriptor&& rhs) = delete;
    cublas_descriptor(cublas_descriptor&& rhs) noexcept
    {
        *this = std::move(rhs);
    }

    cublas_descriptor& operator=(const cublas_descriptor& rhs) = delete;
    cublas_descriptor& operator=(cublas_descriptor&& rhs) noexcept
    {
        if (this != &rhs) {
            handle_ = std::move(rhs.handle_);
            rhs.handle_ = nullptr;
        }

        return *this;
    }

    inline operator T*() const noexcept
    {
        return handle_;
    };

    void release()
    {
        destroy();
    }

private:
    void destroy()
    {
        if (handle_ != nullptr) {
            PX_CHECK_CUBLAS(dtor(handle_));
            handle_ = nullptr;
        }
    }

    T* handle_;
};

using CublasContext = cublas_descriptor<cublasContext, cublasCreate, cublasDestroy>;

#endif  // USE_CUDA

#endif  // PIXIENN_CUBLAS_H

