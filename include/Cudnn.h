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

#ifndef PIXIENN_CUDNN_H
#define PIXIENN_CUDNN_H

#if USE_CUDA

#include <cudnn.h>

#include "Error.h"

namespace px {

#define PX_CHECK_CUDNN(x) \
    PX_CHECK(x == CUDNN_STATUS_SUCCESS, cudnnGetErrorString(x));

template<typename T>
using cudnn_create = cudnnStatus_t (*)(T**);

template<typename T>
using cudnn_destroy = cudnnStatus_t (*)(T*);

template<typename T, cudnn_create<T> ctor, cudnn_destroy<T> dtor>
class cudnn_descriptor
{
public:
    cudnn_descriptor() : handle_(nullptr)
    {
        PX_CHECK_CUDNN(ctor(&handle_));
    }

    ~cudnn_descriptor()
    {
        destroy();
    }

    cudnn_descriptor(const cudnn_descriptor& rhs) = delete;

    cudnn_descriptor(cudnn_descriptor&& rhs) noexcept
    {
        *this = std::move(rhs);
    }

    cudnn_descriptor& operator=(const cudnn_descriptor& rhs) = delete;
    cudnn_descriptor& operator=(cudnn_descriptor&& rhs) noexcept
    {
        handle_ = std::move(rhs.handle_);
        rhs.handle_ = nullptr;

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
            PX_CHECK_CUDNN(dtor(handle_));
            handle_ = nullptr;
        }
    }

    T* handle_;
};

template<typename T>
cudnnStatus_t cudnnDestroyContext(T* c)
{
    auto status = CUDNN_STATUS_SUCCESS;

    try {
        cudnnDestroy(c);
    } catch (...) {
        return CUDNN_STATUS_INTERNAL_ERROR;
    }

    return status;
}

using CudnnContext = cudnn_descriptor<cudnnContext, cudnnCreate, cudnnDestroyContext>;
using CudnnConvDesc = cudnn_descriptor<cudnnConvolutionStruct, cudnnCreateConvolutionDescriptor, cudnnDestroyConvolutionDescriptor>;
using CudnnFilterDesc = cudnn_descriptor<cudnnFilterStruct, cudnnCreateFilterDescriptor, cudnnDestroyFilterDescriptor>;
using CudnnTensorDesc = cudnn_descriptor<cudnnTensorStruct, cudnnCreateTensorDescriptor, cudnnDestroyTensorDescriptor>;

}   // px

#endif // USE_CUDA

#endif // PIXIENN_CUDNN_H
