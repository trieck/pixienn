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

#include "CudaBuffer.h"
#include "CudaError.h"

#include <cuda_runtime_api.h>

namespace px {

CudaBuffer::CudaBuffer(size_t size) : Buffer(size)
{
    auto result = cudaMalloc(&buffer_, size);
    PX_CUDA_CHECK_ERR(result);

    result = cudaMemset(buffer_, 0, size);
    PX_CUDA_CHECK_ERR(result);
}

CudaBuffer::CudaBuffer(const void* p, size_t size) : Buffer(size)
{
    auto result = cudaMalloc(&buffer_, size);
    PX_CUDA_CHECK_ERR(result);

    result = cudaMemcpy(buffer_, p, size, cudaMemcpyHostToDevice);
    PX_CUDA_CHECK_ERR(result);
}

CudaBuffer::CudaBuffer(CudaBuffer&& rhs)
{
    *this = std::move(rhs);
}

CudaBuffer::~CudaBuffer()
{
    release();
}

CudaBuffer& CudaBuffer::operator=(CudaBuffer&& rhs)
{
    if (this != &rhs) {
        buffer_ = std::move(rhs.buffer_);
        size_ = std::move(rhs.size_);

        rhs.release();
    }

    return *this;
}

void CudaBuffer::release()
{
    if (buffer_ != nullptr) {
        PX_CUDA_CHECK_ERR(cudaFree(buffer_));
        buffer_ = nullptr;
        size_ = 0;
    }
}

}   // px