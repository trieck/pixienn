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

#include "CudaError.h"

#include <cstdarg>
#include <cuda_runtime_api.h>

namespace px {

#define PX_ERROR_MAX_LEN (2048)

static std::string cudaErrorString(cudaError_t error)
{
    return cudaGetErrorString(error);
}

static std::string cudaErrorString()
{
    return cudaErrorString(cudaPeekAtLastError());
}

CudaError::CudaError() noexcept: error_(cudaPeekAtLastError()),
                                 Error(__FILENAME__, __LINE__, __FUNCTION__, cudaErrorString())
{
}

CudaError::CudaError(cudaError_t error) noexcept: error_(error),
                                                  Error(__FILENAME__, __LINE__, __FUNCTION__, cudaErrorString(error))
{
}

CudaError::CudaError(cudaError_t error, const char* file, unsigned int line, const char* function,
                     const std::string& message) noexcept: error_(error), Error(file, line, function, message)
{
}

void CudaError::check(cudaError_t error, const char* file, unsigned int line, const char* function,
                      const char* format, ...)
{
    if (error != cudaSuccess) {
        char buf[PX_ERROR_MAX_LEN] = { 0 };

        va_list args;
        va_start(args, format);
        vsnprintf(buf, PX_ERROR_MAX_LEN, format, args);
        va_end(args);

        std::string message = "Cuda error: (" + cudaErrorString(error) + "), " + buf;

        throw CudaError(error, file, line, function, message);
    }
}

void CudaError::check(cudaError_t error, const char* file, unsigned int line, const char* function)
{
    if (error != cudaSuccess) {
        std::string message = "Cuda error: (" + cudaErrorString(error) + ")";
        throw CudaError(error, file, line, function, message);
    }
}

void CudaError::check(const char* file, unsigned int line, const char* function)
{
    check(cudaPeekAtLastError(), file, line, function);
}

}   // px
