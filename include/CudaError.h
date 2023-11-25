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

#ifndef PIXIENN_CUDA_ERROR_H
#define PIXIENN_CUDA_ERROR_H

#include "Error.h"
#include <driver_types.h>

namespace px {

class CudaError : public Error
{
public:
    CudaError() noexcept;
    explicit CudaError(cudaError_t error) noexcept;
    CudaError(cudaError_t error, const char* file, unsigned int line, const char* function,
              const std::string& message) noexcept;

    static void check(cudaError_t error, const char* file, unsigned int line, const char* function,
                      const char* format, ...);
    static void check(cudaError_t error, const char* file, unsigned int line, const char* function);
    static void check(const char* file, unsigned int line, const char* function);

private:
    cudaError_t error_;
};

#define PX_CUDA_CHECK_ERR(error) \
    px::CudaError::check(error, __FILENAME__, __LINE__, __FUNCTION__)

#define PX_CUDA_CHECK_LAST() \
    px::CudaError::check(__FILENAME__, __LINE__, __FUNCTION__)

} // px

#endif // PIXIENN_CUDA_ERROR_H
