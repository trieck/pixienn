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

#ifndef PIXIENN_CPUBUFFER_H
#define PIXIENN_CPUBUFFER_H

#include "Buffer.h"

namespace px {

class CpuBuffer : public Buffer
{
public:
    CpuBuffer(size_t size);
    CpuBuffer(const void* p, size_t size);
    CpuBuffer(const CpuBuffer& rhs) = delete;
    CpuBuffer(CpuBuffer&& rhs);

    CpuBuffer& operator=(const CpuBuffer& rhs) = delete;
    CpuBuffer& operator=(CpuBuffer&& rhs);

    template<typename T>
    CpuBuffer(const T* input, size_t size);
    ~CpuBuffer();

private:
    void release();
};

template<typename T>
CpuBuffer::CpuBuffer(const T* input, size_t size) : CpuBuffer((const void*) input, sizeof(T) * size)
{}


}   // px

#endif // PIXIENN_CPUBUFFER_H
