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

#include "CpuBuffer.h"
#include "Error.h"

namespace px {

CpuBuffer::CpuBuffer(size_t size) : Buffer(size)
{
    buffer_ = std::calloc(size, 1);
    PX_CHECK(buffer_ != nullptr, "Unable to allocate %d bytes.", size);
}

CpuBuffer::CpuBuffer(const void* p, size_t size)
{
    buffer_ = std::malloc(size);
    PX_CHECK(buffer_ != nullptr, "Unable to allocate %d bytes.", size);

    std::memcpy(buffer_, p, size);
}

CpuBuffer::~CpuBuffer()
{
    release();
}

void CpuBuffer::release()
{
    if (buffer_ != nullptr) {
        std::free(buffer_);
        buffer_ = nullptr;
        size_ = 0;
    }
}

CpuBuffer::CpuBuffer(CpuBuffer&& rhs)
{
    *this = std::move(rhs);
}

CpuBuffer& CpuBuffer::operator=(CpuBuffer&& rhs)
{
    if (this != &rhs) {
        buffer_ = std::move(rhs.buffer_);
        size_ = std::move(rhs.size_);

        rhs.release();
    }

    return *this;
}

}   // px
