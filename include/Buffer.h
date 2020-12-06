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

#ifndef PIXIENN_BUFFER_H
#define PIXIENN_BUFFER_H

#include <cstddef>
#include <memory>

namespace px {

class Buffer
{
public:
    Buffer();
    Buffer(size_t size);
    virtual ~Buffer() = 0;

    const void* buffer() const noexcept;
    size_t size() const noexcept;

    using Ptr = std::unique_ptr<Buffer>;

protected:
    void* buffer_;
    size_t size_;
};

}   // px

#endif // PIXIENN_BUFFER_H
