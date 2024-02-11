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

#pragma once

#include <boost/uuid/detail/sha1.hpp>

#include "Common.h"

namespace px {

template<typename T>
std::string sha1(const T* x, std::size_t n)
{
    uint32_t digest[5];
    boost::uuids::detail::sha1 sha1;

    sha1.process_bytes(reinterpret_cast<const char*>(x), n * sizeof(T));
    sha1.get_digest(digest);

    std::ostringstream ss;
    ss << std::setfill('0') << std::hex;

    for (uint32_t i: digest) {
        ss << i;
    }

    return ss.str();
}

}   // px
