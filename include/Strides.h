/********************************************************************************
* Copyright 2023 Thomas A. Rieck, All Rights Reserved
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

#ifndef PIXIENN_STRIDES_H
#define PIXIENN_STRIDES_H

namespace px {

template<class shape_type, class strides_type>
inline void adapt_strides(const shape_type& shape, strides_type& strides, typename strides_type::size_type i) noexcept
{
    if (shape[i] == 1) {
        strides[i] = 0;
    }
}

template<typename shape_type, typename strides_type>
inline std::size_t compute_strides(const shape_type& shape, strides_type& strides)
{
    using strides_value_type = typename std::decay_t<strides_type>::value_type;
    strides_value_type data_size = 1;

    for (auto i = shape.size(); i != 0; --i) {
        strides[i - 1] = data_size;
        data_size = strides[i - 1] * static_cast<strides_value_type>(shape[i - 1]);
        adapt_strides(shape, strides, i - 1);
    }

    return static_cast<std::size_t>(data_size);
}

}   // px

#endif // PIXIENN_STRIDES_H
