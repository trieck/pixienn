/********************************************************************************
* Copyright 2023-2023 Thomas A. Rieck, All Rights Reserved
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

#ifndef PIXIENN_TENSORUTILS_H
#define PIXIENN_TENSORUTILS_H

#include "PxTensor.h"

namespace px {

template<typename T, std::size_t N>
void printTensor(const typename T::value_type* data, const std::array<std::int32_t, N>& shape, std::size_t level,
                 std::ostream& os)
{
    if (level >= N) {
        return;
    }

    os << "[ ";

    for (std::size_t i = 0; i < shape[level]; ++i) {
        if (level == N - 1) {
            os << std::fixed << std::setprecision(4) << data[i];
        } else {
            printRecursive(data + i * shape[level + 1], shape, level + 1, os);
        }

        if (i < shape[level] - 1) {
            os << ", ";
        }
    }

    os << " ]";
}

template<typename T, std::size_t N>
void printTensor(const T& tensor, const std::array<std::int32_t, N>& shape, const char* filename)
{
    std::size_t totalSize = 1;
    for (std::size_t dim: shape) {
        totalSize *= dim;
    }

    PX_CHECK(totalSize == tensor.size(), "Shape size does not match the product of its dimensions.");

    std::ofstream outFile(filename);
    PX_CHECK(outFile.is_open(), "Unable to open file \"%s\" for writing.", filename);

    printTensor(tensor.data(), shape, 0, outFile);
}


}   // px

#endif // PIXIENN_TENSORUTILS_H
