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

#ifndef PIXIENN_ACTIVATION_H
#define PIXIENN_ACTIVATION_H

#include "Common.h"

#ifdef USE_CUDA

#include "PxVector.h"

#else
#include <xtensor/xcontainer.hpp>
#endif
namespace px {

class Activation
{
public:
    using Ptr = std::shared_ptr<Activation>;

    static Activation::Ptr get(const std::string& s);

#ifdef USE_CUDA
    virtual void apply_gpu(float* begin, std::size_t n) const = 0;
#else
    virtual void apply(float* begin, float* end) const = 0;
#endif

    template<typename T>
#ifdef USE_CUDA
    void apply(PxDevVector<T>&) const;
#else
    template<typename T>
    void apply(xt::xcontainer<T>&) const;
#endif
};

#ifdef USE_CUDA
template<typename T>
void Activation::apply(PxDevVector<T>& vec) const
{
    apply_gpu(vec.data(), vec.size());
}

#else
template<typename T>
void Activation::apply(xt::xcontainer<T>& container) const
{
    apply(container.begin(), container.end());
}
#endif // USE_CUDA

}   // px

#endif // PIXIENN_ACTIVATION_H
