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

#ifndef PIXIENN_ACTIVATION_H
#define PIXIENN_ACTIVATION_H

#include "Common.h"
#include "PxTensor.h"

namespace px {

class Activation
{
public:
    using Ptr = std::shared_ptr<Activation>;

    static Activation::Ptr get(const std::string& s);

    virtual float apply(float x) const = 0;
    virtual void apply(float* begin, float* end) const = 0;

    virtual float gradient(float x) const = 0;
    virtual void gradient(float* dbegin, float* dend, const float* x) const = 0;

    float operator()(float x) const;

    void apply(PxCpuVector& container) const;
    void gradient(const PxCpuVector& container, PxCpuVector& delta) const;

#ifdef USE_CUDA
    virtual void applyGpu(float* begin, std::size_t n) const = 0;
    void applyGpu(PxCudaVector&) const;
#endif // USE_CUDA
};

}   // px

#endif // PIXIENN_ACTIVATION_H
