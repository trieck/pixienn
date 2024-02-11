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

#include "Adam.cuh"
#include "Cublas.h"
#include "CudaUtils.cuh"
#include "DeviceTraits.h"

namespace px {

template<Device D>
class Adam
{
public:
    using V = typename DeviceTraits<D>::VectorType;

    Adam(const CublasContext& ctxt, int batch, int t, float lr, float beta1, float beta2, float epsilon, float decay);

    void update(V& w, V& d, V& m, V& v);
private:
    const CublasContext& ctxt_;
    int batch_, t_;
    float lr_, beta1_, beta2_, epsilon_, decay_;
};

template<Device D>
Adam<D>::Adam(const CublasContext& ctxt, int batch, int t, float lr, float beta1, float beta2, float epsilon,
              float decay) : ctxt_(ctxt), batch_(batch), t_(t), lr_(lr), beta1_(beta1), beta2_(beta2),
                             epsilon_(epsilon), decay_(decay)
{
    static_assert(IsCudaDevice<D>::value, "Adam only supports CUDA");
}

template<Device D>
void Adam<D>::update(V& w, V& d, V& m, V& v)
{
    // Scale first moment vector 'm' by beta1_
    auto status = cublasSscal(ctxt_, m.size(), &beta1_, m.data(), 1);
    PX_CHECK_CUBLAS(status);

    // Scale second moment vector 'v' by beta2_
    status = cublasSscal(ctxt_, v.size(), &beta2_, v.data(), 1);
    PX_CHECK_CUBLAS(status);

    // Compute the update term for the weights
    auto alpha = -decay_ * batch_;  // Compute the decay term
    status = cublasSaxpy(ctxt_, w.size(), &alpha, w.data(), 1, d.data(), 1);
    PX_CHECK_CUBLAS(status);

    // Update the first moment vector 'm' with the gradient 'd' using Adam's formula
    // m = m + (1 - beta1_) * d
    alpha = 1.0f - beta1_;
    status = cublasSaxpy(ctxt_, m.size(), &alpha, d.data(), 1, m.data(), 1);

    // Compute the square of gradient vector 'd' in place
    mulGpu(d.size(), d.data(), d.data());

    // Update the second moment vector 'v' with the squared gradient 'd' using Adam's formula
    // v = v + (1 - beta2_) * (d * d)
    alpha = 1.0f - beta2_;
    status = cublasSaxpy(ctxt_, m.size(), &alpha, d.data(), 1, v.data(), 1);

    // Perform the Adam update on the weights 'w' using 'm', 'v', and other parameters
    adamGpu(w.size(), w.data(), m.data(), v.data(), beta1_, beta2_, lr_, epsilon_, t_);

    // Reset the gradient vector 'd' to zeros for the next iteration
    d.fill(0);
}

}   // px