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

#include "PxTensor.h"

namespace px {

template<Device D>
struct DeviceTraits;

template<>
struct DeviceTraits<Device::CPU>
{
    using VectorType = PxCpuVector;
    using ValueType = VectorType::value_type;
};

#ifdef USE_CUDA
template<>
struct DeviceTraits<Device::CUDA>
{
    using VectorType = PxCudaVector;
    using ValueType = VectorType::value_type;
};
#endif  // USE_CUDA

template<Device D>
struct IsCudaDevice : std::false_type
{
};

template<>
struct IsCudaDevice<Device::CUDA> : std::true_type
{
};

template<Device D, typename T = void>
using EnableIfCuda = std::enable_if_t<IsCudaDevice<D>::value, T>;

}
