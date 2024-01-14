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
