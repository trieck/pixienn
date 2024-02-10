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


#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>

#include "CudaUtils.cuh"

namespace px {

dim3 cudaGridsize(std::uint32_t n)
{
    std::uint32_t k = (n - 1) / CUDA_BLOCK_SIZE + 1;
    std::uint32_t x = k;
    std::uint32_t y = 1;

    if (x > 65535) {
        x = ceil(sqrt(k));
        y = (n - 1) / (x * CUDA_BLOCK_SIZE) + 1;
    }

    dim3 d = { x, y, 1 };

    return d;
}

struct random_generator
{
    __host__ __device__ explicit random_generator(float a = 0.f, float b = 1.f);
    __host__ __device__ float operator()(std::size_t n) const;

    float a_, b_;
};

__host__ __device__ random_generator::random_generator(float a, float b) : a_(a), b_(b)
{
}

__host__ __device__ float random_generator::operator()(std::size_t n) const
{
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> dist(a_, b_);
    rng.discard(n);

    return dist(rng);
}

void fillGpu(float* ptr, std::size_t n, float value)
{
    thrust::device_ptr<float> devPtr = thrust::device_pointer_cast(ptr);
    thrust::fill(devPtr, devPtr + n, value);
}

void fillGpu(int* ptr, std::size_t n, int value)
{
    thrust::device_ptr<int> devPtr = thrust::device_pointer_cast(ptr);
    thrust::fill(devPtr, devPtr + n, value);
}

void randomGpu(float* ptr, std::size_t n, float a, float b)
{
    thrust::device_ptr<float> devPtr = thrust::device_pointer_cast(ptr);

    thrust::counting_iterator<unsigned int> index_sequence_begin(0);
    thrust::transform(index_sequence_begin,
                      index_sequence_begin + n,
                      devPtr,
                      random_generator(a, b));
}

struct ConstrainOp
{
    float alpha;

    __host__ __device__ float operator()(float val) const
    {
        return fminf(alpha, fmaxf(-alpha, val));
    }
};

void constrainGpu(int n, float alpha, float* x)
{
    thrust::device_ptr<float> devPtr = thrust::device_pointer_cast(x);

    ConstrainOp op{alpha};

    thrust::transform(devPtr, devPtr + n, devPtr, op);
}

void mulGpu(int n, float* x, float* y)
{
    thrust::device_ptr<float> devPtrX = thrust::device_pointer_cast(x);
    thrust::device_ptr<float> devPtrY = thrust::device_pointer_cast(y);

    thrust::transform(devPtrX, devPtrX + n, devPtrY, devPtrY, thrust::multiplies<float>());
}

}   // px