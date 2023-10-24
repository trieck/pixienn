/********************************************************************************
* Copyright 2023 Maxar Technologies Inc.
* Author: Thomas A. Rieck
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
*
* SBIR DATA RIGHTS
* Contract No. HM0476-16-C-0022
* Contractor Name: Radiant Analytic Solutions Inc.
* Contractor Address: 2325 Dulles Corner Blvd. STE 1000, Herndon VA 20171
* Expiration of SBIR Data Rights Period: 2/13/2029
*
* The Government's rights to use, modify, reproduce, release, perform, display,
* or disclose technical data or computer software marked with this legend are
* restricted during the period shown as provided in paragraph (b)(4) of the
* Rights in Noncommercial Technical Data and Computer Software-Small Business
* Innovation Research (SBIR) Program clause contained in the above identified
* contract. No restrictions apply after the expiration date shown above. Any
* reproduction of technical data, computer software, or portions thereof marked
* with this legend must also reproduce the markings.
********************************************************************************/

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>

namespace px {

struct random_generator
{
    __host__ __device__ random_generator(float a = 0.f, float b = 1.f);
    __host__ __device__ float operator()(const unsigned int n) const;

    float a_, b_;
};

__host__ __device__ random_generator::random_generator(float a, float b) : a_(a), b_(b)
{
}

__host__ __device__ float random_generator::operator()(const unsigned int n) const
{
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> dist(a_, b_);
    rng.discard(n);

    return dist(rng);
}

void fill_gpu(float* ptr, std::size_t N, float value)
{
    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(ptr);
    thrust::fill(dev_ptr, dev_ptr + N, value);
}

void random_generate(float* ptr, std::size_t N, float a, float b)
{
    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(ptr);

    thrust::counting_iterator<unsigned int> index_sequence_begin(0);
    thrust::transform(index_sequence_begin,
                      index_sequence_begin + N,
                      dev_ptr,
                      random_generator(a, b));
}

}   // px