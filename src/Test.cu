
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/system_error.h>

void foo()
{
    size_t N = 10;

    int* raw_ptr;
    cudaMalloc((void**) &raw_ptr, N * sizeof(int));

    thrust::device_ptr<int> dev_ptr(raw_ptr);
    thrust::fill(dev_ptr, dev_ptr + N, (int) 0);
}

