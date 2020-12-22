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

#include "Error.h"
#include "Tensor.cuh"

#include <Cudnn.h>

namespace px {

template<typename T, Device D, typename B>
static void inline print(tensor<T, D, B>&& t)
{
    int i = 0;
    for (const auto& v: t) {
        std::cout << ++i << "    " << v << std::endl;
    }
}

void foobar()
{
    CudnnContext context;

    CudnnTensorDesc xDesc;
    auto status = cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 3, 224, 224);
    PX_CHECK_CUDNN(status);

    CudnnFilterDesc wDesc;
    status = cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 32, 3, 3, 3);
    PX_CHECK_CUDNN(status);

    CudnnConvDesc convDesc;
    status = cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    PX_CHECK_CUDNN(status);

    int n, c, h, w;
    status = cudnnGetConvolution2dForwardOutputDim(convDesc, xDesc, wDesc, &n, &c, &h, &w);
    PX_CHECK_CUDNN(status);

    CudnnTensorDesc yDesc;
    status = cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
    PX_CHECK_CUDNN(status);

    std::cout << n << ", " << c << ", " << h << ", " << w << std::endl;

    int retCount;
    cudnnConvolutionFwdAlgoPerf_t fwdAlgoPerf;

    status = cudnnGetConvolutionForwardAlgorithm_v7(context, xDesc, wDesc, convDesc, yDesc, 1, &retCount, &fwdAlgoPerf);
    PX_CHECK_CUDNN(status);
    PX_CHECK_CUDNN(fwdAlgoPerf.status);

    std::cout << fwdAlgoPerf.memory << std::endl;

    using tensor4d = cuda_tensor<float, 4>;

    tensor4d X = tensor4d::from_shape({1, 3, 224, 224});
    tensor4d W = tensor4d::from_shape({32, 3, 3, 3});
    tensor4d Y = tensor4d::from_shape({ulong(n), ulong(c), ulong(h), ulong(w)});
    cuda_tensor<uint8_t, 1> workspace = cuda_tensor<uint8_t, 1>::from_shape({fwdAlgoPerf.memory});

    void* x = X.data().get();
    void* weights = W.data().get();
    void* wp = workspace.data().get();
    void* y = Y.data().get();

    float one = 1;

    status = cudnnConvolutionForward(context,
                                     &one,
                                     xDesc,
                                     x,
                                     wDesc,
                                     weights,
                                     convDesc,
                                     fwdAlgoPerf.algo,
                                     wp,
                                     fwdAlgoPerf.memory,
                                     &one,
                                     yDesc,
                                     y);

    PX_CHECK_CUDNN(status);

    cuda_tensor<float, 1> T = decltype(T)::random({10});

    for (const auto& v: T) {
        std::cout << v << std::endl;
    }
}

}   // px


