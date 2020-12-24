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

#include "Cudnn.h"
#include "Tensor.h"

namespace px {

// FIXME: make use device tensors
cpu_array convolve_gpu(const cpu_array& input, const cpu_tensor<4>& weights, int padding, int stride,
                       int dilation)
{
    CudnnContext context;

    int n = input.shape(0);
    int c = input.shape(1);
    int h = input.shape(2);
    int w = input.shape(3);

    CudnnTensorDesc xDesc;
    auto status = cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
    PX_CHECK_CUDNN(status);

    n = weights.shape(0);
    c = weights.shape(1);
    h = weights.shape(2);
    w = weights.shape(3);

    CudnnFilterDesc wDesc;
    status = cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w);
    PX_CHECK_CUDNN(status);

    CudnnConvDesc convDesc;
    status = cudnnSetConvolution2dDescriptor(convDesc,
                                             padding,
                                             padding,
                                             stride,
                                             stride,
                                             dilation,
                                             dilation,
                                             CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    PX_CHECK_CUDNN(status);

    status = cudnnGetConvolution2dForwardOutputDim(convDesc, xDesc, wDesc, &n, &c, &h, &w);
    PX_CHECK_CUDNN(status);

    CudnnTensorDesc yDesc;
    status = cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
    PX_CHECK_CUDNN(status);

    int retCount;
    cudnnConvolutionFwdAlgoPerf_t fwdAlgoPerf;

    status = cudnnGetConvolutionForwardAlgorithm_v7(context, xDesc, wDesc, convDesc, yDesc, 1, &retCount, &fwdAlgoPerf);
    PX_CHECK_CUDNN(status);
    PX_CHECK_CUDNN(fwdAlgoPerf.status);

    using tensor4d = cuda_tensor<4>;

    cuda_array X(input);
    tensor4d W(weights);
    tensor4d Y = decltype(Y)::from_shape({ ulong(n), ulong(c), ulong(h), ulong(w) });
    cuda_tensor_t<uint8_t, 1> ws = decltype(ws)::from_shape({ fwdAlgoPerf.memory });

    float one = 1;

    status = cudnnConvolutionForward(context,
                                     &one,
                                     xDesc,
                                     X.data().get(),
                                     wDesc,
                                     W.data().get(),
                                     convDesc,
                                     fwdAlgoPerf.algo,
                                     ws.data().get(),
                                     fwdAlgoPerf.memory,
                                     &one,
                                     yDesc,
                                     Y.data().get());

    PX_CHECK_CUDNN(status);

    cpu_array output(Y);   // weird

    return output;
}

}   // namespace px
