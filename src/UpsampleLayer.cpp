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

#include "UpsampleLayer.h"
#include <opencv2/imgproc.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xtensor.hpp>


#include <opencv2/core/mat.hpp>

using namespace cv;

using namespace cv;
using namespace xt;

namespace px {

UpsampleLayer::UpsampleLayer(const Model& model, const YAML::Node& layerDef) : Layer(model, layerDef),
                                                                               stride_(0), scale_(0)
{
}

void UpsampleLayer::setup()
{
    scale_ = property("scale", 1.0f);
    stride_ = property("stride", 2);    // FIXME: does not support negative stride (reverse upsample)

    setInterpolationFlags();
    setOutChannels(channels());
    setOutHeight(height() * stride_);
    setOutWidth(width() * stride_);
    setOutputs(outHeight() * outWidth() * outChannels());

    output_ = empty<float>({ batch(), outChannels(), outHeight(), outWidth() });

#ifdef USE_CUDA
    if (useGpu()) {
        outputGpu_ = PxDevVector<float>(batch() * outChannels() * outHeight() * outWidth());
    }
#endif
}

std::ostream& UpsampleLayer::print(std::ostream& os)
{
    Layer::print(os, "upsample", { height(), width(), channels() }, { outHeight(), outWidth(), outChannels() });

    return os;
}

void UpsampleLayer::forward(const xarray<float>& input)
{
    for (auto b = 0; b < batch(); ++b) {
        auto* pinput = input.data() + b * inputs();
        auto* poutput = output_.data() + b * outputs();

        Mat mInput(height(), width(), CV_32FC(channels()), (void*) pinput, cv::Mat::AUTO_STEP);
        Mat mOutput(outHeight(), outWidth(), CV_32FC(outChannels()), (void*) poutput, cv::Mat::AUTO_STEP);

        resize(mInput, mOutput, { (int) outWidth(), (int) outHeight() }, scale_, scale_, flags_);
    }
}

#ifdef USE_CUDA

void UpsampleLayer::forwardGpu(const PxDevVector<float>& input)
{
    // FIXME: not implemented on device

    xarray<float> hostInput = adapt(input.asHost());
    forward(hostInput);
    outputGpu_.fromHost(output_);


    /*
     *
     *
     * fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);

     * __global__ void upsample_kernel(size_t N, float *x, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    size_t i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int out_index = i;
    int out_w = i%(w*stride);
    i = i/(w*stride);
    int out_h = i%(h*stride);
    i = i/(h*stride);
    int out_c = i%c;
    i = i/c;
    int b = i%batch;

    int in_w = out_w / stride;
    int in_h = out_h / stride;
    int in_c = out_c;

    int in_index = b*w*h*c + in_c*w*h + in_h*w + in_w;


    if(forward) out[out_index] += scale * x[in_index];
    else atomicAdd(x+in_index, scale * out[out_index]);
}
extern "C" void upsample_gpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    size_t size = w*h*c*batch*stride*stride;
    upsample_kernel<<<cuda_gridsize(size), BLOCK>>>(size, in, w, h, c, batch, stride, forward, scale, out);
    check_error(cudaPeekAtLastError());
}

     */
    /*for (auto b = 0; b < batch(); ++b) {
        auto* pinput = input.data() + b * inputs();
        auto* poutput = outputGpu_.data() + b * outputs();

        cv::cuda::GpuMat mInput(height(), width(), CV_32FC(channels()), (void*) pinput, cv::Mat::AUTO_STEP);
        cv::cuda::GpuMat mOutput(outHeight(), outWidth(), CV_32FC(outChannels()), (void*) poutput, cv::Mat::AUTO_STEP);

        resize(mInput, mOutput, { (int)outWidth(), (int)outHeight() }, scale_, scale_, flags_);
    }*/
}

#endif  // USE_CUDA

void UpsampleLayer::setInterpolationFlags()
{
    auto method = property<std::string>("interpolation", "nearest");

    if (method == "nearest") {
        flags_ = InterpolationFlags::INTER_NEAREST;
    } else if (method == "linear") {
        flags_ = InterpolationFlags::INTER_LINEAR;
    } else if (method == "linear_exact") {
        flags_ = InterpolationFlags::INTER_LINEAR_EXACT;
    } else if (method == "cubic") {
        flags_ = InterpolationFlags::INTER_CUBIC;
    } else if (method == "area") {
        flags_ = InterpolationFlags::INTER_AREA;
    } else {
        PX_ERROR_THROW("Unsupported interpolation method \"%s\".", method.c_str());
    }
}

} // px
