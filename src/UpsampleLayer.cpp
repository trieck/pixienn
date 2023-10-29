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
#include <xtensor/xtensor.hpp>

#ifdef USE_CUDA

#include <opencv2/core/cuda.hpp>

using namespace cv::cuda;
#else
#include <opencv2/core/mat.hpp>
using namespace cv;
#endif

using namespace cv;
using namespace xt;

namespace px {


UpsampleLayer::UpsampleLayer(const Model& model, const YAML::Node& layerDef) : Layer(model, layerDef)
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
    outputGpu_ = PxDevVector<float>(batch() * outChannels() * outHeight() * outWidth());
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

        resize(mInput, mOutput, { outWidth(), outHeight() }, scale_, scale_, flags_);
    }
}

#ifdef USE_CUDA

void UpsampleLayer::forwardGpu(const PxDevVector<float>& input)
{
    for (auto b = 0; b < batch(); ++b) {
        auto* pinput = input.data() + b * inputs();
        auto* poutput = outputGpu_.data() + b * outputs();

        cv::cuda::GpuMat mInput(height(), width(), CV_32FC(channels()), (void*) pinput, cv::Mat::AUTO_STEP);
        cv::cuda::GpuMat mOutput(outHeight(), outWidth(), CV_32FC(outChannels()), (void*) poutput, cv::Mat::AUTO_STEP);

        resize(mInput, mOutput, { outWidth(), outHeight() }, scale_, scale_, flags_);
    }
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
