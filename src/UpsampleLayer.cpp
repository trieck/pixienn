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
#include <cv.hpp>
#include <opencv2/core/mat.hpp>
#include <xtensor/xtensor.hpp>

namespace px {

using namespace xt;
using namespace cv;

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
}

std::ostream& UpsampleLayer::print(std::ostream& os)
{
    Layer::print(os, "upsample", { height(), width(), channels() }, { outHeight(), outWidth(), outChannels() });

    return os;
}

void UpsampleLayer::forward(const xt::xarray<float>& input)
{
    for (auto b = 0; b < batch(); ++b) {
        auto* pinput = input.data() + b * inputs();
        auto* poutput = output_.data() + b * outputs();

        Mat mInput(height(), width(), CV_32FC(channels()), (void*) pinput, cv::Mat::AUTO_STEP);
        Mat mOutput(outHeight(), outWidth(), CV_32FC(outChannels()), (void*) poutput, cv::Mat::AUTO_STEP);

        resize(std::move(mInput), std::move(mOutput), { outWidth(), outHeight() }, scale_, scale_, flags_);
    }
}

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
