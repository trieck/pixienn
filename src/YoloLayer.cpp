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

#include "YoloLayer.h"

namespace px {

using namespace xt;

YoloLayer::YoloLayer(const Model& model, const YAML::Node& layerDef) : Layer(model, layerDef)
{
    classes_ = property<int>("classes", 0);
    num_ = property<int>("num", 1);

    setOutChannels(num_ * (classes_ + 5));
    setOutHeight(height());
    setOutWidth(width());
    setOutputs(outHeight() * outWidth() * outChannels());
}

std::ostream& YoloLayer::print(std::ostream& os)
{
    os << std::setfill('.');

    os << std::setw(60) << std::left << "yolo"
       << std::setw(20) << std::left
       << std::string(std::to_string(height()) + " x " + std::to_string(width()) + " x " + std::to_string(channels()))
       << std::setw(20) << std::left
       << std::string(
               std::to_string(outHeight()) + " x " + std::to_string(outWidth()) + " x " + std::to_string(outChannels()))
       << std::endl;

    return os;
}

void YoloLayer::forward(const xt::xarray<float>& input)
{
    output_ = input;
}

} // px
