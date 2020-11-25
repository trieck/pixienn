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

namespace px {

using namespace xt;

UpsampleLayer::UpsampleLayer(const Model& model, const YAML::Node& layerDef) : Layer(model, layerDef)
{
    stride_ = property<int>("stride", 2);

    setOutChannels(channels());
    setOutHeight(height() * stride_);
    setOutWidth(width() * stride_);
    setOutputs(outHeight() * outWidth() * outChannels());
}

std::ostream& UpsampleLayer::print(std::ostream& os)
{
    os << std::setfill('.');

    os << std::setw(60) << std::left << "upsample"
       << std::setw(20) << std::left
       << std::string(std::to_string(height()) + " x " + std::to_string(width()) + " x " + std::to_string(channels()))
       << std::setw(20) << std::left
       << std::string(
               std::to_string(outHeight()) + " x " + std::to_string(outWidth()) + " x " + std::to_string(outChannels()))
       << std::endl;

    return os;
}

void UpsampleLayer::forward(const xt::xarray<float>& input)
{
    output_ = input;
}

} // px
