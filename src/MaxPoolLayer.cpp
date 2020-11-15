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

#include "MaxPoolLayer.h"

namespace px {

MaxPoolLayer::MaxPoolLayer(const YAML::Node& layerDef) : Layer(layerDef)
{
    kernel_ = property<int>("kernel", 1);
    stride_ = property<int>("stride", 1);
    size_ = property<int>("size", stride_);
    padding_ = property<int>("padding", std::max(0, size_ - 1));

    setOutChannels(channels());
    setOutHeight((height() + padding_ - size_) / stride_ + 1);
    setOutWidth((width() + padding_ - size_) / stride_ + 1);

    setOutputs(outHeight() * outWidth() * outChannels());
}

MaxPoolLayer::~MaxPoolLayer()
{

}

std::ostream& MaxPoolLayer::print(std::ostream& os)
{
    os << std::setfill('.');

    os << std::setw(40) << std::left << "max"
       << std::setw(20) << std::left
       << std::string(std::to_string(size_) + " x " + std::to_string(size_) + " / " + std::to_string(stride_))
       << std::setw(20) << std::left
       << std::string(std::to_string(channels()) + " x " + std::to_string(height()) + " x " + std::to_string(width()))
       << std::setw(20) << std::left
       << std::string(
               std::to_string(outChannels()) + " x " + std::to_string(outHeight()) + " x " + std::to_string(outWidth()))
       << std::endl;

    return os;
}

xt::xarray<float> MaxPoolLayer::forward(const xt::xarray<float>& input)
{
    return xt::xarray<float>();
}

} // px
