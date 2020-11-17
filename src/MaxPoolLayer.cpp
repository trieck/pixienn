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
#include "xtensor/xbuilder.hpp"

namespace px {

using namespace xt;

MaxPoolLayer::MaxPoolLayer(const YAML::Node& layerDef) : Layer(layerDef)
{
    kernel_ = property<int>("kernel", 1);
    stride_ = property<int>("stride", 1);
    padding_ = property<int>("padding", std::max(0, kernel_ - 1));

    setOutChannels(channels());
    setOutHeight((height() + padding_ - kernel_) / stride_ + 1);
    setOutWidth((width() + padding_ - kernel_) / stride_ + 1);

    setOutputs(outHeight() * outWidth() * outChannels());

    output_ = empty<float>({ batch(), outChannels(), outHeight(), outWidth() });
}

std::ostream& MaxPoolLayer::print(std::ostream& os)
{
    os << std::setfill('.');

    os << std::setw(40) << std::left << "max"
       << std::setw(20) << std::left
       << std::string(std::to_string(kernel_) + " x " + std::to_string(kernel_) + " / " + std::to_string(stride_))
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
    int wOffset = -padding_ / 2;
    int hOffset = -padding_ / 2;

    auto ih = height();
    auto iw = width();
    auto oh = outHeight();
    auto ow = outWidth();
    auto c = channels();
    const auto min = -std::numeric_limits<float>::max();

    const auto* pin = input.data();
    auto* pout = output_.data();

    for (auto b = 0; b < batch(); ++b) {
        for (auto k = 0; k < c; ++k) {
            for (auto i = 0; i < oh; ++i) {
                for (auto j = 0; j < ow; ++j) {
                    auto outIndex = j + ow * (i + oh * (k + c * b));
                    float max = min;
                    for (auto n = 0; n < kernel_; ++n) {
                        for (auto m = 0; m < kernel_; ++m) {
                            auto curH = hOffset + i * stride_ + n;
                            auto curW = wOffset + j * stride_ + m;
                            auto index = curW + iw * (curH + ih * (k + b * c));
                            auto valid = (curH >= 0 && curH < ih && curW >= 0 && curW < iw);
                            auto val = valid ? pin[index] : min;
                            max = (val > max) ? val : max;
                        }
                    }

                    pout[outIndex] = max;
                }
            }
        }
    }



    return output_;
}

} // px
