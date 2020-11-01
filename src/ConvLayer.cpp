/********************************************************************************
* Copyright 2020 Maxar Technologies Inc.
* Author: Thomas A. Rieck
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
*
* SBIR DATA RIGHTS
* Contract No. HM0476-16-C-0022
* Contractor Name: Radiant Analytic Solutions Inc.
* Contractor Address: 2325 Dulles Corner Blvd. STE 1000, Herndon VA 20171
* Expiration of SBIR Data Rights Period: 2/13/2029
*
* The Government's rights to use, modify, reproduce, release, perform, display,
* or disclose technical data or computer software marked with this legend are
* restricted during the period shown as provided in paragraph (b)(4) of the
* Rights in Noncommercial Technical Data and Computer Software-Small Business
* Innovation Research (SBIR) Program clause contained in the above identified
* contract. No restrictions apply after the expiration date shown above. Any
* reproduction of technical data, computer software, or portions thereof marked
* with this legend must also reproduce the markings.
********************************************************************************/

#include "ConvLayer.h"
#include "xtensor/xrandom.hpp"

using namespace px;
using namespace xt;

ConvLayer::ConvLayer(const YAML::Node& layerDef) : Layer(layerDef)
{
    dilation_ = property<int>("dilation", 0);
    filters_ = property<int>("filters", 1);
    kernel_ = property<int>("kernel", 1);
    pad_ = property<int>("pad", 0);
    stride_ = property<int>("stride", 1);

    weights_ = random::rand<float>({ filters_, 3 /* channels */, kernel_, kernel_ });
    biases_ = zeros<float>({ filters_ });

    setOutChannels(filters_);
    setOutHeight((height() + 2 * pad_ - kernel_) / stride_ + 1);
    setOutWidth((width() + 2 * pad_ - kernel_) / stride_ + 1);
}

ConvLayer::~ConvLayer()
{

}

std::ostream& ConvLayer::print(std::ostream& os)
{
    os << std::setfill('.');

    os << std::setw(20) << std::left << "conv"
       << std::setw(20) << std::left << filters_
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
