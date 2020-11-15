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

#include "DetectLayer.h"

using namespace px;

DetectLayer::DetectLayer(const YAML::Node& layerDef) : Layer(layerDef)
{
    classScale = property<float>("class_scale", 1.0f);
    classes_ = property<int>("classes", 1);
    coordScale_ = property<float>("coord_scale", 1.0f);
    coords_ = property<int>("coords", 1);
    forced_ = property<bool>("forced", false);
    jitter_ = property<float>("jitter", 0.2f);
    maxBoxes_ = property<int>("max_boxes", 90);
    noObjectScale_ = property<float>("noobject_scale", 1.0f);
    num_ = property<int>("num", 1);
    objectScale_ = property<float>("object_scale", 1.0f);
    random_ = property<bool>("random", false);
    reorg_ = property<bool>("reorg", false);
    rescore_ = property<bool>("rescore", false);
    side_ = property<int>("side", 7);
    softmax_ = property<bool>("softmax", false);
    sqrt_ = property<bool>("sqrt", false);
}

std::ostream& DetectLayer::print(std::ostream& os)
{
    os << std::setfill('.');

    os << std::setw(100) << std::left << "detection" << std::endl;

    return os;
}

xt::xarray<float> DetectLayer::forward(const xt::xarray<float>& input)
{
    return xt::xarray<float>();
}

