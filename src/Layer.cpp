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

#include "Layer.h"
#include "ConvLayer.h"
#include <Error.h>

using namespace px;

Layer::Layer(const YAML::Node& layerDef) : layerDef_(layerDef)
{
}

Layer::~Layer()
{
}

Layer::Ptr Layer::create(const YAML::Node& layerDef)
{
    PX_CHECK(layerDef.IsMap(), "Layer definition is not a map.");

    const auto type = layerDef["type"].as<std::string>();

    if (type == "conv") {
        return std::shared_ptr<Layer>(new ConvLayer(layerDef));
    }

    return px::Layer::Ptr();
}

