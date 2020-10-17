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

#ifndef PIXIENN_LAYER_H
#define PIXIENN_LAYER_H

#include "common.h"
#include "Error.h"

#include <yaml-cpp/yaml.h>

PX_BEGIN

class Layer
{
protected:
    Layer(const YAML::Node& layerDef);

public:
    virtual ~Layer() = 0;
    using Ptr = std::shared_ptr<Layer>;

    static Layer::Ptr create(const YAML::Node& layerDef);

protected:
    template<typename T>
    T property(const std::string& prop) const;

private:
    YAML::Node layerDef_;
};

template<typename T>
T Layer::property(const std::string& prop) const
{
    const auto node = layerDef_[prop];

    PX_CHECK(node.IsDefined() && !node.IsNull(), "Layer has no property named \"%s\".", prop.c_str());

    return node.as<T>();
}

PX_END

#endif // PIXIENN_LAYER_H
