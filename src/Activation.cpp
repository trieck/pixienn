/********************************************************************************
* Copyright 2020-2023 Thomas A. Rieck, All Rights Reserved
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

#include "Activation.h"

namespace px {

static std::unordered_map<std::string, Activations::Ptr> activations = {
        { "leaky",    std::make_shared<LeakyActivation>() },
        { "linear",   std::make_shared<LinearActivation>() },
        { "loggy",    std::make_shared<LoggyActivation>() },
        { "logistic", std::make_shared<LogisticActivation>() },
        { "relu",     std::make_shared<ReLUActivation>() },
};

Activations::Ptr Activations::get(const std::string& name)
{
    const auto& This = Activations::instance();

    PX_CHECK(This.hasActivation(name), "Cannot find activation type \"%s\".", name.c_str());

    return This.at(name);
}

bool Activations::hasActivation(const std::string& s) const
{
    return activations.find(s) != activations.end();
}

Activations::Ptr Activations::at(const std::string& s) const
{
    return activations.at(s);
}

}   // px

