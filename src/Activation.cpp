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

#include "Activation.h"
#include "Singleton.h"
#include "Error.h"

#include <cmath>

namespace px {

using namespace px;

class Activations : public Singleton<Activations>
{
public:
    Activations();


    bool hasActivation(const std::string& s) const;
    Activation::Ptr at(const std::string& s) const;

private:
    std::unordered_map<std::string, Activation::Ptr> activations_;
};

class LinearActivation : public Activation
{
public:
    void apply(float* begin, float* end) const override
    {
        std::for_each(begin, end, [](float& x) {});
    }
};

class LeakyActivation : public Activation
{
public:
    void apply(float* begin, float* end) const override
    {
        std::for_each(begin, end, [](float& x) {
            x = (x > 0) ? x : .1f * x;
        });
    }
};

class LoggyActivation : public Activation
{
public:
    void apply(float* begin, float* end) const override
    {
        std::for_each(begin, end, [](float& x) {
            x = 2.f / (1.f + std::exp(-x)) - 1;
        });
    }
};

class LogisticActivation : public Activation
{
public:
    void apply(float* begin, float* end) const override
    {
        std::for_each(begin, end, [](float& x) {
            x = 1.f / (1.f + std::exp(-x));
        });
    }
};

class ReluActivation : public Activation
{
public:
    void apply(float* begin, float* end) const override
    {
        std::for_each(begin, end, [](float& x) {
            x = x * (x > 0);
        });
    }
};

Activations::Activations()
{
    activations_ = {
            { "leaky",    std::make_shared<LeakyActivation>() },
            { "linear",   std::make_shared<LinearActivation>() },
            { "loggy",    std::make_shared<LoggyActivation>() },
            { "logistic", std::make_shared<LogisticActivation>() },
            { "relu",     std::make_shared<ReluActivation>() },
    };
}

bool Activations::hasActivation(const std::string& s) const
{
    return activations_.find(s) != activations_.end();
}

Activation::Ptr Activations::at(const std::string& s) const
{
    return activations_.at(s);
}

Activation::Ptr Activation::get(const std::string& s)
{
    auto& activations = Activations::instance();

    PX_CHECK(activations.hasActivation(s), "Cannot find activation type \"%s\".", s.c_str());

    return activations.at(s);
}

}   // px
