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

#include "Error.h"
#include "Model.h"
#include "Layer.h"

using namespace px;
using namespace YAML;

Model::Model(const std::string& filename) : filename_(filename)
{
    parse();
}

Model::Ptr Model::create(const std::string& filename)
{
    return std::shared_ptr<Model>(new Model(filename));
}

void Model::parse()
{
    auto config = LoadFile(filename_);

    PX_CHECK(config.IsMap(), "Model config not a map.");
    PX_CHECK(config["model"], "Config has no model.");

    const auto model = config["model"];
    PX_CHECK(model.IsMap(), "Model is not a map.");

    batch_ = model["batch"].as<int>();
    channels_ = model["channels"].as<int>();
    height_ = model["height"].as<int>();
    width_ = model["width"].as<int>();

    const auto layers = model["layers"];
    PX_CHECK(layers.IsSequence(), "Model has no layers.");

    std::cout << std::setfill('_');

    std::cout << std::setw(20) << std::left << "Layer"
              << std::setw(20) << "Filters"
              << std::setw(20) << "Size"
              << std::setw(20) << "Input"
              << std::setw(20) << "Output"
              << std::endl;

    int channels(channels_), height(height_), width(width_);
    for (const auto& layerDef: layers) {
        YAML::Node params(layerDef);
        params["batch"] = batch_;
        params["channels"] = channels;
        params["height"] = height;
        params["width"] = width;

        const auto layer = Layer::create(params);
        channels = layer->outChannels();
        height = layer->outHeight();
        width = layer->outWidth();

        layer->print(std::cout);

        layers_.emplace_back(std::move(layer));
    }
}



