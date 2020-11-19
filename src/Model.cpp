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
#include "Layer.h"
#include "Model.h"
#include <fstream>

using namespace YAML;

namespace px {

Model::Model(const std::string& filename) : filename_(filename)
{
    parse();
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
    int inputs = height_ * width_ * channels_;

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
        params["inputs"] = inputs;
        params["channels"] = channels;
        params["height"] = height;
        params["width"] = width;

        const auto layer = Layer::create(params);

        channels = layer->outChannels();
        height = layer->outHeight();
        width = layer->outWidth();
        inputs = layer->outputs();

        layer->print(std::cout);

        layers_.emplace_back(std::move(layer));
    }
}

const Model::LayerVec& Model::layers() const
{
    return layers_;
}

const int Model::batch() const
{
    return batch_;
}

const int Model::channels() const
{
    return channels_;
}

const int Model::height() const
{
    return height_;
}

const int Model::width() const
{
    return width_;
}

xt::xarray<float> Model::forward(xt::xarray<float>&& input)
{
    auto in = std::move(input);

    for (auto& layer: layers()) {
        in = layer->forward(std::forward<xt::xarray<float>>(in));
    }

    return in;
}

void Model::loadDarknetWeights(const std::string& filename)
{
    std::ifstream ifs(filename, std::ios::in | std::ios::binary);
    PX_CHECK(ifs.good(), "Could not open file \"%s\".", filename.c_str());

    ifs.seekg(0, ifs.end);
    auto length = ifs.tellg();
    ifs.seekg(0, ifs.beg);

    int major, minor, revision;

    ifs.read((char*) &major, sizeof(int));
    ifs.read((char*) &minor, sizeof(int));
    ifs.read((char*) &revision, sizeof(int));

    if ((major * 10 + minor) >= 2 && major < 1000 && minor < 1000) {
        size_t seen;
        ifs.read((char*) &seen, sizeof(size_t));
    } else {
        int iseen = 0;
        ifs.read((char*) &iseen, sizeof(int));
    }

    for (const auto& layer: layers()) {
        layer->loadDarknetWeights(ifs);
    }

    PX_CHECK(ifs.tellg() == length, "Did not fully read weights file.  Model/Weights mismatch?");

    ifs.close();
}

xt::xarray<float> Model::predict(xt::xarray<float>&& input)
{
    auto result = forward(std::forward<xt::xarray<float>>(input));

    return result;
}

} // px