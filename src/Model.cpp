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

    std::cout << std::setw(26) << std::left << "Layer"
              << std::setw(20) << "Filters"
              << std::setw(20) << "Size"
              << std::setw(20) << "Input"
              << std::setw(20) << "Output"
              << std::endl;

    int channels(channels_), height(height_), width(width_);

    auto index = 0;
    for (const auto& layerDef: layers) {
        YAML::Node params(layerDef);
        params["batch"] = batch_;
        params["index"] = index;
        params["inputs"] = inputs;
        params["channels"] = channels;
        params["height"] = height;
        params["width"] = width;

        const auto layer = Layer::create(*this, params);

        channels = layer->outChannels();
        height = layer->outHeight();
        width = layer->outWidth();
        inputs = layer->outputs();

        std::cout << std::setfill(' ') << std::setw(5) << std::right << index++ << ' ';

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
    auto& in = input;

    for (auto& layer: layers()) {
        layer->forward(in);
        in = layer->output();
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

    ifs.read((char*) &major_, sizeof(int));
    ifs.read((char*) &minor_, sizeof(int));
    ifs.read((char*) &revision_, sizeof(int));

    if ((major_ * 10 + minor_) >= 2 && major_ < 1000 && minor_ < 1000) {
        size_t seen;
        ifs.read((char*) &seen, sizeof(size_t));
    } else {
        int iseen = 0;
        ifs.read((char*) &iseen, sizeof(int));
    }

    std::streamoff pos = ifs.tellg();
    for (const auto& layer: layers()) {
        pos += layer->loadDarknetWeights(ifs);
    }

    PX_CHECK(pos == length, "Did not fully read weights file; read %ld bytes, expected to read %ld bytes.",
             pos, length);

    ifs.close();
}

std::vector<Detection> Model::predict(xt::xarray<float>&& input, int width, int height, float threshold)
{
    auto result = forward(std::forward<xt::xarray<float>>(input));

    std::vector<Detection> detections;

    for (auto& layer: layers()) {
        auto* detector = dynamic_cast<Detector*>(layer.get());
        if (detector) {
            detector->addDetects(detections, width, height, threshold);
        }
    }

    return detections;
}

const int Model::layerSize() const
{
    return layers_.size();
}

const Layer::Ptr& Model::layerAt(int index) const
{
    PX_CHECK(index < layers_.size(), "Index out of range.");

    return layers_[index];
}

} // px