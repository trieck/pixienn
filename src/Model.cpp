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
#include "Image.h"
#include "Timer.h"

#include <boost/filesystem.hpp>
#include <fstream>
#include <nlohmann/json.hpp>

using namespace YAML;
using namespace boost::filesystem;
using json = nlohmann::json;

namespace px {

Model::Model(const std::string& cfgFile) : cfgFile_(cfgFile)
{
    parseConfig();
}

void Model::parseConfig()
{
    auto cfgDoc = LoadFile(cfgFile_);

    PX_CHECK(cfgDoc.IsMap(), "Document not a map.");
    PX_CHECK(cfgDoc["configuration"], "Document has no configuration.");

    const auto config = cfgDoc["configuration"];
    PX_CHECK(config.IsMap(), "Configuration is not a map.");

    auto cfgPath = path(cfgFile_);

    auto model = config["model"].as<std::string>();
    modelFile_ = canonical(model, cfgPath.parent_path()).string();
    parseModel();

    auto weights = config["weights"].as<std::string>();
    weightsFile_ = canonical(weights, cfgPath.parent_path()).string();
    loadDarknetWeights();

    auto labels = config["labels"].as<std::string>();
    labelsFile_ = canonical(labels, cfgPath.parent_path()).string();
    loadLabels();
}

void Model::parseModel()
{
    auto modelDoc = LoadFile(modelFile_);

    PX_CHECK(modelDoc.IsMap(), "Model document not a map.");
    PX_CHECK(modelDoc["model"], "Document has no model.");

    const auto model = modelDoc["model"];
    PX_CHECK(model.IsMap(), "Model is not a map.");

    batch_ = model["batch"].as<int>();
    channels_ = model["channels"].as<int>();
    height_ = model["height"].as<int>();
    width_ = model["width"].as<int>();
    int inputs = height_ * width_ * channels_;

    const auto layers = model["layers"];
    PX_CHECK(layers.IsSequence(), "Model has no layers.");

    std::cout << std::setfill('_');

    std::cout << std::setw(21) << std::left << "Layer"
              << std::setw(10) << "Filters"
              << std::setw(20) << "Size"
              << std::setw(20) << "Input"
              << std::setw(20) << "Output"
              << std::endl;

    int channels(channels_), height(height_), width(width_);

    auto index = 0;
    for (const auto& layerDef: layers) {
        YAML::Node params(layerDef);
        params["batch"] = batch_;
        params["index"] = index++;
        params["inputs"] = inputs;
        params["channels"] = channels;
        params["height"] = height;
        params["width"] = width;

        const auto layer = Layer::create(*this, params);

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
    auto& in = input;

    for (auto& layer: layers()) {
        layer->forward(in);
        in = layer->output();
    }

    return in;
}

void Model::loadDarknetWeights()
{
    std::ifstream ifs(weightsFile_, std::ios::in | std::ios::binary);
    PX_CHECK(ifs.good(), "Could not open file \"%s\".", weightsFile_.c_str());

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

std::vector<Detection> Model::predict(const std::string& imageFile, float threshold)
{
    auto image = imread(imageFile.c_str());
    auto sized = imletterbox(image, width(), height());
    auto input = imarray(sized);

    std::cout << "Running network..." << std::endl;

    Timer timer;
    auto result = forward(std::forward<xt::xarray<float>>(input));

    std::vector<Detection> detections;

    for (auto& layer: layers()) {
        auto* detector = dynamic_cast<Detector*>(layer.get());
        if (detector) {
            detector->addDetects(detections, image.cols, image.rows, threshold);
        }
    }

    std::printf("%s: Predicted in %s.\n", imageFile.c_str(), timer.str().c_str());

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

void Model::loadLabels()
{
    std::ifstream ifs(labelsFile_, std::ios::in | std::ios::binary);
    PX_CHECK(ifs.good(), "Could not open file \"%s\".", labelsFile_.c_str());

    labels_.clear();

    for (std::string label; std::getline(ifs, label);) {
        labels_.push_back(label);
    }
}

const std::vector<std::string>& Model::labels() const noexcept
{
    return labels_;
}

std::string Model::asJson(std::vector<Detection>&& detects) const noexcept
{
    auto json = json::object();

    json["type"] = "FeatureCollection";

    auto features = json::array();

    for (const auto& det: detects) {
        auto it = std::max_element(det.prob().cbegin(), det.prob().cend());
        if (*it == 0) {
            continue;   // suppressed
        }

        auto feature = json::object();
        auto geometry = json::object();
        auto props = json::object();
        auto coords = json::array();

        feature["type"] = "Feature";
        geometry["type"] = "Polygon";

        const auto& b = det.box();

        auto left = b.x;
        auto top = -b.y;
        auto right = b.x + b.width;
        auto bottom = -(b.y + b.height);

        coords.emplace_back(json::array({ left, top }));
        coords.emplace_back(json::array({ right, top }));
        coords.emplace_back(json::array({ right, bottom }));
        coords.emplace_back(json::array({ left, bottom }));
        coords.emplace_back(json::array({ left, top }));

        geometry["coordinates"] = json::array({ coords });

        auto index = std::distance(det.prob().cbegin(), it);

        props["top_cat"] = labels_[index];
        props["top_score"] = *it;

        feature["geometry"] = geometry;
        feature["properties"] = props;

        features.emplace_back(std::move(feature));
    }

    json["features"] = features;

    return json.dump(2);
}

} // px