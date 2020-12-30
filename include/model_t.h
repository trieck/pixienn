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

#ifndef PIXIENN_MODEL_T_H
#define PIXIENN_MODEL_T_H

#include "Detection.h"
#include "Image.h"
#include "Timer.h"
#include "layer_t.h"

#include <boost/filesystem.hpp>

namespace px {

template<typename T=cpu_array>
class model_t
{
public:
    model_t(std::string cfgFile);
    model_t(const model_t& rhs) = default;
    model_t(model_t&& rhs) = default;

    model_t& operator=(const model_t& rhs) = default;
    model_t& operator=(model_t&& rhs) = default;

    using layer = typename layer_t<T>::Ptr;
    using layer_vec = std::vector<layer>;
    const layer_vec& layers() const;

    int batch() const;
    int channels() const;
    int height() const;
    int width() const;

    int layerSize() const;
    const layer& layerAt(int index) const;

    std::vector<Detection> predict(const std::string& imageFile, float threshold);
    std::string asJson(std::vector<Detection>&& detects) const noexcept;

    const std::vector<std::string>& labels() const noexcept;

private:
    T forward(T&& input);

    void parseConfig();
    void parseModel();
    void loadDarknetWeights();
    void loadLabels();

    void addLayer(const YAML::Node& params);

    std::string cfgFile_, modelFile_, weightsFile_, labelsFile_;

    int batch_ = 0, channels_ = 0, height_ = 0, width_ = 0;
    int major_ = 0, minor_ = 0, revision_ = 0;

    layer_vec layers_;
    std::vector<std::string> labels_;
};

template<typename T>
T model_t<T>::forward(T&& input)
{
    auto& in = input;

    for (auto& layer: layers()) {
        layer->forward(in);
        in = layer->output();
    }

    return in;
}

template<typename T>
model_t<T>::model_t(std::string cfgFile) :  cfgFile_(std::move(cfgFile))
{
    parseConfig();
}

template<typename T>
auto model_t<T>::layers() const -> const layer_vec&
{
    return layers_;
}

template<typename T>
int model_t<T>::batch() const
{
    return batch_;
}

template<typename T>
int model_t<T>::channels() const
{
    return channels_;
}

template<typename T>
int model_t<T>::height() const
{
    return height_;
}

template<typename T>
int model_t<T>::width() const
{
    return width_;
}

template<typename T>
int model_t<T>::layerSize() const
{
    return layers_.size();
}

template<typename T>
auto model_t<T>::layerAt(int index) const -> const layer&
{
    PX_CHECK(index < layers_.size(), "Index out of range.");

    return layers_[index];
}

template<typename T>
std::vector<Detection> model_t<T>::predict(const std::string& imageFile, float threshold)
{
    auto image = imread(imageFile.c_str());
    auto sized = imletterbox(image, width(), height());
    auto input = imarray(sized);

    std::cout << "Running network..." << std::endl;

    Timer timer;

    forward(std::move(input));

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

template<typename T>
std::string model_t<T>::asJson(std::vector<Detection>&& detects) const noexcept
{
// FIXME: don't do this here ... breaks #include <nlohmann/json.hpp>

//    using json = nlohmann::json;
//
//    auto j = json::object();
//
//    j["type"] = "FeatureCollection";
//
//    auto features = json::array();
//
//    for (const auto& det: detects) {
//        auto max = det.max();
//        if (max == 0) {
//            continue;   // suppressed
//        }
//
//        auto index = det.maxClass();
//
//        auto feature = json::object();
//        auto geometry = json::object();
//        auto props = json::object();
//        auto coords = json::array();
//
//        feature["type"] = "Feature";
//        geometry["type"] = "Polygon";
//
//        const auto& b = det.box();
//
//        auto left = b.x;
//        auto top = -b.y;
//        auto right = b.x + b.width;
//        auto bottom = -(b.y + b.height);
//
//        coords.emplace_back(json::array({ left, top }));
//        coords.emplace_back(json::array({ right, top }));
//        coords.emplace_back(json::array({ right, bottom }));
//        coords.emplace_back(json::array({ left, bottom }));
//        coords.emplace_back(json::array({ left, top }));
//
//        geometry["coordinates"] = json::array({ coords });
//
//        props["top_cat"] = labels_[index];
//        props["top_score"] = max;
//
//        feature["geometry"] = geometry;
//        feature["properties"] = props;
//
//        features.emplace_back(std::move(feature));
//    }
//
//    j["features"] = features;
//
//    return j.dump(2);

    return "";
}

template<typename T>
const std::vector<std::string>& model_t<T>::labels() const noexcept
{
    return labels_;
}

template<typename T>
void model_t<T>::parseConfig()
{
    auto cfgDoc = YAML::LoadFile(cfgFile_);

    PX_CHECK(cfgDoc.IsMap(), "Document not a map.");
    PX_CHECK(cfgDoc["configuration"], "Document has no configuration.");

    const auto config = cfgDoc["configuration"];
    PX_CHECK(config.IsMap(), "Configuration is not a map.");

    auto cfgPath = boost::filesystem::path(cfgFile_);

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

template<typename T>
void model_t<T>::parseModel()
{
    auto modelDoc = YAML::LoadFile(modelFile_);

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

        addLayer(params);
    }
}

template<typename T>
void model_t<T>::loadDarknetWeights()
{
    std::ifstream ifs(weightsFile_, std::ios::in | std::ios::binary);
    PX_CHECK(ifs.good(), "Could not open file \"%s\".", weightsFile_.c_str());

    ifs.seekg(0, std::ifstream::end);
    auto length = ifs.tellg();
    ifs.seekg(0, std::ifstream::beg);

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

    ifs.close();

//    PX_CHECK(pos == length, "Did not fully read weights file; read %ld bytes, expected to read %ld bytes.",
//             pos, length);

}

template<typename T>
void model_t<T>::loadLabels()
{
    std::ifstream ifs(labelsFile_, std::ios::in | std::ios::binary);
    PX_CHECK(ifs.good(), "Could not open file \"%s\".", labelsFile_.c_str());

    labels_.clear();

    for (std::string label; std::getline(ifs, label);) {
        labels_.push_back(label);
    }
}

template<typename T>
void model_t<T>::addLayer(const YAML::Node& params)
{
    const auto layer = layer_t<T>::create(*this, params);
    layers_.emplace_back(std::move(layer));
}

}   // px

#endif // PIXIENN_MODEL_T_H
