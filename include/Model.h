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

#pragma once

#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <boost/program_options/variables_map.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/types_c.h>
#include <utility>
#include <yaml-cpp/node/node.h>

#include "ColorMaps.h"
#include "Detection.h"
#include "DeviceTraits.h"
#include "Error.h"
#include "Image.h"
#include "Layer.h"
#include "PxTensor.h"
#include "Timer.h"

using namespace YAML;
using namespace boost::filesystem;
using json = nlohmann::json;

namespace px {

///////////////////////////////////////////////////////////////////////////////
class BaseModel
{
public:
    using Ptr = std::unique_ptr<BaseModel>;
    using var_map = boost::program_options::variables_map;

    BaseModel() = default;
    BaseModel(const BaseModel& rhs) = delete;
    BaseModel(BaseModel&& rhs) noexcept = delete;
    virtual ~BaseModel() = default;

    static Ptr create(const std::string& cfgFile, var_map options = {});

    virtual Detections predict(const std::string& imageFile) = 0;
    virtual void overlay(const std::string& imageFile, const Detections& detects) const = 0;
    virtual std::string asJson(const Detections& detects) const noexcept = 0;
};

template<Device D>
class DeviceExtras
{
};

#ifdef USE_CUDA

template<>
class DeviceExtras<Device::CUDA>
{
public:
    const CublasContext& cublasContext() const noexcept;
    const CudnnContext& cudnnContext() const noexcept;

protected:
    std::unique_ptr<CublasContext> cublasCtxt_;
    std::unique_ptr<CudnnContext> cudnnCtxt_;
};

inline const CublasContext& DeviceExtras<Device::CUDA>::cublasContext() const noexcept
{
    return *cublasCtxt_;
}

inline const CudnnContext& DeviceExtras<Device::CUDA>::cudnnContext() const noexcept
{
    return *cudnnCtxt_;
}

#endif  // USE_CUDA

///////////////////////////////////////////////////////////////////////////////
template<Device D = Device::CPU>
class Model : public BaseModel, public DeviceExtras<D>
{
public:
    using V = typename DeviceTraits<D>::VectorType;
    using LayerPtr = std::shared_ptr<Layer<D>>;
    using LayerVec = std::vector<LayerPtr>;

    Model(var_map options = {});
    explicit Model(std::string cfgFile, var_map options = {});
    Model(const Model& rhs) = delete;
    Model(Model&& rhs) noexcept = delete;

    Model& operator=(const Model& rhs) = delete;
    Model& operator=(Model&& rhs) = delete;

    Detections predict(const std::string& imageFile) override;
    void overlay(const std::string& imageFile, const Detections& detects) const override;
    std::string asJson(const Detections& detects) const noexcept override;

    void forward(const V& input);
    void backward(const V& input);
    void update();

    template<typename T>
    T option(const std::string& name) const;

    bool hasOption(const std::string& option) const;

    template<typename T>
    void addLayer(YAML::Node layerDef = {});

    void addLayer(LayerPtr layer);

    const LayerVec& layers() const;

    bool inferring() const noexcept;
    bool training() const noexcept;
    int classes() const noexcept;

    V* delta() const noexcept;
    float cost() const noexcept;

private:
    void forward(const ImageVec& image);
    Detections detections(const cv::Size& imageSize) const;

    void loadLabels();
    void parseConfig();
    void parseModel();
    void loadWeights();
    void setup();

    bool training_ = false;
    std::string cfgFile_;
    YAML::Node config_;

    var_map options_;

    LayerVec layers_;
    V* delta_ = nullptr;

    std::string labelsFile_;
    std::string modelFile_;
    std::string backupDir_;
    std::string weightsFile_;

    int maxBatches_ = 0;

    int batch_ = 0;
    int channels_ = 0;
    float decay_ = 0.0f;
    int height_ = 0;
    float momentum_ = 0.0f;
    int subdivs_ = 0;
    int timeSteps_ = 0;
    int width_ = 0;

    // network version
    int major_ = 0;
    int minor_ = 1;
    int revision_ = 0;

    size_t seen_ = 0;

    float threshold_ = 0.0f;    // Threshold for confidence
    float cost_ = 0.0f;

    std::vector<std::string> labels_;
};

template<Device D>
void Model<D>::addLayer(Model::LayerPtr layer)
{
    layers_.emplace_back(std::move(layer));
}

template<Device D>
template<typename T>
void Model<D>::addLayer(YAML::Node layerDef)
{
    layers_.emplace_back(std::make_shared<T>(*this, layerDef));
}

template<Device D>
Model<D>::Model(var_map options) : options_(std::move(options))
{
    training_ = hasOption("train");
    threshold_ = training_ ? 0.0f : option<float>("confidence");

    setup();
}

template<Device D>
void Model<D>::setup()
{
}

template<Device D>
Model<D>::Model(std::string cfgFile, var_map options) : Model<D>(options)
{
    cfgFile_ = std::move(cfgFile);
    parseConfig();
}

template<Device D>
Detections Model<D>::predict(const std::string& imageFile)
{
    auto image = imreadVector(imageFile.c_str(), width_, height_);

    std::printf("\nRunning model...");

    Timer timer;

    forward(image);

    auto detects = detections(image.originalSize);

    std::printf("predicted in %s.\n", timer.str().c_str());

    return detects;
}

template<Device D>
Detections Model<D>::detections(const cv::Size& imageSize) const
{
    Detections detections;

    for (auto& layer: layers()) {
        auto* detector = dynamic_cast<Detector*>(layer.get());
        if (detector) {
            detector->addDetects(detections, imageSize.width, imageSize.height, threshold_);
        }
    }

    return detections;
}

template<Device D>
void Model<D>::overlay(const std::string& imageFile, const Detections& detects) const
{
    auto img = imread(imageFile.c_str());
    cv::cvtColor(img, img, cv::COLOR_BGR2BGRA);

    ColorMaps colors(option<std::string>("color-map"));
    auto thickness = std::max(1, option<int>("line-thickness"));

    for (const auto& detect: detects) {
        auto index = detect.classIndex();
        const auto& label = labels_[index];

        auto bgColor = colors.color(index);
        auto textColor = imtextcolor(bgColor);

        const auto& box = detect.box();
        imrect(img, box, bgColor, thickness);

        auto text = boost::format("%1%: %2$.2f%%") % label % (detect.prob() * 100);
        std::cout << text << std::endl;

        if (!hasOption("no-labels")) {
            imtabbedText(img, text.str().c_str(), box.tl(), textColor, bgColor, thickness);
        }
    }

    if (hasOption("tiff32")) {
        imsaveTiff("predictions.tif", img);
    } else {
        imsave("predictions.jpg", img);
    }
}

template<Device D>
std::string Model<D>::asJson(const Detections& detects) const noexcept
{
    auto json = json::object();

    json["type"] = "FeatureCollection";

    auto features = json::array();

    for (const auto& detect: detects) {
        auto index = detect.classIndex();

        auto feature = json::object();
        auto geometry = json::object();
        auto props = json::object();
        auto coords = json::array();

        feature["type"] = "Feature";
        geometry["type"] = "Polygon";

        const auto& b = detect.box();

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

        props["class"] = labels_[index];
        props["confidence"] = detect.prob();

        feature["geometry"] = geometry;
        feature["properties"] = props;

        features.emplace_back(std::move(feature));
    }

    json["features"] = features;

    return json.dump(2);
}

template<Device D>
void Model<D>::forward(const ImageVec& image)
{
    forward(image.data);
}

template<Device D>
void Model<D>::parseConfig()
{
    config_ = LoadFile(cfgFile_);

    PX_CHECK(config_.IsMap(), "Document not a map.");
    PX_CHECK(config_["configuration"], "Document has no configuration.");

    const auto config = config_["configuration"];
    PX_CHECK(config.IsMap(), "Configuration is not a map.");

    auto cfgPath = path(cfgFile_);

    auto labels = config["labels"].as<std::string>();
    labelsFile_ = canonical(labels, cfgPath.parent_path()).string();
    loadLabels();

    auto model = config["model"].as<std::string>();
    modelFile_ = canonical(model, cfgPath.parent_path()).string();
    parseModel();

    backupDir_ = config["backup-dir"].as<std::string>("backup");

    if (inferring()) {
        auto weights = config["weights"].as<std::string>();
        weightsFile_ = canonical(weights, cfgPath.parent_path()).string();
    } else {
        weightsFile_ = option<std::string>("weights-file");
    }

    loadWeights();
}

template<Device D>
void Model<D>::loadLabels()
{
    std::ifstream ifs(labelsFile_, std::ios::in | std::ios::binary);
    PX_CHECK(ifs.good(), "Could not open file \"%s\".", labelsFile_.c_str());

    labels_.clear();

    for (std::string label; std::getline(ifs, label);) {
        labels_.emplace_back(std::move(label));
    }
}

template<Device D>
void Model<D>::loadWeights()
{
    if (training() && hasOption("clear-weights")) {
        boost::filesystem::remove(weightsFile_);
    }

    std::ifstream ifs(weightsFile_, std::ios::in | std::ios::binary);
    if (inferring() && ifs.bad()) { // it is not an error for training weights to not exist.
        PX_ERROR_THROW("Could not open file \"%s\".", weightsFile_.c_str());
    }

    if (ifs.good()) {
        ifs.seekg(0, std::ifstream::end);
        auto length = ifs.tellg();
        ifs.seekg(0, std::ifstream::beg);

        ifs.read((char*) &major_, sizeof(int));
        ifs.read((char*) &minor_, sizeof(int));
        ifs.read((char*) &revision_, sizeof(int));

        seen_ = 0;
        if ((major_ * 10 + minor_) >= 2 && major_ < 1000 && minor_ < 1000) {
            ifs.read((char*) &seen_, sizeof(size_t));
        } else {
            ifs.read((char*) &seen_, sizeof(int));
        }

        std::streamoff pos = ifs.tellg();
        for (const auto& layer: layers()) {
            pos += layer->loadWeights(ifs);
        }

        PX_CHECK(pos == length, "Did not fully read weights file; read %ld bytes, expected to read %ld bytes.",
                 pos, length);

        ifs.close();
    }
}

}   // px

#include "LayerFactory.h"

namespace px {

template<Device D>
void Model<D>::parseModel()
{
    auto modelDoc = LoadFile(modelFile_);

    PX_CHECK(modelDoc.IsMap(), "Model document not a map.");
    PX_CHECK(modelDoc["model"], "Model document has no model.");

    const auto model = modelDoc["model"];
    PX_CHECK(model.IsMap(), "Model is not a map.");

    maxBatches_ = model["max_batches"].as<int>(0);
    // parsePolicy();

    // TODO: augmentation

    batch_ = model["batch"].as<int>();
    channels_ = model["channels"].as<int>();
    decay_ = model["decay"].as<float>(0.0001f);
    height_ = model["height"].as<int>();
    momentum_ = model["momentum"].as<float>(0.9f);
    subdivs_ = model["subdivisions"].as<int>(1);
    timeSteps_ = model["time_steps"].as<int>(1);
    width_ = model["width"].as<int>();

    batch_ /= subdivs_;
    batch_ *= timeSteps_;

    auto inputs = batch_ * height_ * width_ * channels_;

    const auto layers = model["layers"];
    PX_CHECK(layers.IsSequence(), "Model has no layers.");
    PX_CHECK(layers.size() > 0, "Model has no layers.");

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

        auto layer = LayerFactories<D>::create(*this, params);

        channels = layer->outChannels();
        height = layer->outHeight();
        width = layer->outWidth();
        inputs = layer->outputs();

        layer->print(std::cout);

        layers_.emplace_back(std::move(layer));
    }
}

template<Device D>
void Model<D>::forward(const V& input)
{
    auto sum = 0.0f;
    auto count = 0;

    const auto* in = &input;

    for (const auto& layer: layers_) {
        layer->forward(*in);
        if (training_ && layer->hasCost()) {
            sum += layer->cost();
            ++count;
        }

        in = &layer->output();
    }

    cost_ = count ? sum / count : 0.0f;
}

template<Device D>
void Model<D>::backward(const V& input)
{
    const V* in = &input;

    for (int i = layers_.size() - 1; i >= 0; --i) {
        const auto& layer = layers_[i];

        if (i == 0) {
            in = &input;
        } else {
            auto& prev = layers_[i - 1];
            delta_ = &prev->delta();
            in = &prev->output();
        }

        layer->backward(*in);
    }
}

template<Device D>
void Model<D>::update()
{
}

template<Device D>
template<typename T>
T Model<D>::option(const std::string& name) const
{
    return options_[name].as<T>();
}

template<Device D>
bool Model<D>::hasOption(const std::string& option) const
{
    if (options_.count(option) == 0) {
        return false;
    }

    return options_[option].as<bool>();
}

template<Device D>
const Model<D>::LayerVec& Model<D>::layers() const
{
    return layers_;
}

template<Device D>
bool Model<D>::inferring() const noexcept
{
    return !training_;
}

template<Device D>
bool Model<D>::training() const noexcept
{
    return training_;
}

template<Device D>
int Model<D>::classes() const noexcept
{
    return labels_.size();
}

template<Device D>
auto Model<D>::delta() const noexcept -> V*
{
    return delta_;
}

template<Device D>
float Model<D>::cost() const noexcept
{
    return cost_;
}

using CpuModel = Model<>; // Model<Device::CPU
using CudaModel = Model<Device::CUDA>;

}   // px

#ifdef USE_CUDA

#include "ModelCuda.h"

#endif
