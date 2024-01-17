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

#include "BurnInLRPolicy.h"
#include "ColorMaps.h"
#include "ConstantLRPolicy.h"
#include "CosineLRPolicy.h"
#include "Detection.h"
#include "DeviceTraits.h"
#include "Error.h"
#include "FileUtil.h"
#include "Image.h"
#include "ImageAugmenter.h"
#include "InvLRPolicy.h"
#include "Layer.h"
#include "PxTensor.h"
#include "SigmoidLRPolicy.h"
#include "SmoothSteppedLRPolicy.h"
#include "SteppedLRPolicy.h"
#include "Timer.h"
#include "TrainBatch.h"

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
    virtual void train() = 0;

    virtual void overlay(const std::string& imageFile, const Detections& detects) const = 0;
    virtual std::string asJson(const Detections& detects) const noexcept = 0;

private:
    static BaseModel::Ptr createModel(const std::string& cfgFile, var_map options, bool useGpu);
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
    Model(std::string cfgFile, var_map options = {});
    Model(YAML::Node config, var_map options = {});

    Model(const Model& rhs) = delete;
    Model(Model&& rhs) noexcept = delete;

    Model& operator=(const Model& rhs) = delete;
    Model& operator=(Model&& rhs) = delete;

    void parseModel(const YAML::Node& modelDoc);

    Detections predict(const std::string& imageFile) override;
    void train() override;

    void overlay(const std::string& imageFile, const Detections& detects) const override;
    std::string asJson(const Detections& detects) const noexcept override;

    void forward(const V& input);
    void backward(const V& input);

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
    float learningRate() const;
    float momentum() const noexcept;
    float decay() const noexcept;

    int batch() const noexcept;
    int channels() const noexcept;
    int height() const noexcept;
    int width() const noexcept;

    V* delta() const noexcept;
    float cost() const noexcept;

    int layerSize() const noexcept;
    const LayerPtr& layerAt(int index) const;

    const TrainBatch& trainingBatch() const noexcept;

    bool gradRescaling() const noexcept;
    float gradThreshold() const noexcept;

    bool gradClipping() const noexcept;
    float gradClipValue() const noexcept;

private:
    enum class Category
    {
        TRAIN = 0,
        VAL = 1
    };

    void forward(const ImageVec& image);
    void update();
    void saveWeights(bool final = false);
    void updateLR();

    Detections detections(const cv::Size& imageSize) const;

    void loadModel();
    void loadLabels();
    void parseConfig();
    void parseModel();
    void parsePolicy(const Node& model);
    void parseTrainConfig();
    void loadTrainImages();
    void loadValImages();
    void loadWeights();
    void setup();
    float trainBatch();
    float trainOnce(const V& input);

    using ImageLabels = std::pair<PxCpuVector, GroundTruthVec>;
    ImageLabels loadImgLabels(Category category, const std::string& imagePath, bool augment);
    TrainBatch loadBatch(Category category, int size, bool augment);
    TrainBatch loadBatch(Category category, bool augment);
    GroundTruthVec groundTruth(Category category, const std::string& imagePath);
    int currentBatch() const noexcept;
    std::string weightsFileName(bool final) const;
    void validate();
    LRPolicy* currentPolicy() const noexcept;
    bool isBurningIn() const noexcept;
    void viewImageGT(const std::string& imgPath, const GroundTruthVec& gt, bool augment) const;

    bool training_ = false;
    std::string cfgFile_;
    YAML::Node config_;

    var_map options_;

    LayerVec layers_;
    V* delta_ = nullptr;

    LRPolicy::Ptr policy_;
    LRPolicy::Ptr burnInPolicy_;
    ImageAugmenter::Ptr augmenter_;
    TrainBatch trainBatch_;

    std::size_t burnInBatches_ = 0;

    bool gradRescale_ = false;
    float gradThreshold_ = 0.0f;

    bool gradClip_ = false;
    float gradClipValue_ = 0.0f;

    std::string labelsFile_;
    std::string modelFile_;
    std::string trainImagePath_;
    std::string valImagePath_;
    std::string trainLabelPath_;
    std::string valLabelPath_;
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
    std::vector<std::string> trainImages_;
    std::vector<std::string> valImages_;
};

template<Device D>
Model<D>::Model(YAML::Node config, BaseModel::var_map options)
        : config_(std::move(config)), options_(std::move(options))
{
    loadModel();
}

template<Device D>
void Model<D>::loadModel()
{
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
float Model<D>::learningRate() const
{
    return currentPolicy()->LR();
}

template<Device D>
void Model<D>::train()
{
    std::printf("\nTraining model...\n");

    parseTrainConfig();
    loadTrainImages();
    loadValImages();

    auto avgLoss = std::numeric_limits<float>::lowest();

    Timer timer;
    std::printf("LR: %f%s, Momentum: %f, Decay: %f\n", learningRate(), isBurningIn() ? " (burn-in)" : "", momentum_,
                decay_);

    const auto windowSize = 10;
    const float alpha = 2.0 / (windowSize + 1);

    while (currentBatch() < maxBatches_) {
        Timer batchTimer;
        auto loss = trainBatch();
        avgLoss = avgLoss < 0 ? loss : (avgLoss * (1 - alpha) + loss * alpha);

        auto epoch = seen_ / trainImages_.size();

        if (seen_ % 10 == 0) {
            printf("Epoch: %zu, Seen: %zu, Loss: %f, Avg. Loss: %f, LR: %.12f%s, %s, %zu images\n",
                   epoch, seen_, loss, avgLoss, learningRate(),
                   isBurningIn() ? " (burn-in)" : "",
                   batchTimer.str().c_str(), seen_ * batch_);
        }

        if (seen_ % 1000 == 0 || (seen_ < 1000 && seen_ % 100 == 0)) {
            saveWeights();
        }

        if (seen_ % 1000 == 0) {
            validate();
        }
    }

    saveWeights(true);

    std::printf("trained in %s.\n", timer.str().c_str());
}

template<Device D>
float Model<D>::trainBatch()
{
    trainBatch_ = loadBatch(Category::TRAIN, augmenter_ != nullptr);

    auto error = trainOnce(trainBatch_.imageData());

    return error;
}

template<Device D>
float Model<D>::trainOnce(const V& input)
{
    forward(input);
    backward(input);

    auto error = cost();

    if ((++seen_ / batch_) % subdivs_ == 0) {
        update();
    }

    return error;
}

template<Device D>
void Model<D>::parseTrainConfig()
{
    const auto training = config_["training"];
    PX_CHECK(training.IsMap(), "training is not a map.");

    auto cfgPath = path(cfgFile_);

    auto trainImages = training["train-images"].as<std::string>();
    trainImagePath_ = canonical(trainImages, cfgPath.parent_path()).string();

    auto valImages = training["val-images"].as<std::string>();
    valImagePath_ = canonical(valImages, cfgPath.parent_path()).string();

    auto trainLabels = training["train-labels"].as<std::string>();
    trainLabelPath_ = canonical(trainLabels, cfgPath.parent_path()).string();

    auto valLabels = training["val-labels"].as<std::string>();
    valLabelPath_ = canonical(valLabels, cfgPath.parent_path()).string();
}

template<Device D>
void Model<D>::loadTrainImages()
{
    std::ifstream ifs(trainImagePath_, std::ios::in | std::ios::binary);
    PX_CHECK(ifs.good(), "Could not open file \"%s\".", trainImagePath_.c_str());

    trainImages_.clear();

    for (std::string label; std::getline(ifs, label);) {
        trainImages_.push_back(label);
    }
}

template<Device D>
void Model<D>::loadValImages()
{
    std::ifstream ifs(valImagePath_, std::ios::in | std::ios::binary);
    PX_CHECK(ifs.good(), "Could not open file \"%s\".", valImagePath_.c_str());

    valImages_.clear();

    for (std::string label; std::getline(ifs, label);) {
        valImages_.push_back(label);
    }
}

template<Device D>
int Model<D>::width() const noexcept
{
    return width_;
}

template<Device D>
int Model<D>::height() const noexcept
{
    return height_;
}

template<Device D>
int Model<D>::channels() const noexcept
{
    return channels_;
}

template<Device D>
int Model<D>::batch() const noexcept
{
    return batch_;
}

template<Device D>
const Model<D>::LayerPtr& Model<D>::layerAt(int index) const
{
    PX_CHECK(index < layers_.size(), "Index out of range.");

    return layers_[index];
}

template<Device D>
int Model<D>::layerSize() const noexcept
{
    return layers_.size();
}

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

    if (options_.count("confidence") != 0) {
        threshold_ = option<float>("confidence");
    }

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
    loadModel();
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

    parseModel(modelDoc);
}

template<Device D>
void Model<D>::parseModel(const Node& modelDoc)
{
    PX_CHECK(modelDoc.IsMap(), "Model document not a map.");
    PX_CHECK(modelDoc["model"], "Model document has no model.");

    const auto model = modelDoc["model"];
    PX_CHECK(model.IsMap(), "Model is not a map.");

    if (training_) {
        maxBatches_ = model["max_batches"].as<int>(0);
        PX_CHECK(maxBatches_ > 0, "Model has no max_batches.");

        parsePolicy(model);
        auto augmentNode = model["augmentation"];
        if (augmentNode && augmentNode.IsMap()) {
            auto augment = augmentNode["enabled"].as<bool>(false);
            auto jitter = augmentNode["jitter"].as<float>(0.2f);
            auto hue = augmentNode["hue"].as<float>(0.0f);
            auto saturation = augmentNode["saturation"].as<float>(1.0f);
            auto exposure = augmentNode["exposure"].as<float>(1.0f);

            auto flip = augmentNode["flip"].as<bool>(false);
            if (augment) {
                augmenter_ = std::make_unique<ImageAugmenter>(jitter, hue, saturation, exposure, flip);
            }
        }

        auto gr = model["gradient_rescale"];
        if (gr && gr.IsMap()) {
            gradRescale_ = gr["enabled"].as<bool>(false);
            gradThreshold_ = gr["threshold"].as<float>(0.0f);
        }

        auto gc = model["gradient_clipping"];
        if (gc && gc.IsMap()) {
            gradClip_ = gc["enabled"].as<bool>(false);
            gradClipValue_ = gc["value"].as<float>(1.0f);
        }
    }

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
void Model<D>::parsePolicy(const Node& model)
{
    auto lrNode = model["learning_rate"];
    if (lrNode) {
        PX_CHECK(lrNode.IsMap(), "learning_rate must be a map.");
        auto learningRate = lrNode["initial_learning_rate"].as<float>(0.001f);

        burnInBatches_ = lrNode["burn_in_batches"].as<int>(0);
        if (burnInBatches_ > 0) {
            auto burnInPower = lrNode["burn_in_power"].as<float>(1.0f);
            burnInPolicy_ = std::make_unique<BurnInLRPolicy>(learningRate, burnInBatches_, burnInPower);
        }

        auto sPolicy = lrNode["policy"].as<std::string>("constant");

        if (sPolicy == "constant") {
            policy_ = std::make_unique<ConstantLRPolicy>(learningRate);
        } else if (sPolicy == "cosine_annealing") {
            auto cosineNode = lrNode["cosine_annealing"];
            auto minLR = cosineNode["min_learning_rate"].as<float>(0.0f);
            auto batchesPerCycle = cosineNode["batches_per_cycle"].as<int>(1000);

            policy_ = std::make_unique<CosineAnnealingLRPolicy>(learningRate, minLR, batchesPerCycle);
        } else if (sPolicy == "sigmoid") {
            auto sigmoidNode = lrNode["sigmoid"];
            auto targetLR = sigmoidNode["target_learning_rate"].as<float>(0.001f);
            auto factor = sigmoidNode["factor"].as<float>(12.0f);

            policy_ = std::make_unique<SigmoidLRPolicy>(learningRate, targetLR, factor, maxBatches_);

        } else if (sPolicy == "smooth_stepped") {
            auto smoothNode = lrNode["smooth_stepped"];
            auto steps = smoothNode["steps"];
            PX_CHECK(steps.IsSequence(), "steps must be a sequence of integers.");
            auto vSteps = steps.as<std::vector<int>>();

            auto targets = smoothNode["targets"];
            PX_CHECK(targets.IsSequence(), "targets must be a sequence of floating point numbers.");
            auto vTargets = targets.as<std::vector<float>>();

            policy_ = std::make_unique<SmoothSteppedLRPolicy>(learningRate, vSteps, vTargets);
        } else if (sPolicy == "stepped") {
            auto steppedNode = lrNode["stepped"];
            auto steps = steppedNode["steps"];
            PX_CHECK(steps.IsSequence(), "steps must be a sequence of integers.");
            auto vSteps = steps.as<std::vector<int>>();

            auto scales = steppedNode["scales"];
            PX_CHECK(scales.IsSequence(), "scales must be a sequence of floating point numbers.");
            auto vScales = scales.as<std::vector<float>>();

            policy_ = std::make_unique<SteppedLRPolicy>(learningRate, vSteps, vScales);
        } else if (sPolicy == "inverse") {
            auto invNode = lrNode["inverse"];
            auto gamma = invNode["gamma"].as<float>(0.9f);
            auto power = invNode["power"].as<float>(1.0f);

            policy_ = std::make_unique<InvLRPolicy>(learningRate, gamma, power);
        } else {
            PX_ERROR_THROW("Unknown policy \"%s\".", sPolicy.c_str());
        }
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
    updateLR();

    for (auto& layer: layers_) {
        layer->update();
    }
}

template<Device D>
void Model<D>::updateLR()
{
    auto batchNum = currentBatch();
    auto* policy = currentPolicy();

    policy->update(batchNum);
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

template<Device D>
Model<D>::ImageLabels Model<D>::loadImgLabels(Model::Category category, const std::string& imagePath, bool augment)
{
    auto gts = groundTruth(category, imagePath);

    if (hasOption("view-image")) {
        viewImageGT(imagePath, gts, augment);
    }

    if (augment && augmenter_) {
        auto orig = imreadNormalize(imagePath.c_str());

        auto augmented = augmenter_->augment(orig, { width(), height() }, gts);
        augmenter_->distort(augmented.first);

        auto vector = imvector(augmented.first);

        return { vector, augmented.second };
    } else {
        auto vec = imreadVector(imagePath.c_str(), width(), height());

        GroundTruthVec newGts;

        for (const auto& gt: gts) {
            GroundTruth newGt(gt);
            newGt.box.x() = (gt.box.x() * vec.ax) + vec.dx;
            newGt.box.y() = (gt.box.y() * vec.ay) + vec.dy;
            newGt.box.w() *= vec.ax;
            newGt.box.h() *= vec.ay;

            newGts.emplace_back(std::move(newGt));
        }

        return { vec.data, newGts };
    }
}

template<Device D>
float Model<D>::momentum() const noexcept
{
    return momentum_;
}

template<Device D>
float Model<D>::decay() const noexcept
{
    return decay_;
}

template<Device D>
void Model<D>::saveWeights(bool final)
{
    if (!boost::filesystem::exists(backupDir_)) {
        boost::filesystem::create_directory(backupDir_);
    }

    auto fileName = weightsFileName(final);

    std::ofstream ofs(fileName, std::ios::out | std::ios::trunc | std::ios::binary);
    PX_CHECK(ofs.good(), "Could not open file \"%s\". %s", fileName.c_str(), std::strerror(errno));

    ofs.write((char*) &major_, sizeof(int));
    ofs.write((char*) &minor_, sizeof(int));
    ofs.write((char*) &revision_, sizeof(int));
    ofs.write((char*) &seen_, sizeof(int));

    for (const auto& layer: layers()) {
        layer->saveWeights(ofs);
    }

    ofs.close();
}

template<Device D>
TrainBatch Model<D>::loadBatch(Category type, bool augment)
{
    return loadBatch(type, batch_, augment);
}

template<Device D>
TrainBatch Model<D>::loadBatch(Model::Category category, int size, bool augment)
{
    TrainBatch batch(size, channels_, height_, width_);

    const auto& images = category == Category::TRAIN ?
                         trainImages_ : valImages_;

    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, images.size() - 1);

    auto n = std::min<std::size_t>(size, images.size());

    for (auto i = 0; i < n; ++i) {
        auto j = distribution(generator);
        const auto& imagePath = images[j];
        auto imgLabels = loadImgLabels(category, imagePath, augment);

        batch.setImageData(i, imgLabels.first);  // the image data must be copied
        batch.setGroundTruth(i, std::move(imgLabels.second));
    }

    return batch;
}

template<Device D>
GroundTruthVec Model<D>::groundTruth(Category category, const std::string& imagePath)
{
    auto basePath = baseName(imagePath);

    const auto& path = category == Category::TRAIN ? trainLabelPath_ : valLabelPath_;

    boost::filesystem::path gtFile(path);
    gtFile /= basePath + ".txt";
    gtFile = canonical(gtFile);

    std::ifstream ifs(gtFile);
    PX_CHECK(ifs.good(), "Could not open file \"%s\".", gtFile.c_str());

    GroundTruthVec vector;

    auto nclasses = classes();
    std::size_t id;
    float x, y, w, h;

    while (ifs >> id >> x >> y >> w >> h) {
        GroundTruth gt;
        gt.classId = id;

        gt.box.x() = constrain(0.0f, 1.0f, x);
        gt.box.y() = constrain(0.0f, 1.0f, y);
        gt.box.w() = constrain(0.0f, 1.0f, w);
        gt.box.h() = constrain(0.0f, 1.0f, h);

        vector.emplace_back(std::move(gt));
    }

    return vector;
}

template<Device D>
int Model<D>::currentBatch() const noexcept
{
    if (batch_ == 0 || subdivs_ == 0) {
        return 0;
    }

    auto batchNum = seen_ / (batch_ * subdivs_);

    return batchNum;
}

template<Device D>
std::string Model<D>::weightsFileName(bool final) const
{
    if (final) {
        return weightsFile_;
    }

    auto base = baseName(weightsFile_);

    auto fileName = (boost::format("%s_%u.weights") % base % seen_).str();
    boost::filesystem::path path(backupDir_);
    path /= fileName;
    fileName = path.string();

    return fileName;
}

template<Device D>
void Model<D>::validate()
{
    std::cout << "Pausing training to validate..." << std::flush;

    auto batch = loadBatch(Category::VAL, 100, false);
/*
    validator_.validate(std::move(batch));

    std::printf("\n%zu: mAP: %f, Avg. Recall: %f, micro-Avg. F1: %f\n",
                seen_, validator_.mAP(), validator_.avgRecall(), validator_.microAvgF1());
*/

    std::cout << "Resuming training..." << std::endl << std::flush;
}

template<Device D>
LRPolicy* Model<D>::currentPolicy() const noexcept
{
    auto batchNum = currentBatch();

    LRPolicy* policy;
    if (batchNum < burnInBatches_) {
        policy = burnInPolicy_.get();
    } else {
        policy = policy_.get();
    }

    return policy;
}

template<Device D>
bool Model<D>::isBurningIn() const noexcept
{
    auto batchNum = currentBatch();

    return batchNum < burnInBatches_;
}

template<Device D>
void Model<D>::viewImageGT(const std::string& imagePath, const GroundTruthVec& gt, bool augment) const
{
    ColorMaps colors("plasma");

    cv::Mat image;

    if (augment && augmenter_) {
        auto orig = imread(imagePath.c_str());
        auto augmented = augmenter_->augment(orig, { width(), height() }, gt);

        image = augmented.first;
        augmenter_->distort(image);

        cv::cvtColor(image, image, cv::COLOR_BGR2BGRA);

        for (const auto& g: augmented.second) {
            auto index = g.classId;
            const auto& label = labels_[index];

            auto bgColor = colors.color(index);
            auto textColor = imtextcolor(bgColor);

            auto lb = lightBox(g.box, { width(), height() });

            imrect(image, lb, bgColor, 2);
            imtabbedText(image, label.c_str(), lb.tl(), textColor, bgColor, 2);
        }
    } else {
        auto mat = imread(imagePath.c_str(), width(), height());
        image = mat.image;

        cv::cvtColor(image, image, cv::COLOR_BGR2BGRA);

        for (const auto& g: gt) {
            auto index = g.classId;
            const auto& label = labels_[index];

            auto bgColor = colors.color(index);
            auto textColor = imtextcolor(bgColor);

            auto x = (g.box.x() * mat.ax) + mat.dx;
            auto y = (g.box.y() * mat.ay) + mat.dy;
            auto w = g.box.w() * mat.ax;
            auto h = g.box.h() * mat.ay;

            auto lb = lightBox({ x, y, w, h }, { width(), height() });

            imrect(image, lb, bgColor, 2);
            imtabbedText(image, label.c_str(), lb.tl(), textColor, bgColor, 2);
        }
    }

    cv::imshow("image", image);
    cv::waitKey();
}

template<Device D>
const TrainBatch& Model<D>::trainingBatch() const noexcept
{
    return trainBatch_;
}

template<Device D>
float Model<D>::gradClipValue() const noexcept
{
    return gradClipValue_;
}

template<Device D>
bool Model<D>::gradClipping() const noexcept
{
    return gradClip_;
}

template<Device D>
float Model<D>::gradThreshold() const noexcept
{
    return gradThreshold_;
}

template<Device D>
bool Model<D>::gradRescaling() const noexcept
{
    return gradRescale_;
}

///////////////////////////////////////////////////////////////////////////////

using CpuModel = Model<>; // Model<Device::CPU
using CudaModel = Model<Device::CUDA>;

}   // px

#ifdef USE_CUDA

#include "cuda/Model.h"

#endif
