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

#include "event.pb.h"
#include "summary.pb.h"

#include "BatchLoader.h"
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
#include "MiniBatch.h"
#include "PxTensor.h"
#include "RandomLRPolicy.h"
#include "RecordWriter.h"
#include "SigmoidLRPolicy.h"
#include "SmoothCyclicDecayLRPolicy.h"
#include "SmoothSteppedLRPolicy.h"
#include "SteppedLRPolicy.h"
#include "Timer.h"

using namespace YAML;
using namespace boost::filesystem;
using namespace tensorflow;

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

enum class Mode
{
    INFERRING, VALIDATING, TRAINING
};

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
    Detections detections() const;
    Detections detections(const cv::Size& imageSize) const;

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
    bool validating() const noexcept;

    void setMode(Mode mode) noexcept;
    void setThreshold(float threshold) noexcept;
    int classes() const noexcept;
    float learningRate() const;
    float momentum() const noexcept;
    float decay() const noexcept;

    int batch() const noexcept;
    int channels() const noexcept;
    int height() const noexcept;
    int width() const noexcept;

    float cost() const noexcept;

    int layerSize() const noexcept;
    const LayerPtr& layerAt(int index) const;

    const MiniBatch& trainingBatch() const noexcept;

    bool gradRescaling() const noexcept;
    float gradThreshold() const noexcept;

    bool gradClipping() const noexcept;
    float gradClipValue() const noexcept;

    std::size_t seen() const noexcept;
    RecordWriter& recordWriter() const;

    bool adamEnabled() const noexcept;
    float adamBeta1() const noexcept;
    float adamBeta2() const noexcept;
    float adamEpsilon() const noexcept;

    void setLabels(const std::vector<std::string>& labels);
    const std::vector<std::string>& labels() const noexcept;

    void setTrainBatch(MiniBatch batch) noexcept;
private:
    void forward(const ImageVec& image);
    void update();
    void saveWeights(bool final = false);
    void saveWeights(const std::string& fileName);

    void updateLR();

    void loadModel();
    void loadLabels();
    void parseConfig();
    void parseModel();
    void parsePolicy(const Node& model);
    void parseTrainConfig();
    void loadWeights();
    void setup();
    float trainBatch();
    float trainOnce(const V& input);

    using ImageLabels = std::pair<PxCpuVector, GroundTruthVec>;
    std::string weightsFileName(bool final) const;
    std::string weightsLatestFileName() const;

    void validate();
    LRPolicy* currentPolicy() const noexcept;
    bool isBurningIn() const noexcept;
    void writeMetrics();
    void writeAvgLoss();
    void writeLR();
    void writemAP();
    void writeAvgRecall();
    void writeMicroAvgF1();
    void writeAvgValLoss();
    void writeAccuracy();

    Mode mode_ = Mode::INFERRING;

    std::string cfgFile_;
    YAML::Node config_;

    var_map options_;

    LayerVec layers_;

    LRPolicy::Ptr policy_;
    LRPolicy::Ptr burnInPolicy_;
    ImageAugmenter::Ptr augmenter_;
    BatchLoader::Ptr trainLoader_;
    BatchLoader::Ptr valLoader_;

    MiniBatch trainBatch_;
    RecordWriter::Ptr writer_;

    float avgLoss_ = 0.0f;

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
    std::string eventFile_;

    int maxBatches_ = 0;

    int batch_ = 0;
    int channels_ = 0;
    float decay_ = 0.0f;
    int height_ = 0;
    float momentum_ = 0.0f;
    int subdivs_ = 0;
    int timeSteps_ = 0;
    int width_ = 0;
    bool valEnabled_ = false;
    int valInterval_ = 0;
    int valBatches_ = 0;
    float valThresh_ = 0.0f;

    bool adamEnabled_ = false;
    float adamBeta1_ = 0.0f;
    float adamBeta2_ = 0.0f;
    float adamEpsilon_ = 0.0f;

    bool esEnabled_ = false;    // Early stopping
    float esThreshold_ = 0.0f;
    int esPatience_ = 0;

    int saveWeightsInterval_ = 0;
    int writeMetricsInterval_ = 0;

    // network version
    int major_ = 0;
    int minor_ = 1;
    int revision_ = 0;

    size_t seen_ = 0;
    float threshold_ = 0.0f;    // Threshold for confidence
    float mAP_ = 0.0f;          // Mean Average Precision
    float avgRecall_ = 0.0f;    // Average Recall
    float microAvgF1_ = 0.0f;   // Micro Average F1
    float avgValLoss_ = 0.0f;   // Average Validation Loss
    float valAccuracy_ = 0.0f;  // Validation Accuracy
    float cost_ = 0.0f;         // Network cost

    std::vector<std::string> labels_;
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

    auto viewImage = hasOption("view-image");

    trainLoader_ = std::make_unique<BatchLoader>(trainImagePath_, trainLabelPath_, batch_, channels_, height_, width_,
                                                 labels_, augmenter_, viewImage);

    if (valEnabled_) {
        valLoader_ = std::make_unique<BatchLoader>(valImagePath_, valLabelPath_, batch_, channels_, height_, width_,
                                                   labels_, augmenter_);
    }

    avgLoss_ = std::numeric_limits<float>::lowest();
    auto bestValLoss = std::numeric_limits<float>::max();
    auto valsWithoutImprovement = 0;
    constexpr auto windowSize = 10;
    constexpr auto alpha = 2.0f / (windowSize + 1);

    Timer timer;
    std::printf("LR: %f%s, Momentum: %f, Decay: %f\n", learningRate(), isBurningIn() ? " (burn-in)" : "", momentum_,
                decay_);

    while (seen_ < maxBatches_) {
        Timer batchTimer;
        auto loss = trainBatch();

        if (std::isinf(loss) || std::isnan(loss)) {
            loss = std::numeric_limits<float>::max();
        }

        avgLoss_ = avgLoss_ < 0 ? loss : (avgLoss_ * (1 - alpha) + loss * alpha);
        if (std::isinf(avgLoss_) || std::isnan(avgLoss_)) {
            avgLoss_ = loss;
        }

        auto imagesSeen = seen_ * batch_;
        auto epoch = imagesSeen / trainLoader_->size();

        std::printf("Epoch: %zu, Seen: %zu, Loss: %f, Avg. Loss: %f, LR: %.12f%s, %s, %zu images\n",
                    epoch, seen_, loss, avgLoss_, learningRate(),
                    isBurningIn() ? " (burn-in)" : "",
                    batchTimer.str().c_str(), imagesSeen);

        if (valEnabled_ && valInterval_ && seen_ % valInterval_ == 0) {
            validate();

            if (esEnabled_) {   // check for early stopping
                if (avgValLoss_ < bestValLoss - esThreshold_) {
                    bestValLoss = avgValLoss_;
                    valsWithoutImprovement = 0;
                } else {
                    valsWithoutImprovement++;
                }

                if (valsWithoutImprovement >= esPatience_) {
                    std::printf("Early stopping due to lack of improvement in validation loss.\n");
                    break;
                }
            }
        }

        if (saveWeightsInterval_ && seen_ % saveWeightsInterval_ == 0) {
            saveWeights();
        }

        if (writeMetricsInterval_ && seen_ % writeMetricsInterval_ == 0) {
            writeMetrics();
        }
    }

    saveWeights(true);

    std::printf("trained in %s.\n", timer.str().c_str());
}

template<Device D>
void Model<D>::writeMetrics()
{
    writeAvgLoss();
    writeLR();
    writemAP();
    writeAvgRecall();
    writeMicroAvgF1();
    writeAvgValLoss();
    writeAccuracy();
}

template<Device D>
void Model<D>::writeAvgLoss()
{
    Event event;
    event.set_wall_time(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    event.set_step(seen_);

    auto* summary = event.mutable_summary();
    auto* value = summary->add_value();
    value->set_tag("avg-loss");
    value->set_simple_value(avgLoss_);

    writer_->write(event);
}

template<Device D>
void Model<D>::writeLR()
{
    Event event;
    event.set_wall_time(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    event.set_step(seen_);

    auto* summary = event.mutable_summary();
    auto* value = summary->add_value();
    value->set_tag("learning-rate");
    value->set_simple_value(learningRate());

    writer_->write(event);
}

template<Device D>
void Model<D>::writemAP()
{
    Event event;
    event.set_wall_time(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    event.set_step(seen_);

    auto* summary = event.mutable_summary();
    auto* value = summary->add_value();
    value->set_tag("mAP");
    value->set_simple_value(mAP_);

    writer_->write(event);
}

template<Device D>
void Model<D>::writeAvgRecall()
{
    Event event;
    event.set_wall_time(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    event.set_step(seen_);

    auto* summary = event.mutable_summary();
    auto* value = summary->add_value();
    value->set_tag("avg-recall");
    value->set_simple_value(avgRecall_);

    writer_->write(event);
}

template<Device D>
void Model<D>::writeMicroAvgF1()
{
    Event event;
    event.set_wall_time(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    event.set_step(seen_);

    auto* summary = event.mutable_summary();
    auto* value = summary->add_value();
    value->set_tag("micro-avg-f1");
    value->set_simple_value(microAvgF1_);

    writer_->write(event);
}

template<Device D>
void Model<D>::writeAvgValLoss()
{
    Event event;
    event.set_wall_time(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    event.set_step(seen_);

    auto* summary = event.mutable_summary();
    auto* value = summary->add_value();
    value->set_tag("avg-val-loss");
    value->set_simple_value(avgValLoss_);

    writer_->write(event);
}

template<Device D>
void Model<D>::writeAccuracy()
{
    Event event;
    event.set_wall_time(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    event.set_step(seen_);

    auto* summary = event.mutable_summary();
    auto* value = summary->add_value();
    value->set_tag("accuracy");
    value->set_simple_value(valAccuracy_);

    writer_->write(event);
}

template<Device D>
float Model<D>::trainBatch()
{
    trainBatch_ = trainLoader_->next();

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
    mode_ = hasOption("train") ? Mode::TRAINING : Mode::INFERRING;

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
    auto image = imreadVector(imageFile.c_str(), width_, height_, channels_);

    std::printf("\nRunning model...");

    Timer timer;

    forward(image);

    auto detects = detections(image.originalSize);

    std::printf("predicted in %s.\n", timer.str().c_str());

    return detects;
}

template<Device D>
Detections Model<D>::detections() const
{
    Detections detections;

    for (auto& layer: layers()) {
        auto* detector = dynamic_cast<Detector*>(layer.get());
        if (detector) {
            detector->addDetects(detections, threshold_);
        }
    }

    return detections;
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
    auto img = imread(imageFile.c_str(), channels_);
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

        props["batch_id"] = detect.batchId();
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
    auto clearWeights = hasOption("clear-weights");
    auto latestWeightsFile = weightsLatestFileName();

    if (training() && clearWeights) {
        boost::filesystem::remove(weightsFile_);
    }

    std::ifstream ifs(weightsFile_, std::ios::in | std::ios::binary);
    if (inferring() && ifs.fail()) { // it is not an error for training weights to not exist.
        PX_ERROR_THROW("Could not open file \"%s\".", weightsFile_.c_str());
    }

    if (training() && ifs.fail() && !clearWeights) { // if not found, let's try to load the latest weights
        std::printf("\nweights not found, trying latest weights \"%s\"...", latestWeightsFile.c_str());
        ifs.open(latestWeightsFile, std::ios::in | std::ios::binary);
        if (ifs.is_open()) {
            std::printf("found.\n");
        } else {
            std::printf("not found.\n");
        }
    }

    if (ifs.is_open()) {
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

#ifdef USE_CUDA

#include "cuda/Layer.h"

#endif  // USE_CUDA

#include "LayerFactory.h"
#include "Validator.h"

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

    if (training() || validating()) {
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

        auto adamNode = model["adam"];
        if (adamNode && adamNode.IsMap()) {
            adamEnabled_ = adamNode["enabled"].as<bool>(false);
            adamBeta1_ = adamNode["beta1"].as<float>(0.9f);
            adamBeta2_ = adamNode["beta2"].as<float>(0.999f);
            adamEpsilon_ = adamNode["epsilon"].as<float>(1e-8f);
        }

        auto esNode = model["early_stopping"];
        if (esNode && esNode.IsMap()) {
            esEnabled_ = esNode["enabled"].as<bool>(true);
            if (esEnabled_) {
                esPatience_ = esNode["patience"].as<int>(10);
                esThreshold_ = esNode["threshold"].as<float>(0.0001f);
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

        decay_ = model["decay"].as<float>(0.0001f);
        momentum_ = model["momentum"].as<float>(0.9f);

        eventFile_ = model["event_file"].as<std::string>("events.out.tfevents");
        writer_ = RecordWriter::create(eventFile_, true);

        auto val = model["validation"];
        if (val && val.IsMap()) {
            valEnabled_ = val["enabled"].as<bool>(true);
            valInterval_ = val["interval"].as<int>(1000);
            valBatches_ = val["batches"].as<int>(100);
            valThresh_ = val["threshold"].as<float>(0.2f);
        }

        saveWeightsInterval_ = model["save_weights_interval"].as<int>(1000);
        writeMetricsInterval_ = model["write_metrics_interval"].as<int>(1000);
    }

    batch_ = training() || validating() ? model["batch"].as<int>() : 1;
    channels_ = model["channels"].as<int>();
    height_ = model["height"].as<int>();
    subdivs_ = model["subdivisions"].as<int>(1);
    timeSteps_ = model["time_steps"].as<int>(1);
    width_ = model["width"].as<int>();

    batch_ /= subdivs_;
    batch_ *= timeSteps_;

    auto inputs = batch_ * height_ * width_ * channels_;

    const auto layers = model["layers"];
    if (!layers) {
        return;
    }

    PX_CHECK(layers.IsSequence(), "Model layers must be a sequence.");

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

        auto burnInNode = lrNode["burn_in"];
        if (burnInNode && burnInNode.IsMap()) {
            auto burnInBatches = burnInNode["batches"].as<int>(0);
            if (burnInBatches > 0) {
                auto burnInPower = burnInNode["power"].as<float>(4.0f);
                burnInPolicy_ = std::make_unique<BurnInLRPolicy>(learningRate, burnInBatches, burnInPower);
            }
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
        } else if (sPolicy == "random") {
            auto randomNode = lrNode["random"];
            auto minLR = randomNode["min_learning_rate"].as<float>(0.0f);
            auto updateInterval = randomNode["update_interval"].as<int>(1000);

            policy_ = std::make_unique<RandomLRPolicy>(learningRate, minLR, updateInterval);
        } else if (sPolicy == "smooth_cyclic_decay") {
            auto node = lrNode["smooth_cyclic_decay"];
            auto gamma = node["gamma"].as<float>(0.01f);
            auto peakHeight = node["peak_height"].as<float>(0.1f);
            auto peakWidth = node["peak_width"].as<int>();
            auto peakInterval = node["peak_interval"].as<int>();

            policy_ = std::make_unique<SmoothCyclicDecayLRPolicy>(learningRate, gamma, peakHeight, peakWidth,
                                                                  peakInterval);

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
        if (!inferring() && layer->hasCost()) {
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
    V* grad = nullptr;

    for (int i = layers_.size() - 1; i >= 0; --i) {
        const auto& layer = layers_[i];

        if (i == 0) {
            in = &input;
            grad = nullptr;
        } else {
            auto& prev = layers_[i - 1];
            grad = &prev->delta();
            in = &prev->output();
        }

        layer->backward(*in, grad);
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
    auto* policy = currentPolicy();

    policy->update(seen_);
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
    return mode_ == Mode::INFERRING;
}

template<Device D>
bool Model<D>::training() const noexcept
{
    return mode_ == Mode::TRAINING;
}

template<Device D>
bool Model<D>::validating() const noexcept
{
    return mode_ == Mode::VALIDATING;
}

template<Device D>
void Model<D>::setMode(Mode mode) noexcept
{
    mode_ = mode;
}

template<Device D>
void Model<D>::setThreshold(float threshold) noexcept
{
    threshold_ = threshold;

}

template<Device D>
int Model<D>::classes() const noexcept
{
    return labels_.size();
}

template<Device D>
float Model<D>::cost() const noexcept
{
    return cost_;
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
    saveWeights(fileName);

    fileName = weightsLatestFileName();
    saveWeights(fileName);
}

template<Device D>
void Model<D>::saveWeights(const std::string& fileName)
{
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
std::string Model<D>::weightsLatestFileName() const
{
    auto base = baseName(weightsFile_);

    auto fileName = (boost::format("%s_latest.weights") % base).str();
    boost::filesystem::path path(backupDir_);
    path /= fileName;
    fileName = path.string();

    return fileName;
}

template<Device D>
void Model<D>::validate()
{
    std::cout << "Pausing training to validate..." << std::flush;

    Validator <D> validator(valThresh_, classes());

    for (auto i = 0; i < valBatches_; ++i) {
        auto batch = valLoader_->next();
        validator.validate(*this, std::move(batch));
    }

    mAP_ = validator.mAP();
    avgRecall_ = validator.avgRecall();
    microAvgF1_ = validator.microAvgF1();
    avgValLoss_ = validator.avgLoss();
    valAccuracy_ = validator.accuracy();

    std::printf("\nEpoch: %zu, mAP: %f, Avg. Recall: %f, Micro-Avg. F1: %f, Avg. Val. Loss: %f, Accuracy: %f\n",
                seen_ / valLoader_->size(), mAP_, avgRecall_, microAvgF1_, avgValLoss_, valAccuracy_);

    std::cout << "Resuming training..." << std::endl << std::flush;
}

template<Device D>
LRPolicy* Model<D>::currentPolicy() const noexcept
{
    LRPolicy* policy;
    if (isBurningIn()) {
        policy = burnInPolicy_.get();
    } else {
        policy = policy_.get();
    }

    return policy;
}

template<Device D>
bool Model<D>::isBurningIn() const noexcept
{
    return seen_ < burnInBatches_;
}

template<Device D>
const MiniBatch& Model<D>::trainingBatch() const noexcept
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

template<Device D>
std::size_t Model<D>::seen() const noexcept
{
    return seen_;
}

template<Device D>
RecordWriter& Model<D>::recordWriter() const
{
    PX_CHECK(writer_, "No record writer.");

    return *writer_;
}

template<Device D>
float Model<D>::adamEpsilon() const noexcept
{
    return adamEpsilon_;
}

template<Device D>
float Model<D>::adamBeta2() const noexcept
{
    return adamBeta2_;
}

template<Device D>
float Model<D>::adamBeta1() const noexcept
{
    return adamBeta1_;
}

template<Device D>
bool Model<D>::adamEnabled() const noexcept
{
    return adamEnabled_;
}

template<Device D>
void Model<D>::setLabels(const std::vector<std::string>& labels)
{
    this->labels_ = labels;
}

template<Device D>
const std::vector<std::string>& Model<D>::labels() const noexcept
{
    return labels_;
}

template<Device D>
void Model<D>::setTrainBatch(MiniBatch batch) noexcept
{
    this->trainBatch_ = std::move(batch);
}

///////////////////////////////////////////////////////////////////////////////

using CpuModel = Model<>; // Model<Device::CPU
using CudaModel = Model<Device::CUDA>;

}   // px

#ifdef USE_CUDA

#include "cuda/Model.h"

#endif
