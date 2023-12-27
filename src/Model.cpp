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

#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <utility>
#include <opencv2/imgproc/types_c.h>

#include "ColorMaps.h"
#include "ConstantLRPolicy.h"
#include "Error.h"
#include "FileUtil.h"
#include "Image.h"
#include "ImageAugmenter.h"
#include "InvLRPolicy.h"
#include "Layer.h"
#include "Model.h"
#include "SmoothSteppedLRPolicy.h"
#include "SteppedLRPolicy.h"
#include "Timer.h"
#include "Utility.h"

using namespace YAML;
using namespace boost::filesystem;
using json = nlohmann::json;

namespace po = boost::program_options;

namespace px {

Model::Model(std::string cfgFile, var_map options)
        : options_(std::move(options)), cfgFile_(std::move(cfgFile)), validator_(*this)
{
    training_ = hasOption("train");

#ifdef USE_CUDA
    gpu_ = !hasOption("no-gpu");
    if (gpu_) {
        setupGpu();
    }
#endif
    if (inferring()) {
        threshold_ = options_["confidence"].as<float>();
    }

    parseConfig();
}

void Model::parseConfig()
{
    struct timespec ts;
    ts.tv_sec = 2000 / 1000;
    ts.tv_nsec = 2000 * 1000000;
    nanosleep(&ts, NULL);

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
    } else {    // weights for training, if they exist
        weightsFile_ = option<std::string>("weights-file");
    }

    loadWeights();
}

void Model::parseModel()
{
    auto modelDoc = LoadFile(modelFile_);

    PX_CHECK(modelDoc.IsMap(), "Model document not a map.");
    PX_CHECK(modelDoc["model"], "Document has no model.");

    const auto model = modelDoc["model"];
    PX_CHECK(model.IsMap(), "Model is not a map.");

    parsePolicy(model);

    maxBatches_ = model["max_batches"].as<int>(0);

    batch_ = model["batch"].as<int>();
    channels_ = model["channels"].as<int>();
    height_ = model["height"].as<int>();
    width_ = model["width"].as<int>();
    subdivs_ = model["subdivisions"].as<int>(1);
    timeSteps_ = model["time_steps"].as<int>(1);
    momentum_ = model["momentum"].as<float>(0.9f);
    decay_ = model["decay"].as<float>(0.0001f);
    augment_ = model["augment"].as<bool>(true);
    jitter_ = model["jitter"].as<float>(0.2f);
    saturation_ = model["saturation"].as<float>(1.0f);
    exposure_ = model["exposure"].as<float>(1.0f);
    hue_ = model["hue"].as<float>(0.0f);

    auto grNode = model["gradient_rescale"];
    if (grNode && grNode.IsMap()) {
        gradRescaling_ = grNode["enabled"].as<bool>(false);
        gradThreshold_ = grNode["threshold"].as<float>(0.0f);
    }

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

        const auto layer = Layer::create(*this, params);

        channels = layer->outChannels();
        height = layer->outHeight();
        width = layer->outWidth();
        inputs = layer->outputs();

        layer->print(std::cout);

        layers_.emplace_back(layer);
    }
}

const Model::LayerVec& Model::layers() const
{
    return layers_;
}

int Model::batch() const noexcept
{
    return batch_;
}

int Model::currentBatch() const noexcept
{
    if (batch_ == 0 || subdivs_ == 0) {
        return 0;
    }

    auto batchNum = seen_ / (batch_ * subdivs_);

    return batchNum;
}


int Model::channels() const noexcept
{
    return channels_;
}

int Model::height() const noexcept
{
    return height_;
}

int Model::width() const noexcept
{
    return width_;
}

void Model::forward(const PxCpuVector& input)
{
    auto sum = 0.0f;
    auto count = 0;

    const auto* in = &input;

    for (auto& layer: layers()) {
        layer->forward(*in);
        if (layer->hasCost()) {
            sum += layer->cost();
            ++count;
        }

        in = &layer->output();
    }

    cost_ = count ? sum / count : 0.0f;
}

void Model::backward(const PxCpuVector& input)
{
    const PxCpuVector* in;

    for (int i = layers_.size() - 1; i >= 0; --i) {
        auto& layer = layers_[i];

        if (i == 0) {
            in = &input;
        } else {
            auto& prev = layers_[i - 1];
            delta_ = prev->delta();
            in = &prev->output();
        }
        layer->backward(*in);
    }
}

#ifdef USE_CUDA
void Model::forwardGpu(const PxCpuVector& input) const
{
    PxCudaVector vinput(input.begin().base(), input.end().base());

    const auto* in = &vinput;

    for (auto& layer: layers()) {
        layer->forwardGpu(*in);
        in = &layer->outputGpu();
    }
}
#endif // USE_CUDA

void Model::loadWeights()
{
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

std::vector<Detection> Model::predict(const std::string& imageFile)
{
    auto image = imreadVector(imageFile.c_str(), width(), height());

    std::cout << std::endl << "Running model...";

    Timer timer;

#ifdef USE_CUDA
    if (useGpu()) {
        forwardGpu(image.data);
    } else {
        forward(image.data);
    }
#else
    forward(image.data);
#endif  // USE_CUDA

    auto detects = detections(image.originalSize);

    std::printf("predicted in %s.\n", timer.str().c_str());

    return detects;
}

std::vector<Detection> Model::detections(const cv::Size& imageSize) const
{
    std::vector<Detection> detections;

    for (auto& layer: layers()) {
        auto* detector = dynamic_cast<Detector*>(layer.get());
        if (detector) {
#ifdef USE_CUDA
            if (useGpu()) {
                detector->addDetectsGpu(detections, imageSize.width, imageSize.height, threshold_);
            } else {
                detector->addDetects(detections, imageSize.width, imageSize.height, threshold_);
            }
#else
            detector->addDetects(detections, imageSize.width, imageSize.height, threshold_);
#endif // USE_CUDA
        }
    }

    return detections;
}

std::vector<Detection> Model::detections() const
{
    std::vector<Detection> detections;

    for (auto& layer: layers()) {
        auto* detector = dynamic_cast<Detector*>(layer.get());
        if (detector) {
            detector->addDetects(detections, threshold_);
        }
    }

    return detections;
}

void Model::train()
{
    std::printf("\nTraining model...\n");

    parseTrainConfig();
    loadTrainImages();
    loadValImages();

    auto avgLoss = -std::numeric_limits<float>::max();

    Timer timer;
    std::printf("LR: %.8f, Momentum: %.8f, Decay: %.8f\n", learningRate(), momentum_, decay_);

    while (currentBatch() < maxBatches_) {
        Timer batchTimer;
        auto loss = trainBatch();
        avgLoss = avgLoss < 0 ? loss : (avgLoss * .9f + loss * .1f);

        auto epoch = seen_ / trainImages_.size();

        if (seen_ % 10 == 0) {
            printf("Epoch: %zu, Batches seen: %zu, Loss: %.2f, Avg. Loss: %.2f, LR: %.8f, %s, %zu images\n",
                   epoch, seen_, loss, avgLoss, learningRate(), batchTimer.str().c_str(), seen_ * batch_);
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

float Model::trainBatch()
{
    trainBatch_ = loadBatch(Category::TRAIN, augment_);

    auto error = trainOnce(trainBatch_.imageData());

    return error;
}

float Model::trainOnce(const PxCpuVector& input)
{
    forward(input);
    backward(input);

    auto error = cost();

    if ((++seen_ / batch_) % subdivs_ == 0) {
        update();
    }

    return error;
}

void Model::saveWeights(bool final)
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

void Model::update()
{
    updateLR();

    for (auto& layer: layers()) {
        layer->update();
    }
}

void Model::updateLR()
{
    policy_->update(seen_);
}

void Model::parseTrainConfig()
{
    PX_CHECK(config_["training"], "Configuration has no training section.");

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

TrainBatch Model::loadBatch(Category type, bool augment)
{
    return loadBatch(type, batch_, augment);
}

TrainBatch Model::loadBatch(Category category, int size, bool augment)
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

auto Model::loadImgLabels(Category category, const std::string& imagePath, bool augment) -> ImageLabels
{
    auto gts = groundTruth(category, imagePath);

    if (!augment) {
        auto image = imreadVector(imagePath.c_str(), width(), height());
        return { image.data, gts };
    }

    auto orig = imreadNormalize(imagePath.c_str());

    ImageAugmenter augmenter(jitter_, hue_, saturation_, exposure_);
    auto augmented = augmenter.augment(orig, { width(), height() }, gts);

    auto vector = imvector(augmented.first);

    return { vector, augmented.second };
}

GroundTruthVec Model::groundTruth(Category category, const std::string& imagePath)
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

        auto left = x - w / 2;
        auto right = x + w / 2;
        auto top = y - h / 2;
        auto bottom = y + h / 2;

        id = std::min<std::size_t>(id, nclasses - 1);
        left = std::max(0.0f, std::min(left, 1.0f));
        right = std::max(0.0f, std::min(right, 1.0f));
        top = std::max(0.0f, std::min(top, 1.0f));
        bottom = std::max(0.0f, std::min(bottom, 1.0f));

        gt.box.x = (left + right) / 2;
        gt.box.y = (top + bottom) / 2;
        gt.box.width = right - left;
        gt.box.height = bottom - top;

        vector.emplace_back(std::move(gt));
    }

    return vector;
}

void Model::loadTrainImages()
{
    std::ifstream ifs(trainImagePath_, std::ios::in | std::ios::binary);
    PX_CHECK(ifs.good(), "Could not open file \"%s\".", trainImagePath_.c_str());

    trainImages_.clear();

    for (std::string label; std::getline(ifs, label);) {
        trainImages_.push_back(label);
    }
}

void Model::loadValImages()
{
    std::ifstream ifs(valImagePath_, std::ios::in | std::ios::binary);
    PX_CHECK(ifs.good(), "Could not open file \"%s\".", valImagePath_.c_str());

    valImages_.clear();

    for (std::string label; std::getline(ifs, label);) {
        valImages_.push_back(label);
    }
}

int Model::layerSize() const
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

void Model::overlay(const std::string& imageFile, const Detections& detects) const
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

std::string Model::asJson(const Detections& detects) const noexcept
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

bool Model::hasOption(const std::string& option) const
{
    if (options_.count(option) == 0) {
        return false;
    }

    return options_[option].as<bool>();
}

bool Model::training() const
{
    return training_;
}

bool Model::inferring() const
{
    return !training();
}

int Model::subdivs() const noexcept
{
    return subdivs_;
}

int Model::timeSteps() const noexcept
{
    return timeSteps_;
}

PxCpuVector* Model::delta() noexcept
{
    return delta_;
}

float Model::cost() const noexcept
{
    return cost_;
}

uint32_t Model::classes() const noexcept
{
    return labels_.size();
}

const TrainBatch& Model::trainingBatch() const noexcept
{
    return trainBatch_;
}

float Model::learningRate() const
{
    return policy_->LR();
}

float Model::momentum() const noexcept
{
    return momentum_;
}

float Model::decay() const noexcept
{
    return decay_;
}

void Model::parsePolicy(const Node& model)
{
    auto lrNode = model["learning_rate"];
    if (lrNode) {
        PX_CHECK(lrNode.IsMap(), "learning_rate must be a map.");

        auto learningRate = lrNode["initial_learning_rate"].as<float>(0.001f);
        auto sPolicy = lrNode["policy"].as<std::string>("constant");

        if (sPolicy == "constant") {
            policy_ = std::make_unique<ConstantLRPolicy>(learningRate);
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

std::string Model::weightsFileName(bool final) const
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

void Model::validate()
{
    std::cout << "Pausing training to validate..." << std::flush;

    auto batch = loadBatch(Category::VAL, 100, false);
    validator_.validate(std::move(batch));

    std::printf("\n%zu: mAP: %.4f, Avg. Recall: %.4f, micro-Avg. F1: %.4f\n",
                seen_, validator_.mAP(), validator_.avgRecall(), validator_.microAvgF1());

    std::cout << "Resuming training..." << std::endl << std::flush;
}

void Model::setTraining(bool training) noexcept
{
    training_ = training;
}

void Model::setThreshold(float threshold) noexcept
{
    threshold_ = threshold;
}

size_t Model::seen() const noexcept
{
    return seen_;
}

bool Model::gradRescaling() const noexcept
{
    return gradRescaling_;
}

float Model::gradThreshold() const noexcept
{
    return gradThreshold_;
}

#ifdef USE_CUDA

const CublasContext& Model::cublasContext() const noexcept
{
    return *cublasCtxt_;
}

const CudnnContext& Model::cudnnContext() const noexcept
{
    return *cudnnCtxt_;
}

void Model::setupGpu()
{
    cublasCtxt_ = std::make_unique<CublasContext>();
    cudnnCtxt_ = std::make_unique<CudnnContext>();
}

bool Model::useGpu() const noexcept
{
    return gpu_;
}

#endif

} // px