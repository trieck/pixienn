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

#ifndef PIXIENN_MODEL_H
#define PIXIENN_MODEL_H

#include <boost/program_options.hpp>
#include <boost/program_options/variables_map.hpp>

#include "Detection.h"
#include "Layer.h"
#include "LRPolicy.h"
#include "TrainBatch.h"
#include "Validator.h"

#ifdef USE_CUDA

#include "Cublas.h"
#include "Cudnn.h"

#endif  // USE_CUDA

namespace px {

class Model
{
private:
    using var_map = boost::program_options::variables_map;

public:
    explicit Model(std::string cfgFile, var_map options = {});
    Model(const Model& rhs) = delete;
    Model(Model&& rhs) noexcept = delete;

    Model& operator=(const Model& rhs) = delete;
    Model& operator=(Model&& rhs) = delete;

    using LayerVec = std::vector<Layer::Ptr>;
    const LayerVec& layers() const;

    int batch() const noexcept;
    int channels() const noexcept;
    int height() const noexcept;
    int width() const noexcept;
    int subdivs() const noexcept;
    int timeSteps() const noexcept;
    int layerSize() const;
    const Layer::Ptr& layerAt(int index) const;

    void train();

    /**
     * @brief Perform object detection on an input image.
     *
     * This method takes the path to an image file and performs object detection using the trained
     * neural network model. It returns a vector of Detection objects, each representing a detected
     * object in the image along with its associated information, such as the bounding box coordinates
     * and confidence score.
     *
     * @param imageFile The path to the input image file.
     * @return A vector of Detection objects representing the detected objects.
     *
     * @note The returned vector may be empty if no objects are detected in the image.
     * @warning This method assumes that the neural network model has been trained and loaded
     *          successfully prior to calling predict.
     */
    std::vector<Detection> predict(const std::string& imageFile);

    /**
     * @brief Overlay detections on an input image.
     *
     * This method takes the path to an image file and a set of detections. It overlays bounding boxes
     * and additional information for each detected object on the input image. The resulting image
     * with overlays is not modified; instead, the overlay may be saved to a new file or displayed
     * depending on the implementation.
     *
     * @param imageFile The path to the input image file.
     * @param detects   A set of detections containing information about detected objects.
     *
     * @note This method assumes that the input image exists and that the detections are valid.
     * @warning The overlay result may be saved to a new file or displayed, depending on the
     *          implementation. Ensure appropriate handling of the overlay result.
     */
    void overlay(const std::string& imageFile, const Detections& detects) const;

    void forward(const PxCpuVector& input);
    void backward(const PxCpuVector& input);

    std::vector<Detection> detections(const cv::Size& imageSize) const;
    std::vector<Detection> detections() const;

    std::string asJson(const Detections& detects) const noexcept;
    const std::vector<std::string>& labels() const noexcept;

    bool hasOption(const std::string& option) const;
    bool training() const;
    bool inferring() const;
    float cost() const noexcept;
    float learningRate() const;
    float momentum() const noexcept;
    float decay() const noexcept;

    template<typename T>
    T option(const std::string& name) const;
    PxCpuVector* delta() noexcept;

    uint32_t classes() const noexcept;
    const TrainBatch& trainingBatch() const noexcept;

    void setTraining(bool training) noexcept;
    void setThreshold(float threshold) noexcept;
    size_t seen() const noexcept;

    bool gradRescaling() const noexcept;
    float gradThreshold() const noexcept;

#ifdef USE_CUDA
    const CublasContext& cublasContext() const noexcept;
    const CudnnContext& cudnnContext() const noexcept;
    bool useGpu() const noexcept;
#endif
private:
    enum class Category
    {
        TRAIN = 0,
        VAL = 1
    };

    float trainBatch();
    float trainOnce(const PxCpuVector& input);

    void saveWeights(bool final = false);
    void update();
    void updateLR();

#ifdef USE_CUDA
    void forwardGpu(const PxCpuVector& input) const;
    void setupGpu();
#endif
    void parseConfig();
    void parseTrainConfig();
    void parseModel();
    void parsePolicy(const YAML::Node& model);
    void loadWeights();
    void loadLabels();
    void loadTrainImages();
    void loadValImages();

    using ImageLabels = std::pair<PxCpuVector, GroundTruthVec>;
    ImageLabels loadImgLabels(Category category, const std::string& imagePath, bool augment);
    TrainBatch loadBatch(Category category, int size, bool augment);
    TrainBatch loadBatch(Category category, bool augment);
    GroundTruthVec groundTruth(Category category, const std::string& imagePath);
    int currentBatch() const noexcept;
    std::string weightsFileName(bool final) const;
    void validate();

    // file paths
    std::string cfgFile_;
    std::string modelFile_;
    std::string weightsFile_;
    std::string labelsFile_;
    std::string trainImagePath_;
    std::string valImagePath_;
    std::string trainLabelPath_;
    std::string valLabelPath_;
    std::string backupDir_;

    // network dimensions
    int batch_ = 0;
    int channels_ = 0;
    int height_ = 0;
    int width_ = 0;

    // network version
    int major_ = 0;
    int minor_ = 1;
    int revision_ = 0;

    // training parameters
    bool training_ = false;               // Flag indicating whether the model is in training mode
    TrainBatch trainBatch_;               // Instance of TrainBatch class for managing training batches
    PxCpuVector* delta_ = nullptr;        // Pointer to a PxCpuVector for storing delta values (nullptr by default)
    LRPolicy::Ptr policy_;                // Pointer to an LRPolicy object for managing learning rate policies
    bool augment_ = false;                // Flag indicating whether data augmentation is enabled

    // optimization parameters
    int maxBatches_ = 0;                   // Maximum number of batches for training
    int subdivs_ = 0;                      // Number of subdivisions for training batches
    int timeSteps_ = 0;                    // Number of time steps for training
    size_t seen_ = 0;                      // Total number of batches seen during training

    // hyperparameters for data augmentation
    float threshold_ = 0.0f;               // Threshold for data augmentation
    float momentum_ = 0.0f;                // Momentum for optimization
    float decay_ = 0.0f;                   // Decay rate for optimization
    float jitter_ = 0.0f;                  // Jitter for data augmentation
    float saturation_ = 0.0f;              // Saturation for data augmentation
    float exposure_ = 0.0f;                // Exposure for data augmentation
    float hue_ = 0.0f;                     // Hue for data augmentation
    float cost_ = 0.0f;                    // Cost associated with the training process

    // gradient rescaling parameters
    bool gradRescaling_ = false;           // Flag indicating whether gradient rescaling is enabled
    float gradThreshold_ = 0.0f;           // Threshold for gradient rescaling

    // configuration
    YAML::Node config_;

    // network layers
    LayerVec layers_;

    // labels and training data
    std::vector<std::string> labels_;
    std::vector<std::string> trainImages_;
    std::vector<std::string> valImages_;

    // program options
    var_map options_;

    Validator validator_;

#ifdef USE_CUDA
    CublasContext::Ptr cublasCtxt_;
    CudnnContext::Ptr cudnnCtxt_;
    bool gpu_;
#endif
};

template<typename T>
T Model::option(const std::string& name) const
{
    return options_[name].as<T>();
}

}   // px

#endif // PIXIENN_MODEL_H