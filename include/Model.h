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
#include "GroundTruth.h"
#include "Layer.h"

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
    std::vector<Detection> predict(const std::string& imageFile);
    void overlay(const std::string& imageFile, const Detections& detects) const;
    std::string asJson(const Detections& detects) const noexcept;
    const std::vector<std::string>& labels() const noexcept;

    bool hasOption(const std::string& option) const;
    bool training() const;
    bool inferring() const;
    float cost() const noexcept;

    template<typename T>
    T option(const std::string& name) const;
    PxCpuVector::pointer delta() noexcept;
    uint32_t classes() const noexcept;
    const ImageTruths& truth() const noexcept;
    const uint32_t truths() const noexcept;

#ifdef USE_CUDA
    const CublasContext& cublasContext() const noexcept;
    const CudnnContext& cudnnContext() const noexcept;
    bool useGpu() const noexcept;
#endif
private:
    float trainBatch(ImageTruths&& batch);
    void forward(const PxCpuVector& input);
    void backward(const PxCpuVector& input);
#ifdef USE_CUDA
    void forwardGpu(const PxCpuVector& input) const;
    void setupGpu();
#endif
    void parseConfig();
    void parseTrainConfig();
    void parseModel();
    void loadWeights();
    void loadLabels();
    std::uint32_t detectTruths() const;
    void loadTrainImages();
    ImageTruths loadBatch();
    PxCpuVector groundTruth(const std::string& imagePath);

    // file paths
    std::string cfgFile_;
    std::string modelFile_;
    std::string weightsFile_;
    std::string labelsFile_;
    std::string trainImagePath_;
    std::string trainGTPath_;

    // network dimensions
    int batch_ = 0;
    int channels_ = 0;
    int height_ = 0;
    int width_ = 0;

    // network version
    int major_ = 0;
    int minor_ = 0;
    int revision_ = 0;

    // training parameters
    ImageTruths truth_;
    PxCpuVector::pointer delta_ = nullptr;

    std::uint32_t truths_ = 0;
    int maxBoxes_ = 0;
    int subdivs_ = 0;
    int timeSteps_ = 0;
    int outputs_ = 0;
    float threshold_ = 0.0f;
    float learningRate_ = 0.0f;
    float momentum_ = 0.0f;
    float decay_ = 0.0f;
    float jitter_ = 0.0f;
    float angle_ = 0.0f;
    float aspect_ = 0.0f;
    float saturation_ = 0.0f;
    float exposure_ = 0.0f;
    float hue_ = 0.0f;
    float cost_ = 0.0f;

    // configuration
    YAML::Node config_;

    // network layers
    LayerVec layers_;

    // labels and training data
    std::vector<std::string> labels_;
    std::vector<std::string> trainImages_;

    // program options
    var_map options_;

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
