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

#ifdef USE_CUDA

#include "Cublas.h"
#include "Cudnn.h"

#endif  // USE_CUDA

namespace px {

class Model
{
public:
    explicit Model(std::string cfgFile, const boost::program_options::variables_map& options = {});
    Model(const Model& rhs) = delete;
    Model(Model&& rhs) noexcept = delete;

    Model& operator=(const Model& rhs) = delete;
    Model& operator=(Model&& rhs) = delete;

    using LayerVec = std::vector<Layer::Ptr>;
    [[nodiscard]] const LayerVec& layers() const;

    [[nodiscard]] int batch() const;
    [[nodiscard]] int channels() const;
    [[nodiscard]] int height() const;
    [[nodiscard]] int width() const;

    int layerSize() const;
    [[nodiscard]] const Layer::Ptr& layerAt(int index) const;

    std::vector<Detection> predict(const std::string& imageFile);
    void overlay(const std::string& imageFile, const Detections& detects) const;
    std::string asJson(const Detections& detects) const noexcept;

    [[nodiscard]] const std::vector<std::string>& labels() const noexcept;

#ifdef USE_CUDA
    [[nodiscard]] const CublasContext& cublasContext() const noexcept;
    [[nodiscard]] const CudnnContext& cudnnContext() const noexcept;
    [[nodiscard]] bool useGpu() const noexcept;
#endif

private:
    void forward(const PxCpuVector& input) const;
#ifdef USE_CUDA
    void forwardGpu(const PxCpuVector& input) const;
    void setupGpu();
#endif

    void parseConfig();
    void parseModel();
    void loadDarknetWeights();
    void loadLabels();

    std::string cfgFile_, modelFile_, weightsFile_, labelsFile_;
    int batch_ = 0, channels_ = 0, height_ = 0, width_ = 0;
    int major_ = 0, minor_ = 0, revision_ = 0;
    float threshold_;

    LayerVec layers_;
    std::vector<std::string> labels_;

#ifdef USE_CUDA
    CublasContext::Ptr cublasCtxt_;
    CudnnContext::Ptr cudnnCtxt_;
    bool gpu_;
#endif
};

}   // px

#endif // PIXIENN_MODEL_H
