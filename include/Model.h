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

#ifndef PIXIENN_MODEL_H
#define PIXIENN_MODEL_H

#include <boost/program_options.hpp>

#include "Cublas.h"
#include "Cudnn.h"
#include "Detection.h"
#include "Layer.h"

namespace px {

class Model
{
public:
    Model(std::string cfgFile);
    Model(const Model& rhs) = delete;
    Model(Model&& rhs) = default;

    Model& operator=(const Model& rhs) = delete;
    Model& operator=(Model&& rhs) = default;

    using LayerVec = std::vector<Layer::Ptr>;
    const LayerVec& layers() const;

    int batch() const;
    int channels() const;
    int height() const;
    int width() const;

    const int layerSize() const;
    const Layer::Ptr& layerAt(int index) const;

    std::vector<Detection> predict(const std::string& imageFile, float threshold,
                                   const boost::program_options::variables_map& options);
    std::string asJson(std::vector<Detection>&& detects) const noexcept;

    const std::vector<std::string>& labels() const noexcept;

#ifdef USE_CUDA
    const CublasContext& cublasContext() const noexcept;
    const CudnnContext& cudnnContext() const noexcept;
#endif

private:
    xt::xarray<float> forward(const xt::xarray<float>& input);

#ifdef USE_CUDA
    PxDevVector<float> forwardGpu(const PxDevVector<float>& input);
#endif

    void parseConfig();
    void parseModel();
    void loadDarknetWeights();
    void loadLabels();

    std::string cfgFile_, modelFile_, weightsFile_, labelsFile_;

    int batch_ = 0, channels_ = 0, height_ = 0, width_ = 0;
    int major_ = 0, minor_ = 0, revision_ = 0;

    LayerVec layers_;
    std::vector<std::string> labels_;

#ifdef USE_CUDA
    CublasContext cublasCtxt_;
    CudnnContext cudnnCtxt_;
#endif
};

}   // px

#endif // PIXIENN_MODEL_H
