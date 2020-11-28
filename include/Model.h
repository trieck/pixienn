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

#include "Detection.h"
#include "Layer.h"

namespace px {

class Model
{
public:
    Model(const std::string& cfgFile);
    Model(const Model& rhs) = default;
    Model(Model&& rhs) = default;

    Model& operator=(const Model& rhs) = default;
    Model& operator=(Model&& rhs) = default;

    using LayerVec = std::vector<Layer::Ptr>;
    const LayerVec& layers() const;

    const int batch() const;
    const int channels() const;
    const int height() const;
    const int width() const;

    const int layerSize() const;
    const Layer::Ptr& layerAt(int index) const;

    std::vector<Detection> predict(const std::string& imageFile, float threshold);
    std::string asJson(std::vector<Detection>&& detects) const noexcept;

    const std::vector<std::string>& labels() const noexcept;

private:
    xt::xarray<float> forward(xt::xarray<float>&& input);

    void parseConfig();
    void parseModel();
    void loadDarknetWeights();
    void loadLabels();

    std::string cfgFile_, modelFile_, weightsFile_, labelsFile_;

    int batch_ = 0, channels_ = 0, height_ = 0, width_ = 0;
    int major_ = 0, minor_ = 0, revision_ = 0;

    LayerVec layers_;
    std::vector<std::string> labels_;
};

}   // px

#endif // PIXIENN_MODEL_H
