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

#ifndef PIXIENN_YOLOLAYER_H
#define PIXIENN_YOLOLAYER_H

#include "Activation.h"
#include "Detection.h"
#include "Layer.h"

namespace px {

class YoloLayer : public Layer, public Detector
{
protected:
    YoloLayer(Model& model, const YAML::Node& layerDef);

public:
    ~YoloLayer() override = default;

    std::ostream& print(std::ostream& os) override;
    void forward(const PxCpuVector& input) override;
    void backward(const PxCpuVector& input) override;

    void addDetects(Detections& detections, int width, int height, float threshold) override;
    void addDetects(Detections& detections, float threshold) override;

#ifdef USE_CUDA
    void forwardGpu(const PxCudaVector& input) override;
    void addDetectsGpu(Detections& detections, int width, int height, float threshold) override;
#endif
private:
    friend LayerFactories;

    void setup() override;
    void addDetects(Detections& detections, int width, int height, float threshold, const float* predictions) const;
    void addDetects(Detections& detections, float threshold, const float* predictions) const;

    int entryIndex(int batch, int location, int entry) const noexcept;
    cv::Rect yoloBox(const float* p, int mask, int index, int col, int row, int w, int h) const;

    Activation<Logistic<Activations::Type>> logistic_;

    int total_ = 0;
    std::vector<int> mask_, anchors_;
};

} // px

#endif // PIXIENN_YOLOLAYER_H
