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

    inline bool hasCost() const noexcept override
    {
        return true;
    }

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
    cv::Rect2f yoloBox(const float* p, int mask, int index, int i, int j) const;
    cv::Rect scaledYoloBox(const float* p, int mask, int index, int i, int j, int w, int h) const;
    void resetStats();
    void processRegion(int b, int i, int j);
    void deltaYoloClass(int index, int classId);
    float deltaYoloBox(const GroundTruth& truth, int mask, int index, int i, int j);
    void processObjects(int b);
    int maskIndex(int n);

    Activation<Logistic<Activations::Type>> logistic_;
    PxCpuTensor<1> biases_, biasUpdates_;

    int num_ = 0;
    std::vector<int> mask_, anchors_;
    float ignoreThresh_ = 0.0f;
    float truthThresh_ = 0.0f;

    float avgIoU = 0.0f;
    float recall_ = 0.0f;
    float recall75_ = 0.0f;
    float avgCat_ = 0.0f;
    float avgObj_ = 0.0f;
    float avgAnyObj_ = 0.0f;
    int count_ = 0;
    int classCount_ = 0;
};

} // px

#endif // PIXIENN_YOLOLAYER_H
