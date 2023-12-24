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

#ifndef PIXIENN_REGIONLAYER_H
#define PIXIENN_REGIONLAYER_H

#include "Detection.h"
#include "Layer.h"

namespace px {

class RegionLayer : public Layer, public Detector
{
protected:
    RegionLayer(Model& model, const YAML::Node& layerDef);

public:
    ~RegionLayer() override = default;

    std::ostream& print(std::ostream& os) override;
    void forward(const PxCpuVector& input) override;
    void backward(const PxCpuVector& input) override;

    void addDetects(Detections& detections, float threshold) override;
    void addDetects(Detections& detections, int width, int height, float threshold) override;

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
    void processRegion(int b, int i, int j);
    void processObjects(int i);
    cv::Rect2f regionBox(int n, int index, int i, int j);
    float deltaRegionBox(const cv::Rect2f& truth, int n, int index, int i, int j, float scale);
    void deltaRegionClass(const GroundTruth& truth, int index, float scale);

    void resetStats();

    PxCpuTensor<1> biases_, biasUpdates_;
    Activation::Ptr activation_;

    std::vector<float> anchors_;
    bool biasMatch_, softmax_, rescore_, absolute_, random_, focalLoss_;
    int coords_, num_;
    float jitter_, objectScale_, noObjectScale_, classScale_, coordScale_, thresh_;

    float avgAnyObj_ = 0.0f;
    float avgCat_ = 0.0f;
    float avgIoU_ = 0.0f;
    float avgObj_ = 0.0f;
    float recall = 0.0f;
    int count_ = 0, classCount_ = 0;
};

} // px

#endif // PIXIENN_REGIONLAYER_H
