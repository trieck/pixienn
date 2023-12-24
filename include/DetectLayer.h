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

#ifndef PIXIENN_DETECTLAYER_H
#define PIXIENN_DETECTLAYER_H

#include "DetectAlgo.h"
#include "Detection.h"
#include "Layer.h"

namespace px {

class DetectLayer : public Layer, public Detector
{
protected:
    DetectLayer(Model& model, const YAML::Node& layerDef);

public:
    ~DetectLayer() override = default;

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
    DetectContext makeContext(const PxCpuVector& input);

    void setup() override;
    void addDetects(Detections& detections, int width, int height, float threshold, const float* predictions) const;
    void addDetects(Detections& detections, float threshold, const float* predictions) const;
    void printStats(const DetectContext& ctxt);

    friend LayerFactories;

    int coords_, num_, side_;
    bool rescore_, softmax_, sqrt_, forced_, random_, reorg_;
    float coordScale_, objectScale_, noObjectScale_, classScale_, jitter_;
};

} // px

#endif // PIXIENN_DETECTLAYER_H
