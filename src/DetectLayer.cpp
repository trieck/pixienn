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

#include <cblas.h>

#include "DetectLayer.h"
#include "Model.h"
#include "Math.h"
#include "Box.h"

namespace px {

DetectLayer::DetectLayer(Model& model, const YAML::Node& layerDef) : Layer(model, layerDef)
{
}

void DetectLayer::setup()
{
    classScale_ = property<float>("class_scale", 1.0f);
    coordScale_ = property<float>("coord_scale", 1.0f);
    coords_ = property<int>("coords", 1);
    forced_ = property<bool>("forced", false);
    jitter_ = property<float>("jitter", 0.2f);
    noObjectScale_ = property<float>("noobject_scale", 1.0f);
    num_ = property<int>("num", 1);
    objectScale_ = property<float>("object_scale", 1.0f);
    random_ = property<bool>("random", false);
    reorg_ = property<bool>("reorg", false);
    rescore_ = property<bool>("rescore", false);
    side_ = property<int>("side", 7);
    softmax_ = property<bool>("softmax", false);
    sqrt_ = property<bool>("sqrt", false);

    setOutChannels(channels());
    setOutHeight(height());
    setOutWidth(width());
    setOutputs(inputs());

#ifdef USE_CUDA
    if (useGpu()) {
        outputGpu_ = PxCudaVector(outputs(), 0.0f);
    } else {
        output_ = PxCpuVector(batch() * outputs(), 0.0f);
        delta_ = PxCpuVector(batch() * outputs(), 0.0f);
    }
#else
    output_ = PxCpuVector(batch() * outputs(), 0.0f);
    delta_ = PxCpuVector(batch() * outputs(), 0.0f);
#endif
}

std::ostream& DetectLayer::print(std::ostream& os)
{
    Layer::print(os, "detection", { height(), width(), channels() }, { outHeight(), outWidth(), outChannels() });

    return os;
}

DetectContext DetectLayer::makeContext(const PxCpuVector& input)
{
    DetectContext ctxt{};

    ctxt.avgAllCat = 0;
    ctxt.avgAnyObj = 0;
    ctxt.avgCat = 0;
    ctxt.avgIoU = 0;
    ctxt.avgObj = 0;
    ctxt.batch = batch();
    ctxt.classScale = classScale_;
    ctxt.classes = classes();
    ctxt.coordScale = coordScale_;
    ctxt.coords = coords_;
    ctxt.count = 0;
    ctxt.delta = &delta_;
    ctxt.forced = forced_;
    ctxt.groundTruths = &groundTruth();
    ctxt.input = &input;
    ctxt.inputs = inputs();
    ctxt.jitter = jitter_;
    ctxt.netDelta = model().delta();
    ctxt.noObjectScale = noObjectScale_;
    ctxt.num = num_;
    ctxt.objectScale = objectScale_;
    ctxt.output = &output_;
    ctxt.outputs = outputs();
    ctxt.random = random_;
    ctxt.reorg = reorg_;
    ctxt.rescore = rescore_;
    ctxt.side = side_;
    ctxt.softmax = softmax_;
    ctxt.sqrt = sqrt_;
    ctxt.training = training();

    return ctxt;
}

void DetectLayer::forward(const PxCpuVector& input)
{
    Layer::forward(input);

    auto ctxt = makeContext(input);

    detectForward(ctxt);

    if (training()) {
        if (gradientClipping_) {
            clipGradients();
        }

        cost_ = std::pow(magArray(delta_.data(), delta_.size()), 2);

        printStats(ctxt);
    }
}

void DetectLayer::printStats(const DetectContext& ctxt)
{
    auto locations = side_ * side_;

    if (ctxt.count > 0) {
        printf("Detection: Avg. IoU: %f, Pos Cat: %f, All Cat: %f, Pos Obj: %f, Any Obj: %f, count: %d\n",
               (ctxt.avgIoU / ctxt.count),
               (ctxt.avgCat / ctxt.count),
               (ctxt.avgAllCat / (ctxt.count * classes())),
               (ctxt.avgObj / ctxt.count),
               (ctxt.avgAnyObj / (batch() * locations * num_)),
               ctxt.count);
    }
}


void DetectLayer::backward(const PxCpuVector& input)
{
    Layer::backward(input);

    auto ctxt = makeContext(input);

    detectBackward(ctxt);
}

#ifdef USE_CUDA
void DetectLayer::forwardGpu(const PxCudaVector& input)
{
    outputGpu_.copy(input);
}
#endif  // USE_CUDA

void DetectLayer::addDetects(Detections& detections, int width, int height, float threshold)
{
    addDetects(detections, width, height, threshold, output_.data());
}

void DetectLayer::addDetects(Detections& detections, float threshold)
{
    addDetects(detections, threshold, output_.data());
}

#ifdef USE_CUDA

void DetectLayer::addDetectsGpu(Detections& detections, int width, int height, float threshold)
{
    auto predv = outputGpu_.asVector();
    addDetects(detections, width, height, threshold, predv.data());
}

#endif  // USE_CUDA

void DetectLayer::addDetects(Detections& detections, int width, int height, float threshold,
                             const float* predictions) const
{
    PredictContext ctxt{};
    ctxt.classes = classes();
    ctxt.coords = coords_;
    ctxt.detections = &detections;
    ctxt.height = height;
    ctxt.num = num_;
    ctxt.predictions = predictions;
    ctxt.side = side_;
    ctxt.sqrt = sqrt_;
    ctxt.threshold = threshold;
    ctxt.width = width;

    detectAddPredicts(ctxt);
}

void DetectLayer::addDetects(Detections& detections, float threshold, const float* predictions) const
{
    PredictContext ctxt{};
    ctxt.classes = classes();
    ctxt.coords = coords_;
    ctxt.detections = &detections;
    ctxt.num = num_;
    ctxt.side = side_;
    ctxt.predictions = predictions;
    ctxt.threshold = threshold;

    detectAddRawPredicts(ctxt);
}

}   // px
