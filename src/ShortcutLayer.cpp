/********************************************************************************
* Copyright 2023 Thomas A. Rieck, All Rights Reserved
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

#include "Model.h"
#include "ShortcutLayer.h"

namespace px {

ShortcutLayer::ShortcutLayer(Model& model, const YAML::Node& layerDef) : Layer(model, layerDef)
{
}

void ShortcutLayer::setup()
{
    auto activation = property<std::string>("activation", "linear");
    activationFnc_ = Activations::get(activation);

    alpha_ = property<float>("alpha", 1.0f);
    beta_ = property<float>("beta", 1.0f);

    auto index = property<int>("from");
    if (index < 0) {   // relative backwards
        index = this->index() + index;
    }

    PX_CHECK(index < model().layerSize(), "Layer index out of range.");
    PX_CHECK(index < this->index(), "Layer index ahead of current layer.");

    from_ = model().layerAt(index);

    setOutChannels(channels());
    setOutHeight(height());
    setOutWidth(width());
    setOutputs(outHeight() * outWidth() * outChannels());

#ifdef USE_CUDA
    if (useGpu()) {
        outputGpu_ = PxCudaVector(batch() * outChannels() * outHeight() * outWidth());
    } else {
        output_ = PxCpuVector(batch() * outChannels() * outHeight() * outWidth());
    }
#else
    output_ = PxCpuVector(batch() * outHeight() * outWidth() * outChannels(), 0.0f);
    delta_ = PxCpuVector(batch() * outHeight() * outWidth() * outChannels(), 0.0f);
#endif
}

std::ostream& ShortcutLayer::print(std::ostream& os)
{
    Layer::print(os, "shortcut", { height(), width(), channels() }, { outHeight(), outWidth(), outChannels() });

    return os;
}

void ShortcutLayer::forward(const PxCpuVector& input)
{
    Layer::forward(input);

    output_.copy(input);

    auto ctxt = makeContext(input);

    shortcutForward(ctxt);

    activationFnc_->apply(output_);
}

void ShortcutLayer::backward(const PxCpuVector& input)
{
    Layer::backward(input);

    activationFnc_->gradient(output_, delta_);

    cblas_saxpy(batch() * outputs(), alpha_, delta_.data(), 1, model().delta()->data(), 1);

    auto ctxt = makeContext(input);

    shortcutBackward(ctxt);
}

ShortcutContext ShortcutLayer::makeContext(const PxCpuVector&)
{
    ShortcutContext ctxt;
    ctxt.output = &output_;
    ctxt.delta = &delta_;
    ctxt.from = &from_->output();
    ctxt.fromDelta = from_->delta();
    ctxt.alpha = alpha_;
    ctxt.beta = beta_;
    ctxt.batch = batch();
    ctxt.channels = channels();
    ctxt.height = height();
    ctxt.width = width();
    ctxt.outChannels = outChannels();
    ctxt.outHeight = outHeight();
    ctxt.outWidth = outWidth();

    return ctxt;
}

#ifdef USE_CUDA

void ShortcutLayer::forwardGpu(const PxCudaVector& input)
{
    outputGpu_.copy(input);

    auto ctxt = makeContext(input);

    shortcutForwardGpu(ctxt);

    activationFnc_->applyGpu(outputGpu_);
}

ShortcutContext ShortcutLayer::makeContext(const PxCudaVector&)
{
    ShortcutContext ctxt;
    ctxt.outputGpu = &outputGpu_;
    ctxt.addGpu = &from_->outputGpu();
    ctxt.alpha = alpha_;
    ctxt.beta = beta_;
    ctxt.batch = batch();
    ctxt.channels = channels();
    ctxt.height = height();
    ctxt.width = width();
    ctxt.outChannels = outChannels();
    ctxt.outHeight = outHeight();
    ctxt.outWidth = outWidth();

    return ctxt;
}

#endif  // USE_CUDA

}   // px
