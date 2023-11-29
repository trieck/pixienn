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

#include "Model.h"
#include "ShortcutLayer.h"

namespace px {

ShortcutLayer::ShortcutLayer(const Model& model, const YAML::Node& layerDef) : Layer(model, layerDef)
{
}

void ShortcutLayer::setup()
{
    auto activation = property<std::string>("activation", "linear");
    activationFnc_ = Activation::get(activation);

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

    output_ = PxCpuVector(batch() * outChannels() * outHeight() * outWidth());

#ifdef USE_CUDA
    if (useGpu()) {
        outputGpu_ = PxCudaVector(batch() * outChannels() * outHeight() * outWidth());
    }
#endif
}

std::ostream& ShortcutLayer::print(std::ostream& os)
{
    Layer::print(os, "shortcut", { height(), width(), channels() }, { outHeight(), outWidth(), outChannels() });

    return os;
}

void ShortcutLayer::forward(const PxCpuVector& input)
{
    output_.copy(input);

    auto ctxt = makeContext(input);

    shortcutForward(ctxt);

    activationFnc_->apply(output_);
}

ShortcutContext ShortcutLayer::makeContext(const PxCpuVector&)
{
    ShortcutContext ctxt;
    ctxt.output = &output_;
    ctxt.add = &from_->output();
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
