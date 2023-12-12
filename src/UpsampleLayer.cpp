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

#include "UpsampleLayer.h"

namespace px {

UpsampleLayer::UpsampleLayer(Model& model, const YAML::Node& layerDef) : Layer(model, layerDef),
                                                                         stride_(0), scale_(0)
{
}

void UpsampleLayer::setup()
{
    scale_ = property("scale", 1.0f);
    stride_ = property("stride", 2);    // FIXME: does not support negative stride (reverse upsample)

    setOutChannels(channels());
    setOutHeight(height() * stride_);
    setOutWidth(width() * stride_);
    setOutputs(outHeight() * outWidth() * outChannels());

#ifdef USE_CUDA
    if (useGpu()) {
        outputGpu_ = PxCudaVector(batch() * outChannels() * outHeight() * outWidth());
    } else {
        output_ = PxCpuVector(batch() * outChannels() * outHeight() * outWidth());
    }
#else
    output_ = PxCpuVector(batch() * outChannels() * outHeight() * outWidth());
#endif
}

std::ostream& UpsampleLayer::print(std::ostream& os)
{
    Layer::print(os, "upsample", { height(), width(), channels() }, { outHeight(), outWidth(), outChannels() });

    return os;
}

void UpsampleLayer::forward(const PxCpuVector& input)
{
    Layer::forward(input);

    auto ctxt = makeContext(input);
    upsampleForward(ctxt);
}

void UpsampleLayer::backward(const PxCpuVector& input)
{
}

#ifdef USE_CUDA

void UpsampleLayer::forwardGpu(const PxCudaVector& input)
{
    auto ctxt = makeContext(input);
    upsampleForwardGpu(ctxt);
}

UpsampleContext UpsampleLayer::makeContext(const PxCudaVector& input)
{
    UpsampleContext ctxt;

    ctxt.batch = batch();
    ctxt.channels = channels();
    ctxt.forward = true;
    ctxt.height = height();
    ctxt.inputGpu = &input;
    ctxt.outChannels = outChannels();
    ctxt.outHeight = outHeight();
    ctxt.outWidth = outWidth();
    ctxt.outputGpu = &outputGpu_;
    ctxt.scale = scale_;
    ctxt.stride = stride_;
    ctxt.width = width();

    return ctxt;
}

#endif  // USE_CUDA

UpsampleContext UpsampleLayer::makeContext(const PxCpuVector& input)
{
    UpsampleContext ctxt;

    ctxt.batch = batch();
    ctxt.channels = channels();
    ctxt.forward = true;
    ctxt.height = height();
    ctxt.input = &input;
    ctxt.outChannels = outChannels();
    ctxt.outHeight = outHeight();
    ctxt.outWidth = outWidth();
    ctxt.output = &output_;
    ctxt.scale = scale_;
    ctxt.stride = stride_;
    ctxt.width = width();

    return ctxt;
}

} // px
