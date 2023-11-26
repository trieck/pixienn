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

#include "MaxPoolLayer.h"

#ifdef USE_CUDA

#include "PoolKernels.cuh"

#endif // USE_CUDA

namespace px {

MaxPoolLayer::MaxPoolLayer(const Model& model, const YAML::Node& layerDef) : Layer(model, layerDef)
{
}

void MaxPoolLayer::setup()
{
    kernel_ = property<int>("kernel", 1);
    stride_ = property<int>("stride", 1);
    padding_ = property<int>("padding", std::max<int>(0, kernel_ - 1));

    setOutChannels(channels());
    setOutHeight((height() + padding_ - kernel_) / stride_ + 1);
    setOutWidth((width() + padding_ - kernel_) / stride_ + 1);

    setOutputs(batch() * outChannels() * outHeight() * outWidth());

    output_ = PxCpuVector(batch() * outChannels() * outHeight() * outWidth());

#ifdef USE_CUDA
    if (useGpu()) {
        outputGpu_ = PxCudaVector(batch() * outChannels() * outHeight() * outWidth());
    }
#endif
}

std::ostream& MaxPoolLayer::print(std::ostream& os)
{
    Layer::print(os, "maxpool", { height(), width(), channels() }, { outHeight(), outWidth(), outChannels() },
                 std::nullopt, std::array<int, 3>{ kernel_, kernel_, stride_ });

    return os;
}

void MaxPoolLayer::forward(const PxCpuVector& input)
{
    auto ctxt = makeContext(input);
    maxPoolForward(ctxt);
}

MaxPoolContext MaxPoolLayer::makeContext(const PxCpuVector& input)
{
    MaxPoolContext ctxt;

    ctxt.input = &input;
    ctxt.output = &output_;
    ctxt.batch = batch();
    ctxt.channels = outChannels();
    ctxt.height = height();
    ctxt.width = width();
    ctxt.outHeight = outHeight();
    ctxt.outWidth = outWidth();
    ctxt.kernel = kernel_;
    ctxt.stride = stride_;
    ctxt.padding = padding_;

    return ctxt;
}

#ifdef USE_CUDA

MaxPoolContext MaxPoolLayer::makeContext(const PxCudaVector& input)
{
    MaxPoolContext ctxt;

    ctxt.inputGpu = &input;
    ctxt.outputGpu = &outputGpu_;
    ctxt.batch = batch();
    ctxt.channels = outChannels();
    ctxt.height = height();
    ctxt.width = width();
    ctxt.outHeight = outHeight();
    ctxt.outWidth = outWidth();
    ctxt.kernel = kernel_;
    ctxt.stride = stride_;
    ctxt.padding = padding_;

    return ctxt;
}

void MaxPoolLayer::forwardGpu(const PxCudaVector& input)
{
    auto ctxt = makeContext(input);
    maxPoolForwardGpu(ctxt);
}

#endif  // USE_CUDA

} // px
