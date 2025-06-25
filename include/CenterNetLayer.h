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

#pragma once

#include "Detection.h"
#include "event.pb.h"
#include "Layer.h"

using namespace tensorflow;

namespace px {

template<Device D>
class CenterNetExtras
{
};

template<Device D = Device::CPU>
class CenterNetLayer : public Layer<D>, public Detector, public CenterNetExtras<D>
{
public:
    using V = typename Layer<D>::V;

    CenterNetLayer(Model<D>& model, const YAML::Node& layerDef);

    void forward(const V& input) override;
    void backward(const V& input, V* grad) override;

    bool hasCost() const noexcept override
    {
        return true;
    }

    std::ostream& print(std::ostream& os) override;

    void addDetects(Detections& detections, float threshold) override;
    void addDetects(Detections& detections, int width, int height, float threshold) override;

private:
    void forwardCpu(const PxCpuVector& input);

    void addDetects(Detections& detections, int batch, int width, int height, float threshold,
                    const float* predictions) const;
    void addDetects(Detections& detections, int batch, float threshold, const float* predictions) const;
};

template<Device D>
CenterNetLayer<D>::CenterNetLayer(Model<D>& model, const YAML::Node& layerDef) : Layer<D>(model, layerDef)
{
}

template<Device D>
void CenterNetLayer<D>::forward(const CenterNetLayer::V& input)
{
    Layer<D>::forward(input);
}


template<Device D>
void CenterNetLayer<D>::forwardCpu(const PxCpuVector& input)
{
}

template<Device D>
void CenterNetLayer<D>::backward(const CenterNetLayer::V& input, CenterNetLayer::V* grad)
{
    Layer<D>::backward(input, grad);
}

template<Device D>
void CenterNetLayer<D>::addDetects(Detections& detections, float threshold)
{
}

template<Device D>
void CenterNetLayer<D>::addDetects(Detections& detections, int width, int height, float threshold)
{
}

template<Device D>
void CenterNetLayer<D>::addDetects(Detections& detections, int batch, int width, int height, float threshold,
                                   const float* predictions) const
{
}

template<Device D>
void CenterNetLayer<D>::addDetects(Detections& detections, int batch, float threshold, const float* predictions) const
{
}

template<Device D>
std::ostream& CenterNetLayer<D>::print(std::ostream& os)
{
    return os;
}


} // px
