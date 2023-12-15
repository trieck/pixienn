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

#include "Model.h"
#include "YoloLayer.h"

namespace px {

YoloLayer::YoloLayer(Model& model, const YAML::Node& layerDef) : Layer(model, layerDef)
{
}

void YoloLayer::setup()
{
    activation_ = Activation::get("logistic");
    anchors_ = property<std::vector<int>>("anchors");
    mask_ = property<std::vector<int>>("mask");
    total_ = property<int>("num", 1);

    PX_CHECK(anchors_.size() == total_ * 2, "Anchors size must be twice num size.");

    auto nclasses = classes();
    setOutChannels(mask_.size() * (nclasses + 5));
    setOutHeight(height());
    setOutWidth(width());
    setOutputs(outHeight() * outWidth() * outChannels() * (nclasses + 4 + 1));

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

std::ostream& YoloLayer::print(std::ostream& os)
{
    Layer::print(os, "yolo", { height(), width(), channels() }, { outHeight(), outWidth(), outChannels() });

    return os;
}

void YoloLayer::forward(const PxCpuVector& input)
{
    Layer::forward(input);

    std::copy(input.begin(), input.end(), output_.begin());

    auto area = std::max(1, width() * height());
    auto nclasses = classes();

    auto* poutput = output_.data();
    for (auto b = 0; b < batch(); ++b) {
        for (auto n = 0; n < mask_.size(); ++n) {
            auto index = entryIndex(b, n * area, 0);
            auto* start = poutput + index;
            auto* end = start + 2 * area + 1;
            activation_->apply(start, end);
            index = entryIndex(b, n * area, 4);
            start = poutput + index;
            end = start + (1 + nclasses) * area + 1;
            activation_->apply(start, end);
        }
    }
}

void YoloLayer::backward(const PxCpuVector& input)
{
}


#ifdef USE_CUDA
void YoloLayer::forwardGpu(const PxCudaVector& input)
{
    outputGpu_.copy(input);

    auto area = std::max(1, width() * height());
    auto nclasses = classes();

    auto* poutput = outputGpu_.data();
    for (auto b = 0; b < batch(); ++b) {
        for (auto n = 0; n < mask_.size(); ++n) {
            auto index = entryIndex(b, n * area, 0);
            auto* start = poutput + index;
            activation_->applyGpu(start, 2 * area);
            index = entryIndex(b, n * area, 4);
            start = poutput + index;
            activation_->applyGpu(start, (1 + nclasses) * area + 1);
        }
    }
}
#endif // USE_CUDA

int YoloLayer::entryIndex(int batch, int location, int entry) const noexcept
{
    auto area = std::max(1, width() * height());

    auto n = location / area;
    int loc = location % area;

    return batch * outputs() + n * area * (classes() + 5) + entry * area + loc;
}

cv::Rect
YoloLayer::yoloBox(const float* p, int mask, int index, int col, int row, int w,
                   int h) const
{
    const auto stride = width() * height();
    const auto netW = model().width();
    const auto netH = model().height();

    int newW, newH;
    if (((float) netW / w) < ((float) netH / h)) {
        newW = netW;
        newH = (h * netW) / w;
    } else {
        newH = netH;
        newW = (w * netH) / h;
    }

    auto x = (col + p[index + 0 * stride]) / width();
    x = (x - (netW - newW) / 2.0f / netW) / ((float) newW / netW);

    auto y = (row + p[index + 1 * stride]) / height();
    y = (y - (netH - newH) / 2.0f / netH) / ((float) newH / netH);

    auto width = std::exp(p[index + 2 * stride]) * anchors_[2 * mask] / netW;
    width *= (float) netW / newW;

    auto height = std::exp(p[index + 3 * stride]) * anchors_[2 * mask + 1] / netH;
    height *= (float) netH / newH;

    auto left = std::max<int>(0, (x - width / 2) * w);
    auto right = std::min<int>(w - 1, (x + width / 2) * w);
    auto top = std::max<int>(0, (y - height / 2) * h);
    auto bottom = std::min<int>(h - 1, (y + height / 2) * h);

    cv::Rect b;
    b.x = left;
    b.y = top;
    b.width = right - left;
    b.height = bottom - top;

    return b;
}

void YoloLayer::addDetects(Detections& detections, int width, int height, float threshold)
{
    const auto* predictions = output_.data();
    addDetects(detections, width, height, threshold, predictions);
}

void
YoloLayer::addDetects(Detections& detections, int width, int height, float threshold,
                      const float* predictions) const
{
    auto area = std::max(1, this->width() * this->height());
    auto nclasses = classes();

    for (auto i = 0; i < area; ++i) {
        auto row = i / this->width();
        auto col = i % this->width();

        for (auto n = 0; n < mask_.size(); ++n) {
            auto objIndex = entryIndex(0, n * area + i, 4);
            auto objectness = predictions[objIndex];
            if (objectness < threshold) {
                continue;
            }

            auto boxIndex = entryIndex(0, n * area + i, 0);
            auto box = yoloBox(predictions, mask_[n], boxIndex, col, row, width, height);

            Detection det(nclasses, box);

            int max = 0;
            for (auto j = 0; j < nclasses; ++j) {
                int clsIndex = entryIndex(0, n * area + i, 5 + j);
                det[j] = objectness * predictions[clsIndex];
                if (det[j] > det[max]) {
                    max = j;
                }
            }

            if (det[max] >= threshold) {
                det.setMaxClass(max);
                detections.emplace_back(std::move(det));
            }
        }
    }
}

#ifdef USE_CUDA

void YoloLayer::addDetectsGpu(Detections& detections, int width, int height, float threshold)
{
    auto predv = outputGpu_.asVector();
    addDetects(detections, width, height, threshold, predv.data());
}

#endif // USE_CUDA

} // px
