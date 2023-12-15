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
#include <opencv2/core/types.hpp>

#include "Box.h"
#include "Common.h"
#include "DetectAlgo.h"
#include "Math.h"

namespace px {

struct GroundTruthContext
{
    const DetectContext* ctxt;
    const GroundTruth* bestGT;
    cv::Rect2f pred;
    float bestIoU;
    int batch;
    int gridIndex;
};

struct GroundTruthResult
{
    const GroundTruth* gt;
    float bestIoU;
};

static void resetStats(DetectContext& ctxt);
static void processDetects(DetectContext& ctxt, int b, int i);
static cv::Rect2f makeBox(const float* pbox);
static cv::Rect2f predBox(const DetectContext& ctxt, const float* pbox);
static GroundTruthResult groundTruth(const GroundTruthContext& ctxt);

void detectForward(DetectContext& ctxt)
{
    if (ctxt.softmax) {
        ctxt.output->copy(softmax(*ctxt.input));
    } else {
        ctxt.output->copy(*ctxt.input);
    }

    if (!ctxt.training) {
        return;
    }

    resetStats(ctxt);

    auto locations = ctxt.side * ctxt.side;
    for (auto b = 0; b < ctxt.batch; ++b) {
        for (auto i = 0; i < locations; ++i) {
            processDetects(ctxt, b, i);
        }
    }

    *(ctxt.cost) = std::pow(magArray(ctxt.delta->data(), ctxt.delta->size()), 2);
}

void processDetects(DetectContext& ctxt, int b, int i)
{
    // The output tensor has dimensions [S, S, (B * 5 + C)].
    //  - S represents the size of each grid cell.
    //  - B is the number of bounding boxes predicted by each grid cell.
    //  - C is the number of classes.

    // The tensor layout is as follows:
    //   - Class probabilities for each grid cell (C x S x S).
    //   - Objectness score for each bounding box (B x S x S).
    //   - Coordinates of each bounding box ([x, y, w, h] x (B x S x S)).
    //
    // The tensor is organized in memory as a contiguous block with the following structure:
    //  ---------------------------------------------------------------------------------------------
    //  |   Class Probs (C x S x S)  |     Objectness (B x S x S)     | Coordinates (B x S x S x 4) |
    //  ---------------------------------------------------------------------------------------------
    //  |<----- C*S*S elements ----->|<------- B*S*S elements ------->|<----- B*S*S*4 elements ----->

    const auto* poutput = ctxt.output->data();
    const auto index = b * i;
    const auto nclasses = ctxt.classes;
    const auto locations = ctxt.side * ctxt.side;
    auto* pdelta = ctxt.delta->data();
    auto bestJ = 0;

    const GroundTruth* gt = nullptr;
    GroundTruthContext gtCtxt;
    gtCtxt.ctxt = &ctxt;
    gtCtxt.gridIndex = i;
    gtCtxt.bestGT = gt;
    gtCtxt.bestIoU = -std::numeric_limits<float>::max();
    gtCtxt.batch = b;

    for (auto j = 0; j < ctxt.num; ++j) {
        auto pobject = index + locations * nclasses + i * ctxt.num + j;
        pdelta[pobject] = ctxt.noObjectScale * (0 - poutput[pobject]);
        *ctxt.cost += ctxt.noObjectScale * std::pow(poutput[pobject], 2);
        ctxt.avgAnyObj += poutput[pobject];

        auto boxIndex = index + locations * (nclasses + ctxt.num) + (i * ctxt.num + j) * ctxt.coords;
        gtCtxt.pred = predBox(ctxt, poutput + boxIndex);
        auto result = groundTruth(gtCtxt);
        if (result.bestIoU > gtCtxt.bestIoU) {
            gtCtxt.bestIoU = result.bestIoU;
            gt = gtCtxt.bestGT = result.gt;
            bestJ = j;
        }
    }

    if (gt == nullptr) {
        return;     // no ground truth in this grid cell, do not penalize
    }

    // Compute the class loss
    auto classIndex = index + i * nclasses;
    for (auto j = 0; j < nclasses; ++j) {
        float netTruth = gt->classId == j ? 1.0f : 0.0f;
        pdelta[classIndex + j] = ctxt.classScale * (netTruth - poutput[classIndex + j]);
        *(ctxt.cost) += ctxt.classScale * pow(netTruth - poutput[classIndex + j], 2);
        if (netTruth) {
            ctxt.avgCat += poutput[classIndex + j];
        }
        ctxt.avgAllCat += poutput[classIndex + j];
    }

    cv::Rect2f truthBox(gt->box);
    truthBox.x /= ctxt.side;
    truthBox.y /= ctxt.side;

    auto row = i / ctxt.side;
    auto col = i % ctxt.side;
    auto truthRow = (int) (gt->box.y * ctxt.side);
    auto truthCol = (int) (gt->box.x * ctxt.side);

    PX_CHECK(row == truthRow, "The ground truth box is not in the grid cell row.");
    PX_CHECK(col == truthCol, "The ground truth box is not in the grid cell column.");

    auto pobject = index + locations * nclasses + i * ctxt.num + bestJ;
    auto boxIndex = index + locations * (nclasses + ctxt.num) + (i * ctxt.num + bestJ) * ctxt.coords;

    auto pred = predBox(ctxt, poutput + boxIndex);
    auto iou = boxIou(pred, truthBox);

    *(ctxt.cost) -= ctxt.noObjectScale * std::pow(poutput[pobject], 2);
    *(ctxt.cost) += ctxt.objectScale * std::pow(1 - poutput[pobject], 2);
    ctxt.avgObj += poutput[pobject];
    pdelta[pobject] = ctxt.objectScale * (1. - poutput[pobject]);

    if (ctxt.rescore) {
        pdelta[pobject] = ctxt.objectScale * (iou - poutput[pobject]);
    }

    pdelta[boxIndex + 0] = ctxt.coordScale * (gt->box.x - poutput[boxIndex + 0]);
    pdelta[boxIndex + 1] = ctxt.coordScale * (gt->box.y - poutput[boxIndex + 1]);
    pdelta[boxIndex + 2] = ctxt.coordScale * (gt->box.width - poutput[boxIndex + 2]);
    pdelta[boxIndex + 3] = ctxt.coordScale * (gt->box.height - poutput[boxIndex + 3]);

    if (ctxt.sqrt) {
        pdelta[boxIndex + 2] = ctxt.coordScale * (std::sqrt(gt->box.width) - poutput[boxIndex + 2]);
        pdelta[boxIndex + 3] = ctxt.coordScale * (std::sqrt(gt->box.height) - poutput[boxIndex + 3]);
    }

    *(ctxt.cost) += std::pow(1 - iou, 2);
    ctxt.avgIoU += iou;
    ctxt.count++;
}

GroundTruthResult groundTruth(const GroundTruthContext& ctxt)
{
    GroundTruthResult result;
    result.gt = ctxt.bestGT;
    result.bestIoU = ctxt.bestIoU;

    auto row = ctxt.gridIndex / ctxt.ctxt->side;
    auto col = ctxt.gridIndex % ctxt.ctxt->side;

    const auto& gts = (*ctxt.ctxt->groundTruths)[ctxt.batch];
    for (const auto& gt: gts) {
        auto truthRow = static_cast<int>(gt.box.y * ctxt.ctxt->side);
        auto truthCol = static_cast<int>(gt.box.x * ctxt.ctxt->side);
        if (!(truthRow == row && truthCol == col)) {
            continue;
        }

        cv::Rect2f truthBox(gt.box);
        truthBox.x /= ctxt.ctxt->side;
        truthBox.y /= ctxt.ctxt->side;

        auto iou = boxIou(ctxt.pred, truthBox);
        if (iou > result.bestIoU) {
            result.bestIoU = iou;
            result.gt = &gt;
        }
    }

    return result;
}

void resetStats(DetectContext& ctxt)
{
    ctxt.avgAnyObj = ctxt.avgCat = ctxt.avgAllCat = ctxt.avgObj, ctxt.avgIoU = 0;
    ctxt.count = 0;
}

cv::Rect2f makeBox(const float* pbox)
{
    cv::Rect2f box;
    box.x = *pbox++;
    box.y = *pbox++;
    box.width = *pbox++;
    box.height = *pbox++;

    return box;
}

cv::Rect2f predBox(const DetectContext& ctxt, const float* pbox)
{
    auto box = makeBox(pbox);
    box.x /= ctxt.side;
    box.y /= ctxt.side;

    if (ctxt.sqrt) {
        box.width *= box.width;
        box.height *= box.height;
    }

    return box;
}

void detectBackward(DetectContext& ctxt)
{
    auto* pDelta = ctxt.delta->data();
    auto* pNetDelta = ctxt.netDelta;

    PX_CHECK(pNetDelta != nullptr, "Model delta tensor is null");
    PX_CHECK(pNetDelta->data() != nullptr, "Model delta tensor is null");
    PX_CHECK(pDelta != nullptr, "Delta tensor is null");

    const auto n = ctxt.batch * ctxt.inputs;

    PX_CHECK(ctxt.delta->size() >= n, "Delta tensor is too small");
    PX_CHECK(pNetDelta->size() >= n, "Model tensor is too small");

    cblas_saxpy(n, 1, pDelta, 1, pNetDelta->data(), 1);
}

void detectAddPredictions(const PredictContext& ctxt)
{
    auto nclasses = ctxt.classes;

    const auto locations = ctxt.side * ctxt.side;
    for (auto i = 0; i < locations; ++i) {
        auto row = i / ctxt.side;
        auto col = i % ctxt.side;
        for (auto n = 0; n < ctxt.num; ++n) {
            auto pindex = locations * nclasses + i * ctxt.num + n;
            auto scale = ctxt.predictions[pindex];
            auto bindex = locations * (nclasses + ctxt.num) + (i * ctxt.num + n) * ctxt.coords;
            auto x = (ctxt.predictions[bindex + 0] + col) / ctxt.side * ctxt.width;
            auto y = (ctxt.predictions[bindex + 1] + row) / ctxt.side * ctxt.height;
            auto w = pow(ctxt.predictions[bindex + 2], (ctxt.sqrt ? 2 : 1)) * ctxt.width;
            auto h = pow(ctxt.predictions[bindex + 3], (ctxt.sqrt ? 2 : 1)) * ctxt.height;
            auto left = std::max<int>(0, (x - w / 2));
            auto right = std::min<int>(ctxt.width - 1, (x + w / 2));
            auto top = std::max<int>(0, (y - h / 2));
            auto bottom = std::min<int>(ctxt.height - 1, (y + h / 2));
            cv::Rect b{ left, top, right - left, bottom - top };

            Detection det(nclasses, b);
            int max = 0;
            for (auto j = 0; j < nclasses; ++j) {
                auto index = i * nclasses;
                det[j] = scale * ctxt.predictions[index + j];
                if (det[j] > det[max]) {
                    max = j;
                }
            }
            if (det[max] >= ctxt.threshold) {
                det.setMaxClass(max);
                ctxt.detections->emplace_back(std::move(det));
            }
        }
    }
}

}   // px
