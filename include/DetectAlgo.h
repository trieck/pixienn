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


#ifndef PIXIENN_DETECT_ALGO_H
#define PIXIENN_DETECT_ALGO_H

#include "Detection.h"
#include "PxTensor.h"
#include "TrainBatch.h"

namespace px {

// Represents the context needed for a detect operation
struct DetectContext
{
    const PxCpuVector* input = nullptr;
    PxCpuVector* output = nullptr;
    PxCpuVector* delta = nullptr;
    const GroundTruths* groundTruths = nullptr;
    PxCpuVector* netDelta = nullptr;

    bool forced = false;
    bool random = false;
    bool reorg = false;
    bool rescore = false;
    bool softmax = false;
    bool sqrt = false;
    float classScale = 1.0f;
    float coordScale = 1.0f;
    float jitter = 0.0f;
    float noObjectScale = 1.0f;
    float objectScale = 1.0f;
    float* cost = nullptr;
    int batch = 0;
    int classes = 0;
    int coords = 0;    // number of coordinates in box
    int inputs = 0;
    int num = 0;       // number of boxes a grid-cell is responsible for predicting
    int outputs = 0;
    int side;          // the length of a side

    // statistics
    float avgAllCat = 0.0f;
    float avgAnyObj = 0.0f;
    float avgCat = 0.0f;
    float avgIoU = 0.0f;
    float avgObj = 0.0f;
    int count = 0;

    bool training = false;
};

struct PredictContext
{
    Detections* detections = nullptr;
    bool sqrt = false;
    const float* predictions = nullptr;
    float threshold = 0.0f;
    int classes = 0;
    int coords = 0;
    int height = 0;
    int num = 0;
    int side = 0;
    int width = 0;
};

void detectForward(DetectContext& ctxt);
void detectBackward(DetectContext& ctxt);
void detectAddPredicts(const PredictContext& ctxt);
void detectAddRawPredicts(const PredictContext& ctxt);

}   // px


#endif // PIXIENN_DETECT_ALGO_H
