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

#include "ConfusionMatrix.h"
#include "Error.h"

namespace px {

ConfusionMatrix::ConfusionMatrix()
{
}

ConfusionMatrix::ConfusionMatrix(int numClasses) : numClasses_(numClasses)
{
    resize(numClasses);
}

void ConfusionMatrix::resize(int numClasses)
{
    numClasses_ = numClasses;
    matrix_.clear();
    matrix_.resize(numClasses + 1, std::vector<int>(numClasses + 1, 0));
}

void ConfusionMatrix::update(int trueClass, int predictedClass)
{
    PX_CHECK(trueClass < numClasses_ && predictedClass < numClasses_, "Index out of range");
    PX_CHECK(trueClass >= -1 && predictedClass >= -1, "Index out of range");
    PX_CHECK(!(trueClass == -1 && predictedClass == -1), "Index out of range");

    if (trueClass == -1) {              // a "ghost prediction", a type of false positive
        trueClass = numClasses_;
    } else if (predictedClass == -1) {  // an "undetected object", a type of false negative
        predictedClass = numClasses_;
    }

    matrix_[trueClass][predictedClass]++;
}

int ConfusionMatrix::TP(int clsIndex) const
{
    PX_CHECK(clsIndex < numClasses_, "Index out of range");
    return matrix_[clsIndex][clsIndex];
}

int ConfusionMatrix::FP(int clsIndex) const
{
    PX_CHECK(clsIndex < numClasses_, "Index out of range");

    auto falsePositives = 0;
    for (auto i = 0; i <= numClasses_; ++i) {
        if (i != clsIndex) {
            falsePositives += matrix_[clsIndex][i];
        }
    }

    return falsePositives;
}

int ConfusionMatrix::FN(int clsIndex) const
{
    PX_CHECK(clsIndex <= numClasses_, "Index out of range");

    auto falseNegatives = 0;
    for (auto i = 0; i <= numClasses_; ++i) {
        if (i != clsIndex) {
            falseNegatives += matrix_[i][clsIndex];
        }
    }

    return falseNegatives;
}

float ConfusionMatrix::precision(int clsIndex) const
{
    PX_CHECK(clsIndex <= numClasses_, "Index out of range");

    auto truePos = matrix_[clsIndex][clsIndex];
    auto falsePos = 0;

    for (auto i = 0; i <= numClasses_; ++i) {
        if (i != clsIndex) {
            falsePos += matrix_[i][clsIndex];
        }
    }

    if (truePos + falsePos == 0) {
        return 0.0f;
    }

    return static_cast<float>(truePos) / (truePos + falsePos);
}

float ConfusionMatrix::recall(int clsIndex) const
{
    PX_CHECK(clsIndex <= numClasses_, "Index out of range");

    auto truePos = matrix_[clsIndex][clsIndex];
    auto falseNeg = 0;

    for (auto i = 0; i <= numClasses_; ++i) {
        if (i != clsIndex) {
            falseNeg += matrix_[clsIndex][i];
        }
    }

    if (truePos + falseNeg == 0) {
        return 0.0;
    }

    return static_cast<float>(truePos) / (truePos + falseNeg);
}

int ConfusionMatrix::undetected(int clsIndex) const
{
    PX_CHECK(clsIndex < numClasses_, "Index out of range");

    return matrix_[clsIndex][numClasses_];
}

int ConfusionMatrix::ghosts(int clsIndex) const
{
    PX_CHECK(clsIndex < numClasses_, "Index out of range");

    return matrix_[numClasses_][clsIndex];
}

void ConfusionMatrix::reset()
{
    for (auto& v: matrix_) {
        std::fill(v.begin(), v.end(), 0);
    }
}

float ConfusionMatrix::avgRecall(int classes) const
{
    if (classes == -1) {
        classes = numClasses_;
    }

    if (classes == 0) {
        return 0.0f;
    }

    auto totalRecall = 0.0f;
    for (auto i = 0; i < classes; ++i) {
        auto recallValue = recall(i);
        totalRecall += recallValue;
    }

    return totalRecall / classes;
}

float ConfusionMatrix::mAP(int classes) const
{
    if (classes == -1) {
        classes = numClasses_;
    }

    if (classes == 0) {
        return 0.0f;
    }

    auto totalPrecision = 0.0f;
    for (int i = 0; i < classes; ++i) {
        auto precisionValue = precision(i);
        totalPrecision += precisionValue;
    }

    return totalPrecision / classes;
}

float ConfusionMatrix::F1(int clsIndex) const
{
    auto precisionValue = precision(clsIndex);
    auto recallValue = recall(clsIndex);

    if (precisionValue + recallValue == 0) {
        return 0.0f;
    }

    return 2 * (precisionValue * recallValue) / (precisionValue + recallValue);
}

float ConfusionMatrix::microAvgF1() const
{
    auto truePositives = 0.0f;
    auto falsePositives = 0.0f;
    auto falseNegatives = 0.0f;

    for (auto i = 0; i < numClasses_; ++i) {
        truePositives += matrix_[i][i];
        for (int j = 0; j < numClasses_; ++j) {
            if (j != i) {
                falsePositives += matrix_[j][i];
                falseNegatives += matrix_[i][j];
            }
        }
    }

    float precision = truePositives / (truePositives + falsePositives);
    float recall = truePositives / (truePositives + falseNegatives);

    if (precision + recall == 0) {
        return 0.0f;
    }

    return 2 * (precision * recall) / (precision + recall);
}

int ConfusionMatrix::classes() const noexcept
{
    return numClasses_;
}

}   // px
