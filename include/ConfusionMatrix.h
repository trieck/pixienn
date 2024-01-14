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

#pragma once

#include "Common.h"

namespace px {

/**
 * @brief Multiclass Confusion Matrix for Object Detection
 *
 * The Multiclass Confusion Matrix (MCM) is designed to evaluate the performance of object detection models
 * across multiple classes. It organizes predictions into an (N+1) * (N+1) matrix, where N is the number of classes.
 * Each cell (i, j) in the matrix represents the count of bounding boxes with predicted class i
 * (or no prediction, if i=N+1), and annotation class j (or no annotation, if j=N+1).
 *
 * This matrix structure allows for a comprehensive analysis of the model's behavior, capturing various scenarios:
 * - True positives (i=j): Instances where the model correctly predicted the specified class.
 * - False positives (i!=j): Instances where the model incorrectly predicted the specified class.
 * - Undetected objects (i=N+1, j): Instances where there is a ground truth for the specified class,
 *   but no corresponding predictions exist. Represented by the last row in the matrix.
 * - Ghost predictions (i, j=N+1): Instances where there are predictions for the specified class,
 *   but no corresponding ground truths exist. Represented by the last column in the matrix.
 *
 * This class provides methods to update the matrix based on ground truth and prediction information,
 * as well as methods to calculate various evaluation metrics such as precision, recall, F1 score,
 * mean Average Precision (mAP), and micro-averaged F1 score across all classes.
 */
class ConfusionMatrix
{
public:
    ConfusionMatrix();
    ConfusionMatrix(int numClasses);

    void resize(int numClasses);
    void update(int trueClass, int predictedClass);
    void reset();

    /**
    * @brief Get the number of true positives (TP) for a specific class.
    *
    * True positives are instances where the model correctly predicted the given class.
    *
    * @param clsIndex The index of the class for which to retrieve true positives.
    * @return The number of true positives for the specified class.
    */
    int TP(int clsIndex) const;

    /**
     * @brief Get the number of false positives (FP) for a specific class.
     *
     * False positives are instances where the model incorrectly predicted the given class.
     *
     * @param clsIndex The index of the class for which to retrieve false positives.
     * @return The number of false positives for the specified class.
     */
    int FP(int clsIndex) const;

    /**
     * @brief Get the number of false negatives (FN) for a specific class.
     *
     * False negatives are instances where the model failed to predict the given class.
     *
     * @param clsIndex The index of the class for which to retrieve false negatives.
     * @return The number of false negatives for the specified class.
     */
    int FN(int clsIndex) const;

    /**
     *
     * @brief Calculates the F1 score for a specific class.
     *
     * The F1 score is the harmonic mean of precision and recall. It provides a balanced
     * measure of a model's performance, taking into account both false positives and false negatives.
     *
     * @param clsIndex The index of the class for which to calculate the F1 score.
     * @return The F1 score for the specified class.
     */
    float F1(int clsIndex) const;

    /**
     *
     * @brief Calculates the average recall across all classes.
     *
     * The average recall is the mean of recall values for each class, providing
     * an overall measure of model recall considering all classes.
     *
     * @param classes The number of classes to consider. If -1, all classes are considered.
     *
     * @return The average recall value across all classes.
     */
    float avgRecall(int classes = -1) const;

    /**
     * @brief Calculates the mean Average Precision (mAP) across all classes.
     *
     * The mean Average Precision is the mean of precision values for each class, providing
     * an overall measure of model precision considering all classes.
     *
     * @param classes The number of classes to consider. If -1, all classes are considered.
     *
     * @return The mAP value across all classes.
     */
    float mAP(int classes = -1) const;

    /**
     * @brief Calculates the micro-averaged F1 score across all classes.
     *
     * The micro-averaged F1 score considers the overall performance of the model
     * by aggregating true positives, false positives, and false negatives across all classes.
     * It provides a single F1 score that reflects the model's overall ability to balance precision and recall.
     *
     * @return The micro-averaged F1 score across all classes.
     */
    float microAvgF1() const;

    /**
     * @brief Calculates the precision for a specific class.
     *
     * Precision is the ratio of true positives to the total predicted positives,
     * providing an indication of how many of the predicted positive instances are relevant.
     *
     * @param clsIndex The index of the class for which to calculate precision.
     * @return The precision value for the specified class.
     */
    float precision(int clsIndex) const;

    /**
     * @brief Calculates the recall for a specific class.
     *
     * Recall is the ratio of true positives to the total actual positives,
     * providing an indication of how many of the actual positive instances are captured by the model.
     *
     * @param clsIndex The index of the class for which to calculate recall.
     * @return The recall value for the specified class.
     */
    float recall(int clsIndex) const;

    /**
     * @brief Get the number of undetected objects for a specific class.
     *
     * Undetected objects are instances where there is a ground truth for the specified class,
     * but no corresponding predictions exist.
     *
     * @param clsIndex The index of the class for which to retrieve undetected objects.
     * @return The number of undetected objects for the specified class.
    */
    int undetected(int clsIndex) const;

    /**
     * @brief Get the number of ghost predictions for a specific class.
     *
     * Ghost predictions are instances where there are predictions for the specified class,
     * but no corresponding ground truths exist.
     *
     * @param clsIndex The index of the class for which to retrieve ghost predictions.
     * @return The number of ghost predictions for the specified class.
    */
    int ghosts(int clsIndex) const;

    int classes() const noexcept;

private:
    int numClasses_;
    std::vector<std::vector<int>> matrix_;
};

}   // px
