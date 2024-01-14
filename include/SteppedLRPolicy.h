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
#include "LRPolicy.h"

namespace px {

/**
 * @brief A class representing a learning rate policy based on a stepped schedule.
 *
 * This class provides functionality to update the learning rate based on a stepped schedule
 * during training iterations. It also allows resetting the learning rate to its original value.
 */
class SteppedLRPolicy : public LRPolicy
{
public:
    /**
     * @brief Default constructor.
     */
    SteppedLRPolicy();

    /**
     * @brief Parameterized constructor to initialize the learning rate policy.
     * @param lr The initial learning rate.
     * @param steps A vector of batch numbers at which the learning rate will be adjusted.
     * @param scales A vector of scaling factors corresponding to each step.
     */
    SteppedLRPolicy(float lr, std::vector<int> steps, std::vector<float> scales);

    /**
     * @brief Get the current learning rate.
     * @return The current learning rate.
     */
    float LR() const noexcept override;

    /**
     * @brief Get the original (initial) learning rate.
     * @return The original learning rate.
     */
    float origLR() const noexcept;

    /**
     * @brief Get the scaling factor for the current step.
     * @return The scaling factor for the current step.
     */
    float scale() const noexcept;

    /**
     * @brief Update the learning rate based on the stepped schedule.
     * @param batchNum The current batch number or iteration count.
     * @return The updated learning rate.
     */
    float update(int batchNum) override;

    /**
     * @brief Get the current step number.
     * @return The current step number.
     */
    int step() const noexcept;

    /**
     * @brief Set the learning rate, steps, and scales.
     * @param lr The initial learning rate.
     * @param steps A vector of batch numbers at which the learning rate will be adjusted.
     * @param scales A vector of scaling factors corresponding to each step.
     */
    void set(float lr, std::vector<int> steps, std::vector<float> scales);

    /**
     * @brief Reset the learning rate to its original value and the step count to zero.
     */
    void reset() override;

private:
    int step_ = 0;                /**< The current step number. */
    float lr_ = 0;                /**< The current learning rate. */
    float origLr_ = 0;            /**< The original (initial) learning rate. */
    std::vector<int> steps_;      /**< A vector of batch numbers at which the learning rate will be adjusted. */
    std::vector<float> scales_;   /**< A vector of scaling factors corresponding to each step. */
};

}   // px
