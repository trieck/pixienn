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

#ifndef PIXIENN_INVLRPOLICY_H__
#define PIXIENN_INVLRPOLICY_H__

#include "Common.h"
#include "LRPolicy.h"

namespace px {

/**
 * @brief A class representing a learning rate policy based on inverse power-law decay.
 *
 * This class provides functionality to update the learning rate using an inverse power-law decay
 * policy during training iterations. It also allows resetting the learning rate to its original value.
 */
class InvLRPolicy : public LRPolicy
{
public:
    /**
     * @brief Default constructor.
     */
    InvLRPolicy();

    /**
     * @brief Parameterized constructor to initialize the learning rate policy.
     * @param lr The initial learning rate.
     * @param gamma The decay factor for the learning rate.
     * @param power The power to which the iteration number is raised in the decay formula.
     */
    InvLRPolicy(float lr, float gamma, float power);

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
     * @brief Update the learning rate based on inverse power-law decay.
     * @param batchNum The current batch number or iteration count.
     * @return The updated learning rate.
     */
    float update(int batchNum) override;

    /**
     * @brief Reset the learning rate to its original value.
     */
    void reset() override;

private:
    float origLr_ = 0;   /**< The original (initial) learning rate. */
    float lr_ = 0;       /**< The current learning rate. */
    float gamma_ = 0;    /**< The decay factor for the learning rate. */
    float power_ = 0;    /**< The power to which the iteration number is raised in the decay formula. */
};

}   // px

#endif // PIXIENN_INVLRPOLICY_H__