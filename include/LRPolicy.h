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

#ifndef PIXIENN_LRPOLICY_H__
#define PIXIENN_LRPOLICY_H__

#include "Common.h"

namespace px {

/**
 * @brief Abstract base class for learning rate policies.
 */
class LRPolicy
{
public:
    using Ptr = std::unique_ptr<LRPolicy>;

    /**
     * @brief Virtual destructor to ensure proper cleanup in derived classes.
     */
    virtual ~LRPolicy() = default;

    /**
     * @brief Get the current learning rate.
     * @return The current learning rate.
     */
    virtual float LR() const noexcept = 0;

    /**
     * @brief Update the learning rate based on the policy.
     * @param batchNum The current batch number or iteration count.
     * @return The updated learning rate.
     */
    virtual float update(int batchNum) = 0;

    /**
     * @brief Reset the learning rate to its original value.
     */
    virtual void reset() = 0;
};

}   // px

#endif // PIXIENN_LRPOLICY_H__
