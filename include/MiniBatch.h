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

#include "GroundTruth.h"
#include "PxTensor.h"

namespace px {

class MiniBatch
{
public:
    MiniBatch();
    MiniBatch(std::uint32_t batchSize, std::uint32_t channels, std::uint32_t height, std::uint32_t width);
    MiniBatch(const MiniBatch& rhs);
    MiniBatch(MiniBatch&& rhs);
    ~MiniBatch() = default;

    MiniBatch& operator=(const MiniBatch& rhs);
    MiniBatch& operator=(MiniBatch&& rhs);

    std::uint32_t batchSize() const noexcept;
    std::uint32_t channels() const noexcept;
    std::uint32_t height() const noexcept;
    std::uint32_t width() const noexcept;
    const PxCpuVector& imageData() const;
    std::size_t imageDataSize() const noexcept;
    PxCpuVector::const_pointer slice(uint32_t batch) const;
    const GroundTruths& groundTruth() const noexcept;
    const GroundTruthVec& groundTruth(uint32_t batch) const;

    void allocate(std::uint32_t batchSize, std::uint32_t channels, std::uint32_t height, std::uint32_t width);
    void setImageData(std::uint32_t batch, const PxCpuVector& imageData);
    void setGroundTruth(std::uint32_t batch, GroundTruthVec&& groundTruthVec);
    void addGroundTruth(std::uint32_t batch, GroundTruth&& groundTruth);

    void release();
private:
    GroundTruths groundTruth_;
    PxCpuVector imageData_;

    std::uint32_t batchSize_;
    std::uint32_t channels_;
    std::uint32_t height_;
    std::uint32_t width_;
};

}   // px

