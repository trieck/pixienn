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

#ifndef PIXIENN_IMAGEBATCH_H
#define PIXIENN_IMAGEBATCH_H

#include <opencv2/core/types.hpp>
#include "PxTensor.h"

namespace px {

struct GroundTruth
{
    std::size_t classId;
    cv::Rect2f box;
};

using GroundTruthVec = std::vector<GroundTruth>;
using GroundTruths = std::vector<GroundTruthVec>;

class ImageBatch
{
public:
    ImageBatch();
    ImageBatch(std::uint32_t batchSize, std::uint32_t channels, std::uint32_t height, std::uint32_t width);
    ImageBatch(const ImageBatch& rhs);
    ImageBatch(ImageBatch&& rhs);
    ~ImageBatch() = default;

    ImageBatch& operator=(const ImageBatch& rhs);
    ImageBatch& operator=(ImageBatch&& rhs);

    std::uint32_t batchSize() const noexcept;
    std::uint32_t channels() const noexcept;
    std::uint32_t height() const noexcept;
    std::uint32_t width() const noexcept;
    const PxCpuVector& imageData() const;
    std::size_t imageDataSize() const noexcept;
    PxCpuVector::const_pointer slice(uint32_t batch) const;
    GroundTruths groundTruth() const noexcept;
    GroundTruthVec groundTruth(uint32_t batch) const;

    void allocate(std::uint32_t batchSize, std::uint32_t channels, std::uint32_t height, std::uint32_t width);
    void setImageData(std::uint32_t batch, const PxCpuVector& imageData);
    void setGroundTruth(std::uint32_t batch, GroundTruthVec&& groundTruthVec);
    void addGroundTruth(std::uint32_t batch, GroundTruth&& groundTruth);

private:
    GroundTruths groundTruth_;
    PxCpuVector imageData_;

    std::uint32_t batchSize_;
    std::uint32_t channels_;
    std::uint32_t height_;
    std::uint32_t width_;
};


}   // px

#endif // PIXIENN_IMAGEBATCH_H
