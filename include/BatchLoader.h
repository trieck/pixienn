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

#pragma once

#include "Common.h"
#include "GroundTruth.h"
#include "ImageAugmenter.h"
#include "MiniBatch.h"

namespace px {

class BatchLoader
{
public:
    BatchLoader(std::string imagesPath, std::string labelsPath, std::uint32_t batchSize, std::uint32_t channels,
                std::uint32_t height, std::uint32_t width, const ImageAugmenter::Ptr& augmenter = nullptr,
                std::uint32_t queueSize = 10);
    ~BatchLoader();

    using Ptr = std::unique_ptr<BatchLoader>;

    MiniBatch next();
    void stop();
    std::size_t size() const;

private:
    using ImageLabels = std::pair<PxCpuVector, GroundTruthVec>;

    void loadPaths();
    void loadBatches();
    ImageLabels loadImgLabels(const std::string& imagePath);
    GroundTruthVec groundTruth(const std::string& imagePath);

    ImageAugmenter::Ptr augmenter_;
    std::thread worker_;

    std::vector<std::string> imageFiles_;
    std::queue<MiniBatch> batches_;
    std::mutex mutex_;
    std::condition_variable cv_;

    std::string imagesPath_, labelsPath_;
    std::uint32_t batchSize_, channels_, height_, width_, queueSize_;
    bool stop_;
};

} // px
