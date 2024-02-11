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

#include "Error.h"
#include "TrainBatch.h"

namespace px {

TrainBatch::TrainBatch() : batchSize_(0), channels_(0), height_(0), width_(0)
{
}

TrainBatch::TrainBatch(std::uint32_t batchSize, uint32_t channels, uint32_t height, uint32_t width)
        : batchSize_(batchSize), channels_(channels), height_(height), width_(width),
          groundTruth_(batchSize)
{
    imageData_ = PxCpuVector(batchSize * height * channels * width, 0.0f);
}

TrainBatch::TrainBatch(const TrainBatch& rhs)
{
    *this = rhs;
}

TrainBatch::TrainBatch(TrainBatch&& rhs)
{
    *this = std::move(rhs);
}

TrainBatch& TrainBatch::operator=(const TrainBatch& rhs)
{
    if (this != &rhs) {
        imageData_ = rhs.imageData_;
        groundTruth_ = rhs.groundTruth_;
        batchSize_ = rhs.batchSize_;
        channels_ = rhs.channels_;
        height_ = rhs.height_;
        width_ = rhs.width_;
    }

    return *this;
}

TrainBatch& TrainBatch::operator=(TrainBatch&& rhs)
{
    std::swap(imageData_, rhs.imageData_);
    std::swap(groundTruth_, rhs.groundTruth_);
    std::swap(batchSize_, rhs.batchSize_);
    std::swap(channels_, rhs.channels_);
    std::swap(height_, rhs.height_);
    std::swap(width_, rhs.width_);

    return *this;
}

void TrainBatch::allocate(std::uint32_t batchSize, std::uint32_t channels, std::uint32_t height, std::uint32_t width)
{
    batchSize_ = batchSize;
    channels_ = channels;
    height_ = height;
    width_ = width;

    imageData_ = PxCpuVector(batchSize * height * channels * width, 0.0f);

    groundTruth_.clear();
    groundTruth_.reserve(batchSize);
}

std::uint32_t TrainBatch::batchSize() const noexcept
{
    return batchSize_;
}

std::uint32_t TrainBatch::channels() const noexcept
{
    return channels_;
}

std::uint32_t TrainBatch::height() const noexcept
{
    return height_;
}
std::uint32_t TrainBatch::width() const noexcept
{
    return width_;
}

const PxCpuVector& TrainBatch::imageData() const
{
    return imageData_;
}

PxCpuVector::const_pointer TrainBatch::slice(uint32_t batch) const
{
    PX_CHECK(batch < batchSize_, "Index out of range.");

    auto index = batch * channels_ * height_ * width_;

    return imageData_.data() + index;
}

const GroundTruths& TrainBatch::groundTruth() const noexcept
{
    return groundTruth_;
}

const GroundTruthVec& TrainBatch::groundTruth(uint32_t batch) const
{
    PX_CHECK(batch < batchSize_, "Index out of range.");

    return groundTruth_[batch];
}

void TrainBatch::setImageData(std::uint32_t batch, const PxCpuVector& imageData)
{
    PX_CHECK(batch < batchSize_, "Index out of range.");

    auto index = batch * channels_ * height_ * width_;

    std::copy(imageData.begin(), imageData.end(), imageData_.begin() + index);
}

void TrainBatch::setGroundTruth(std::uint32_t batch, GroundTruthVec&& groundTruthVec)
{
    PX_CHECK(batch < batchSize_, "Index out of range.");
    groundTruth_[batch] = std::move(groundTruthVec);
}

void TrainBatch::addGroundTruth(std::uint32_t batch, GroundTruth&& groundTruth)
{
    PX_CHECK(batch < batchSize_, "Index out of range.");
    groundTruth_[batch].emplace_back(std::move(groundTruth));
}

std::size_t TrainBatch::imageDataSize() const noexcept
{
    return imageData_.size();
}

void TrainBatch::release()
{
    groundTruth_.clear();
    imageData_.release();
}

}   // px
