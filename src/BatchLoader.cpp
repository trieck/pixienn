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

#include <boost/filesystem.hpp>

#include "BatchLoader.h"
#include "Error.h"
#include "FileUtil.h"
#include "Image.h"
#include "ImageAugmenter.h"

namespace px {

BatchLoader::BatchLoader(std::string imagesPath, std::string labelsPath, std::uint32_t batchSize,
                         std::uint32_t channels, std::uint32_t height, std::uint32_t width,
                         const ImageAugmenter::Ptr& augmenter, std::uint32_t queueSize)
        : imagesPath_(std::move(imagesPath)), labelsPath_(std::move(labelsPath)), batchSize_(batchSize),
          channels_(channels), height_(height), width_(width), stop_(false), augmenter_(augmenter),
          queueSize_(queueSize)
{
    loadPaths();

    worker_ = std::thread(&BatchLoader::loadBatches, this);
}

void BatchLoader::loadBatches()
{
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, imageFiles_.size() - 1);

    while (true) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return stop_ || batches_.size() < queueSize_; });
        if (stop_) {
            break;
        }

        if (batches_.size() >= queueSize_) {
            continue;
        }

        MiniBatch batch(batchSize_, channels_, height_, width_);
        for (auto i = 0; i < batchSize_; ++i) {
            auto j = distribution(generator);
            const auto& path = imageFiles_[j];
            auto imgLabels = loadImgLabels(path);

            batch.setImageData(i, imgLabels.first);  // the image data must be copied
            batch.setGroundTruth(i, std::move(imgLabels.second));
        }

        batches_.push(std::move(batch));
        lock.unlock();
        cv_.notify_all();
    }
}

MiniBatch BatchLoader::next()
{
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return stop_ || !batches_.empty(); });

    PX_CHECK(!batches_.empty(), "No more batches to load");

    auto batch = std::move(batches_.front());
    batches_.pop();

    lock.unlock();
    cv_.notify_all();

    return batch;
}

void BatchLoader::stop()
{
    std::unique_lock<std::mutex> lock(mutex_);
    stop_ = true;
    lock.unlock();

    cv_.notify_all();
    worker_.join();
}

BatchLoader::~BatchLoader()
{
    stop();
}

void BatchLoader::loadPaths()
{
    std::ifstream ifs(imagesPath_, std::ios::in | std::ios::binary);
    PX_CHECK(ifs.is_open(), "Could not open file \"%s\"", imagesPath_.c_str());

    imageFiles_.clear();

    for (std::string line; std::getline(ifs, line);) {
        imageFiles_.push_back(line);
    }
}

auto BatchLoader::loadImgLabels(const std::string& imagePath) -> ImageLabels
{
    auto gts = groundTruth(imagePath);

    if (augmenter_) {
        auto orig = imreadNormalize(imagePath.c_str(), channels_);
        auto augmented = augmenter_->augment(orig, { (int) width_, (int) height_ }, gts);
        auto vector = imvector(augmented.first);

        return { vector, augmented.second };
    } else {
        auto vec = imreadVector(imagePath.c_str(), width_, height_, channels_);

        // Shift the ground truth boxes to the new image size
        GroundTruthVec newGts;

        for (const auto& gt: gts) {
            GroundTruth newGt(gt);
            newGt.box.x() = (gt.box.x() * vec.ax) + vec.dx;
            newGt.box.y() = (gt.box.y() * vec.ay) + vec.dy;
            newGt.box.w() = gt.box.w() * vec.ax;
            newGt.box.h() = gt.box.h() * vec.ay;

            newGts.emplace_back(std::move(newGt));
        }

        return { vec.data, newGts };
    }
}

GroundTruthVec BatchLoader::groundTruth(const std::string& imagePath)
{
    auto basePath = baseName(imagePath);

    boost::filesystem::path gtFile(labelsPath_);
    gtFile /= basePath += ".txt";
    gtFile = canonical(gtFile);

    std::ifstream ifs(gtFile);
    PX_CHECK(ifs.good(), "Could not open file \"%s\".", gtFile.c_str());

    GroundTruthVec vector;

    std::size_t id;
    float x, y, w, h;
    while (ifs >> id >> x >> y >> w >> h) {
        GroundTruth gt;
        gt.classId = id;

        gt.box.x() = constrain(0.0f, 1.0f, x);
        gt.box.y() = constrain(0.0f, 1.0f, y);
        gt.box.w() = constrain(0.0f, 1.0f, w);
        gt.box.h() = constrain(0.0f, 1.0f, h);

        vector.emplace_back(std::move(gt));
    }

    return vector;
}

std::size_t BatchLoader::size() const
{
    return imageFiles_.size();
}

} // px
