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
#include <boost/algorithm/string/trim.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/types_c.h>
#include "BatchLoader.h"
#include "ColorMaps.h"
#include "Error.h"
#include "FileUtil.h"
#include "Image.h"
#include "ImageAugmenter.h"

namespace px {

BatchLoader::BatchLoader(std::string imagesPath, std::string labelsPath, std::uint32_t batchSize,
        std::uint32_t channels, std::uint32_t height, std::uint32_t width,
        std::vector<std::string> labels, const ImageAugmenter::Ptr& augmenter,
        bool viewImage, std::uint32_t queueSize, bool randomize)
        : augmenter_(augmenter), labels_(std::move(labels)), imagesPath_(std::move(imagesPath)),
          labelsPath_(std::move(labelsPath)), batchSize_(batchSize), channels_(channels),
          height_(height), width_(width), queueSize_(queueSize), stop_(false), viewImage_(viewImage),
          randomize_(randomize),
          generator_(std::random_device{}())
{
    PX_CHECK(queueSize_ > 0, "Batch loader queue size must be positive.");
    loadPaths();

    imageOrder_.resize(imageFiles_.size());
    std::iota(imageOrder_.begin(), imageOrder_.end(), 0);
    if (randomize_) {
        std::shuffle(imageOrder_.begin(), imageOrder_.end(), generator_);
    }

    const auto hardwareThreads = std::max(1u, std::thread::hardware_concurrency());
    const auto workerCount = std::max(1u, std::min({ 4u, hardwareThreads, queueSize_ }));
    workers_.reserve(workerCount);
    for (auto i = 0u; i < workerCount; ++i) {
        workers_.emplace_back(&BatchLoader::loadBatches, this);
    }
}

void BatchLoader::loadBatches()
{
    auto reservedBatch = false;
    try {
        while (true) {
            std::vector<std::string> paths(batchSize_);
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] {
                    return stop_ || workerError_ || batches_.size() + batchesInFlight_ < queueSize_;
                });
                if (stop_ || workerError_) {
                    break;
                }

                ++batchesInFlight_;
                reservedBatch = true;
                for (auto& path: paths) {
                    if (nextImage_ == imageOrder_.size()) {
                        nextImage_ = 0;
                        if (randomize_) {
                            std::shuffle(imageOrder_.begin(), imageOrder_.end(), generator_);
                        }
                    }
                    path = imageFiles_[imageOrder_[nextImage_++]];
                }
            }

            // Image decoding, augmentation, and packing are intentionally done
            // without the queue mutex. Consumers can pop prefetched batches and
            // other workers can prepare subsequent batches concurrently.
            MiniBatch batch(batchSize_, channels_, height_, width_);
            for (std::uint32_t i = 0; i < batchSize_; ++i) {
                auto imgLabels = loadImgLabels(paths[i]);

                batch.setImageData(i, imgLabels.first);  // the image data must be copied
                batch.setGroundTruth(i, std::move(imgLabels.second));
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                --batchesInFlight_;
                reservedBatch = false;
                if (stop_) {
                    break;
                }
                batches_.push(std::move(batch));
            }
            cv_.notify_all();
        }
    } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (reservedBatch) {
            --batchesInFlight_;
        }
        workerError_ = std::current_exception();
        stop_ = true;
        cv_.notify_all();
    }
}

MiniBatch BatchLoader::next()
{
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return stop_ || workerError_ || !batches_.empty(); });

    if (workerError_) {
        std::rethrow_exception(workerError_);
    }

    PX_CHECK(!batches_.empty(), "No more batches to load");

    auto batch = std::move(batches_.front());
    batches_.pop();

    lock.unlock();
    cv_.notify_all();

    return batch;
}

void BatchLoader::stop()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }

    cv_.notify_all();
    for (auto& worker: workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
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
    const auto basePath = boost::filesystem::path(imagesPath_).parent_path();

    for (std::string line; std::getline(ifs, line);) {
        boost::algorithm::trim(line);
        if (line.empty()) {
            continue;
        }

        boost::filesystem::path imagePath(line);
        if (imagePath.is_relative()) {
            imagePath = basePath / imagePath;
        }

        PX_CHECK(boost::filesystem::is_regular_file(imagePath),
                 "Could not open image file \"%s\"", imagePath.c_str());
        imageFiles_.push_back(boost::filesystem::canonical(imagePath).string());
    }

    PX_CHECK(!imageFiles_.empty(), "Image list \"%s\" is empty", imagesPath_.c_str());
}

auto BatchLoader::loadImgLabels(const std::string& imagePath) -> ImageLabels
{
    auto gts = groundTruth(imagePath);

    if (viewImage_) {
        viewImageGT(imagePath, gts);
    }

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
        PX_CHECK(id <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                 "Class ID is too large: %zu.", id);
        gt.classId = static_cast<int>(id);

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

void BatchLoader::viewImageGT(const std::string& imagePath, const GroundTruthVec& gt) const
{
    ColorMaps colors("plasma");

    cv::Mat image;

    if (augmenter_) {
        auto orig = imread(imagePath.c_str(), channels_);
        auto augmented = augmenter_->augment(orig, { static_cast<int>(width_), static_cast<int>(height_) }, gt);

        image = augmented.first;

        cv::cvtColor(image, image, cv::COLOR_BGR2BGRA);

        for (const auto& g: augmented.second) {
            auto index = g.classId;
            const auto& label = labels_[index];

            auto bgColor = colors.color(index);
            auto textColor = imtextcolor(bgColor);

            auto lb = lightBox(g.box, { static_cast<int>(width_), static_cast<int>(height_) });

            imrect(image, lb, bgColor, 2);
            imtabbedText(image, label.c_str(), lb.tl(), textColor, bgColor, 2);
        }
    } else {
        auto mat = imread(imagePath.c_str(), width_, height_, channels_);
        image = mat.image;

        cv::cvtColor(image, image, cv::COLOR_BGR2BGRA);

        for (const auto& g: gt) {
            auto index = g.classId;
            const auto& label = labels_[index];

            auto bgColor = colors.color(index);
            auto textColor = imtextcolor(bgColor);

            auto x = (g.box.x() * mat.ax) + mat.dx;
            auto y = (g.box.y() * mat.ay) + mat.dy;
            auto w = g.box.w() * mat.ax;
            auto h = g.box.h() * mat.ay;

            auto lb = lightBox({ x, y, w, h }, { static_cast<int>(width_), static_cast<int>(height_) });

            imrect(image, lb, bgColor, 2);
            imtabbedText(image, label.c_str(), lb.tl(), textColor, bgColor, 2);
        }
    }

    cv::imshow("image", image);
    cv::waitKey();
}

} // px
