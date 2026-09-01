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
#include "Utility.h"

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
            auto validSize = batchSize_;
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
                for (std::uint32_t i = 0; i < batchSize_; ++i) {
                    if (nextImage_ == imageOrder_.size()) {
                        if (!randomize_ && i > 0) {
                            validSize = i;
                            break;
                        }
                        nextImage_ = 0;
                        if (randomize_) {
                            std::shuffle(imageOrder_.begin(), imageOrder_.end(), generator_);
                        }
                    }
                    paths[i] = imageFiles_[imageOrder_[nextImage_++]];
                }
            }

            // Image decoding, augmentation, and packing are intentionally done
            // without the queue mutex. Consumers can pop prefetched batches and
            // other workers can prepare subsequent batches concurrently.
            MiniBatch batch(batchSize_, channels_, height_, width_);
            for (std::uint32_t i = 0; i < validSize; ++i) {
                ImageLabels imgLabels;
                if (augmenter_ && augmenter_->useMosaic()) {
                    std::array<ImageLabel, 4> sources;
                    for (auto tile = 0; tile < 4; ++tile) {
                        auto sourcePath = tile == 0 ? paths[i]
                                                    : imageFiles_[randomUniform<std::size_t>(0, imageFiles_.size() - 1)];
                        auto source = loadRawImgLabels(sourcePath);
                        sources[tile] = std::make_pair(std::move(source.first), std::move(source.second));
                    }
                    auto mosaic = augmenter_->augmentMosaic(
                            sources, { static_cast<int>(width_), static_cast<int>(height_) });
                    imgLabels = std::move(mosaic);
                } else {
                    imgLabels = loadImgLabels(paths[i]);
                }

                batch.setImageData(i, imvector(imgLabels.first));  // the image data must be copied
                batch.setGroundTruth(i, std::move(imgLabels.second));
            }
            batch.setValidSize(validSize);

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
    auto raw = loadRawImgLabels(imagePath);
    auto& gts = raw.second;

    if (augmenter_) {
        auto augmented = augmenter_->augment(raw.first, { (int) width_, (int) height_ }, gts);
        return augmented;
    } else {
        auto mat = imletterbox(raw.first, width_, height_);

        // Shift the ground truth boxes to the new image size
        GroundTruthVec newGts;

        for (const auto& gt: gts) {
            GroundTruth newGt(gt);
            newGt.box.x() = (gt.box.x() * mat.ax) + mat.dx;
            newGt.box.y() = (gt.box.y() * mat.ay) + mat.dy;
            newGt.box.w() = gt.box.w() * mat.ax;
            newGt.box.h() = gt.box.h() * mat.ay;

            newGts.emplace_back(std::move(newGt));
        }

        return { std::move(mat.image), std::move(newGts) };
    }
}

auto BatchLoader::loadRawImgLabels(const std::string& imagePath) -> RawImageLabels
{
    return { imreadNormalize(imagePath.c_str(), channels_), groundTruth(imagePath) };
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

void BatchLoader::viewBatch(const MiniBatch& batch) const
{
    const auto planeSize = static_cast<std::size_t>(batch.width()) * batch.height();
    const auto type = CV_MAKETYPE(CV_32F, batch.channels());
    for (auto b = 0u; b < batch.validSize(); ++b) {
        cv::Mat image(static_cast<int>(batch.height()), static_cast<int>(batch.width()), type);
        const auto* planes = batch.slice(b);
        for (auto y = 0u; y < batch.height(); ++y) {
            auto* pixels = image.ptr<float>(static_cast<int>(y));
            for (auto x = 0u; x < batch.width(); ++x) {
                for (auto channel = 0u; channel < batch.channels(); ++channel) {
                    pixels[x * batch.channels() + channel] =
                            planes[channel * planeSize + y * batch.width() + x];
                }
            }
        }
        viewImageGT(image, batch.groundTruth(b));
    }
}

void BatchLoader::viewImageGT(const cv::Mat& source, const GroundTruthVec& gt) const
{
    auto image = source.depth() == CV_8U ? source.clone() : imdenormalize(source);
    if (image.channels() == 3) {
        cv::cvtColor(image, image, cv::COLOR_BGR2BGRA);
    } else if (image.channels() == 1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGRA);
    }

    ColorMaps colors("plasma");
    for (const auto& g: gt) {
        const auto index = g.classId;
        const auto& label = labels_[index];
        const auto bgColor = colors.color(index);
        const auto textColor = imtextcolor(bgColor);
        const auto box = lightBox(g.box, { static_cast<int>(width_), static_cast<int>(height_) });
        imrect(image, box, bgColor, 2);
        imtabbedText(image, label.c_str(), box.tl(), textColor, bgColor, 2);
    }

    cv::imshow("image", image);
    cv::waitKey();
}

} // px
