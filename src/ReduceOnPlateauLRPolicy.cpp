#include "ReduceOnPlateauLRPolicy.h"

#include <cstdint>

namespace px {

ReduceOnPlateauLRPolicy::ReduceOnPlateauLRPolicy(float initialLR, float factor, int patience,
                                                 float threshold, int cooldown, float minLR, int smoothing)
        : initialLR_(initialLR), currentLR_(initialLR), factor_(factor), patience_(patience),
          threshold_(threshold), cooldown_(cooldown), minLR_(minLR), smoothing_(smoothing)
{
    PX_CHECK(initialLR_ > 0.0f, "ReduceOnPlateauLRPolicy: initial learning rate must be positive");
    PX_CHECK(factor_ > 0.0f && factor_ < 1.0f, "ReduceOnPlateauLRPolicy: factor must be between zero and one");
    PX_CHECK(patience_ > 0, "ReduceOnPlateauLRPolicy: patience must be positive");
    PX_CHECK(threshold_ >= 0.0f, "ReduceOnPlateauLRPolicy: threshold must not be negative");
    PX_CHECK(cooldown_ >= 0, "ReduceOnPlateauLRPolicy: cooldown must not be negative");
    PX_CHECK(minLR_ > 0.0f && minLR_ <= initialLR_,
             "ReduceOnPlateauLRPolicy: min learning rate must be positive and no greater than initial rate");
    PX_CHECK(smoothing_ > 0, "ReduceOnPlateauLRPolicy: smoothing must be positive");
}

float ReduceOnPlateauLRPolicy::update(int /*batchNum*/)
{
    return currentLR_;
}

float ReduceOnPlateauLRPolicy::LR() const noexcept
{
    return currentLR_;
}

void ReduceOnPlateauLRPolicy::onValidation(const ValidationMetrics& metrics, int /*batchNum*/)
{
    history_.push_back(metrics.mAP50);
    while (static_cast<int>(history_.size()) > smoothing_) {
        history_.pop_front();
    }

    auto smoothedMetric = 0.0f;
    for (const auto value: history_) {
        smoothedMetric += value;
    }
    smoothedMetric /= static_cast<float>(history_.size());

    if (history_.size() == 1 || smoothedMetric > bestMetric_ + threshold_) {
        bestMetric_ = smoothedMetric;
        badValidations_ = 0;
        return;
    }

    if (cooldownRemaining_ > 0) {
        --cooldownRemaining_;
        return;
    }

    ++badValidations_;
    if (badValidations_ < patience_) {
        return;
    }

    currentLR_ = std::max(minLR_, currentLR_ * factor_);
    badValidations_ = 0;
    cooldownRemaining_ = cooldown_;
}

void ReduceOnPlateauLRPolicy::reset()
{
    currentLR_ = initialLR_;
    bestMetric_ = 0.0f;
    badValidations_ = 0;
    cooldownRemaining_ = 0;
    history_.clear();
}

void ReduceOnPlateauLRPolicy::saveState(std::ostream& state) const
{
    constexpr std::uint32_t magic = 0x52504c31; // RPL1
    const auto historySize = static_cast<std::uint32_t>(history_.size());
    state.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    state.write(reinterpret_cast<const char*>(&currentLR_), sizeof(currentLR_));
    state.write(reinterpret_cast<const char*>(&bestMetric_), sizeof(bestMetric_));
    state.write(reinterpret_cast<const char*>(&badValidations_), sizeof(badValidations_));
    state.write(reinterpret_cast<const char*>(&cooldownRemaining_), sizeof(cooldownRemaining_));
    state.write(reinterpret_cast<const char*>(&historySize), sizeof(historySize));
    for (const auto value: history_) {
        state.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
}

void ReduceOnPlateauLRPolicy::loadState(std::istream& state)
{
    constexpr std::uint32_t expectedMagic = 0x52504c31; // RPL1
    std::uint32_t magic = 0;
    std::uint32_t historySize = 0;
    state.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    PX_CHECK(magic == expectedMagic, "Invalid ReduceOnPlateauLRPolicy state");
    state.read(reinterpret_cast<char*>(&currentLR_), sizeof(currentLR_));
    state.read(reinterpret_cast<char*>(&bestMetric_), sizeof(bestMetric_));
    state.read(reinterpret_cast<char*>(&badValidations_), sizeof(badValidations_));
    state.read(reinterpret_cast<char*>(&cooldownRemaining_), sizeof(cooldownRemaining_));
    state.read(reinterpret_cast<char*>(&historySize), sizeof(historySize));
    PX_CHECK(historySize <= static_cast<std::uint32_t>(smoothing_),
             "Invalid ReduceOnPlateauLRPolicy history size");
    history_.clear();
    for (std::uint32_t index = 0; index < historySize; ++index) {
        float value = 0.0f;
        state.read(reinterpret_cast<char*>(&value), sizeof(value));
        history_.push_back(value);
    }
    PX_CHECK(state.good(), "Could not read ReduceOnPlateauLRPolicy state");
}

} // namespace px
