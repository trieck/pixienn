#pragma once

#include <deque>

#include "Error.h"
#include "LRPolicy.h"

namespace px {

/**
 * Validation-driven learning-rate policy that reduces the current rate when
 * the monitored mAP50 metric stops improving.
 *
 * Each validation appends mAP50 to a bounded deque containing at most
 * `smoothing` recent values. The policy compares their moving average with
 * the best smoothed metric seen so far. Improvements larger than `threshold`
 * reset the bad-validation counter; otherwise the counter advances after any
 * cooldown period. Once `patience` consecutive non-improving validations have
 * accumulated, the learning rate is multiplied by `factor`, never dropping
 * below `minLR`, and the cooldown timer is restarted.
 *
 * The policy state, including the recent validation window and plateau
 * counters, is serialized so a resumed training run preserves its schedule.
 */
class ReduceOnPlateauLRPolicy : public LRPolicy
{
public:
    ReduceOnPlateauLRPolicy(float initialLR, float factor, int patience, float threshold,
                            int cooldown, float minLR, int smoothing);

    float update(int batchNum) override;
    float LR() const noexcept override;
    void onValidation(const ValidationMetrics& metrics, int batchNum) override;
    void saveState(std::ostream& state) const override;
    void loadState(std::istream& state) override;
    void reset() override;

private:
    float initialLR_;
    float currentLR_;
    float factor_;
    int patience_;
    float threshold_;
    int cooldown_;
    float minLR_;
    int smoothing_;
    float bestMetric_ = 0.0f;
    int badValidations_ = 0;
    int cooldownRemaining_ = 0;
    std::deque<float> history_;
};

} // namespace px
