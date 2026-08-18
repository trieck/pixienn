#pragma once

#include <deque>

#include "Error.h"
#include "LRPolicy.h"

namespace px {

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
