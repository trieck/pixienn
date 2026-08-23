#pragma once

#include <cstddef>

namespace px {

void layerNormForwardGpu(int batch, int channels, int spatial, float epsilon,
                         const float* input, const float* scales, const float* biases,
                         float* mean, float* variance, float* normalized, float* output);
void layerNormBackwardGpu(int batch, int channels, int spatial, float epsilon,
                          const float* delta, const float* scales, const float* normalized,
                          const float* variance, float* inputDelta, float* scaleUpdates,
                          float* biasUpdates);

void positionalEncodingGpu(std::size_t size, const float* input, const float* encoding, float* output);
void addGpu(std::size_t size, const float* input, float* output);

void selfAttentionProjectGpu(int batch, int tokens, int channels,
                             const float* input, const float* queryWeights, const float* queryBiases,
                             const float* keyWeights, const float* keyBiases,
                             const float* valueWeights, const float* valueBiases,
                             float* query, float* key, float* value);
void selfAttentionForwardGpu(int batch, int tokens, int channels, int heads, float scale,
                             const float* query, const float* key, const float* value,
                             float* attention, float* context);
void selfAttentionOutputGpu(int batch, int tokens, int channels,
                            const float* context, const float* weights, const float* biases, float* output);
void selfAttentionOutputBackwardGpu(int batch, int tokens, int channels,
                                    const float* delta, const float* context, const float* weights,
                                    float* contextGradient, float* weightUpdates, float* biasUpdates);
void selfAttentionAttentionBackwardGpu(int batch, int tokens, int channels, int heads, float scale,
                                       const float* attention, const float* query, const float* key,
                                       const float* value, const float* contextGradient,
                                       float* queryGradient, float* keyGradient, float* valueGradient);
void selfAttentionProjectBackwardGpu(int batch, int tokens, int channels,
                                     const float* input, const float* queryGradient, const float* keyGradient,
                                     const float* valueGradient, const float* queryWeights, const float* keyWeights,
                                     const float* valueWeights, float* inputGradient,
                                     float* queryUpdates, float* queryBiasUpdates,
                                     float* keyUpdates, float* keyBiasUpdates,
                                     float* valueUpdates, float* valueBiasUpdates);

} // namespace px
