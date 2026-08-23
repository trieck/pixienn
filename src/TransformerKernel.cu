#include <cuda_runtime.h>
#include <cmath>

#include "CudaError.h"
#include "CudaUtils.cuh"
#include "TransformerKernel.cuh"

namespace px {

__global__ void layerNormForwardKernel(int n, int channels, int spatial, float epsilon,
                                       const float* input, const float* scales, const float* biases,
                                       float* mean, float* variance, float* normalized, float* output)
{
    const auto token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= n) return;
    const auto batch = token / spatial;
    const auto location = token % spatial;
    auto average = 0.0f;
    for (auto c = 0; c < channels; ++c) average += input[location + spatial * (c + channels * batch)];
    average /= channels;
    auto var = 0.0f;
    for (auto c = 0; c < channels; ++c) {
        const auto x = input[location + spatial * (c + channels * batch)] - average;
        var += x * x;
    }
    var /= channels;
    mean[token] = average;
    variance[token] = var;
    const auto inverse = rsqrtf(var + epsilon);
    for (auto c = 0; c < channels; ++c) {
        const auto index = location + spatial * (c + channels * batch);
        const auto x = (input[index] - average) * inverse;
        normalized[index] = x;
        output[index] = scales[c] * x + biases[c];
    }
}

__global__ void layerNormBackwardKernel(int n, int channels, int spatial, float epsilon,
                                        const float* delta, const float* scales, const float* normalized,
                                        const float* variance, float* inputDelta, float* scaleUpdates,
                                        float* biasUpdates)
{
    const auto token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= n) return;
    const auto batch = token / spatial;
    const auto location = token % spatial;
    auto sum = 0.0f;
    auto sumNormalized = 0.0f;
    for (auto c = 0; c < channels; ++c) {
        const auto index = location + spatial * (c + channels * batch);
        const auto scaled = delta[index] * scales[c];
        sum += scaled;
        sumNormalized += scaled * normalized[index];
        atomicAdd(scaleUpdates + c, delta[index] * normalized[index]);
        atomicAdd(biasUpdates + c, delta[index]);
    }
    const auto inverse = rsqrtf(variance[token] + epsilon);
    for (auto c = 0; c < channels; ++c) {
        const auto index = location + spatial * (c + channels * batch);
        const auto scaled = delta[index] * scales[c];
        inputDelta[index] = inverse / channels * (channels * scaled - sum - normalized[index] * sumNormalized);
    }
}

__global__ void addKernel(std::size_t n, const float* input, const float* encoding, float* output)
{
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = input[i] + encoding[i];
}

__global__ void copyAddKernel(std::size_t n, const float* input, float* output)
{
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] += input[i];
}

__global__ void projectKernel(int n, int tokens, int channels, const float* input,
                              const float* qw, const float* qb, const float* kw, const float* kb,
                              const float* vw, const float* vb, float* q, float* k, float* v)
{
    const auto id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const auto c = id % channels;
    const auto token = (id / channels) % tokens;
    const auto batch = id / (tokens * channels);
    const auto offset = batch * tokens * channels + token * channels;
    auto qv = qb[c], kv = kb[c], vv = vb[c];
    for (auto i = 0; i < channels; ++i) {
        const auto x = input[batch * channels * tokens + token + tokens * i];
        qv += qw[c * channels + i] * x;
        kv += kw[c * channels + i] * x;
        vv += vw[c * channels + i] * x;
    }
    q[offset + c] = qv; k[offset + c] = kv; v[offset + c] = vv;
}

__global__ void attentionForwardKernel(int n, int tokens, int channels, int heads, float scale,
                                       const float* q, const float* k, const float* v,
                                       float* attention, float* context)
{
    const auto id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const auto c = id % channels;
    const auto query = (id / channels) % tokens;
    const auto head = c / (channels / heads);
    const auto local = c % (channels / heads);
    const auto batch = id / (tokens * channels);
    const auto channelOffset = head * (channels / heads);
    auto maximum = -1.0e30f;
    for (auto j = 0; j < tokens; ++j) {
        auto score = 0.0f;
        for (auto x = 0; x < channels / heads; ++x)
            score += q[batch*tokens*channels + query*channels + channelOffset+x]
                    * k[batch*tokens*channels + j*channels + channelOffset+x];
        score *= scale;
        attention[batch*heads*tokens*tokens + head*tokens*tokens + query*tokens + j] = score;
        maximum = fmaxf(maximum, score);
    }
    auto total = 0.0f;
    for (auto j = 0; j < tokens; ++j) {
        auto& weight = attention[batch*heads*tokens*tokens + head*tokens*tokens + query*tokens + j];
        weight = expf(weight - maximum); total += weight;
    }
    auto result = 0.0f;
    for (auto j = 0; j < tokens; ++j) {
        auto& weight = attention[batch*heads*tokens*tokens + head*tokens*tokens + query*tokens + j];
        weight /= total;
        result += weight * v[batch*tokens*channels + j*channels + channelOffset + local];
    }
    context[batch*tokens*channels + query*channels + c] = result;
}

__global__ void outputKernel(int n, int tokens, int channels, const float* context,
                             const float* weights, const float* biases, float* output)
{
    const auto id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const auto c = id % channels;
    const auto token = (id / channels) % tokens;
    const auto batch = id / (tokens * channels);
    auto result = biases[c];
    for (auto i = 0; i < channels; ++i)
        result += weights[c*channels+i] * context[batch*tokens*channels+token*channels+i];
    output[batch*channels*tokens + token + tokens*c] = result;
}

__global__ void outputBackwardKernel(int n, int tokens, int channels, const float* delta,
                                     const float* context, const float* weights,
                                     float* contextGradient, float* weightUpdates, float* biasUpdates)
{
    const auto id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const auto o = id % channels;
    const auto token = (id / channels) % tokens;
    const auto batch = id / (tokens * channels);
    const auto gradient = delta[batch * channels * tokens + token + tokens * o];
    atomicAdd(biasUpdates + o, gradient);
    for (auto c = 0; c < channels; ++c) {
        atomicAdd(weightUpdates + o * channels + c,
                  gradient * context[batch * tokens * channels + token * channels + c]);
        atomicAdd(contextGradient + batch * tokens * channels + token * channels + c,
                  weights[o * channels + c] * gradient);
    }
}

__global__ void attentionBackwardKernel(int n, int tokens, int channels, int heads, float scale,
                                        const float* attention, const float* query, const float* key,
                                        const float* value, const float* contextGradient,
                                        float* queryGradient, float* keyGradient, float* valueGradient)
{
    const auto id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const auto j = id % tokens;
    const auto queryToken = (id / tokens) % tokens;
    const auto head = (id / (tokens * tokens)) % heads;
    const auto batch = id / (heads * tokens * tokens);
    const auto headChannels = channels / heads;
    const auto channelOffset = head * headChannels;
    const auto attentionOffset = batch * heads * tokens * tokens + head * tokens * tokens;
    auto weighted = 0.0f;
    auto valueDot = 0.0f;
    for (auto c = 0; c < headChannels; ++c) {
        const auto gradient = contextGradient[batch*tokens*channels + queryToken*channels + channelOffset+c];
        valueDot += gradient * value[batch*tokens*channels + j*channels + channelOffset+c];
    }
    for (auto kToken = 0; kToken < tokens; ++kToken) {
        auto dot = 0.0f;
        for (auto c = 0; c < headChannels; ++c)
            dot += contextGradient[batch*tokens*channels + queryToken*channels + channelOffset+c]
                   * value[batch*tokens*channels + kToken*channels + channelOffset+c];
        weighted += attention[attentionOffset + queryToken*tokens + kToken] * dot;
    }
    const auto scoreGradient = attention[attentionOffset + queryToken*tokens + j] * (valueDot - weighted);
    for (auto c = 0; c < headChannels; ++c) {
        const auto queryIndex = batch*tokens*channels + queryToken*channels + channelOffset+c;
        const auto keyIndex = batch*tokens*channels + j*channels + channelOffset+c;
        atomicAdd(valueGradient + keyIndex, attention[attentionOffset + queryToken*tokens+j] *
                  contextGradient[queryIndex]);
        atomicAdd(queryGradient + queryIndex, scoreGradient * key[keyIndex] * scale);
        atomicAdd(keyGradient + keyIndex, scoreGradient * query[queryIndex] * scale);
    }
}

__global__ void projectBackwardKernel(int n, int tokens, int channels, const float* input,
                                      const float* qg, const float* kg, const float* vg,
                                      const float* qw, const float* kw, const float* vw,
                                      float* inputGradient, float* qu, float* qbu,
                                      float* ku, float* kbu, float* vu, float* vbu)
{
    const auto id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const auto c = id % channels;
    const auto token = (id / channels) % tokens;
    const auto batch = id / (tokens * channels);
    const auto offset = batch*tokens*channels + token*channels;
    const auto inputValue = input[batch*channels*tokens + token + tokens*c];
    auto inputValueGradient = 0.0f;
    for (auto o = 0; o < channels; ++o) {
        const auto q = qg[offset+o], k = kg[offset+o], v = vg[offset+o];
        atomicAdd(qu + o*channels+c, q * inputValue);
        atomicAdd(ku + o*channels+c, k * inputValue);
        atomicAdd(vu + o*channels+c, v * inputValue);
        inputValueGradient += qw[o*channels+c]*q + kw[o*channels+c]*k + vw[o*channels+c]*v;
        if (c == 0) { atomicAdd(qbu+o, q); atomicAdd(kbu+o, k); atomicAdd(vbu+o, v); }
    }
    inputGradient[batch*channels*tokens + token + tokens*c] = inputValueGradient;
}

void layerNormForwardGpu(int b,int c,int s,float e,const float* i,const float* sc,const float* bi,float* m,float* v,float* n,float* o) { layerNormForwardKernel<<<cudaGridsize(b*s),CUDA_BLOCK_SIZE>>>(b*s,c,s,e,i,sc,bi,m,v,n,o); PX_CUDA_CHECK_LAST(); }
void layerNormBackwardGpu(int b,int c,int s,float e,const float* d,const float* sc,const float* n,const float* v,float* id,float* su,float* bu) { layerNormBackwardKernel<<<cudaGridsize(b*s),CUDA_BLOCK_SIZE>>>(b*s,c,s,e,d,sc,n,v,id,su,bu); PX_CUDA_CHECK_LAST(); }
void positionalEncodingGpu(std::size_t n,const float* i,const float* e,float* o) { addKernel<<<cudaGridsize(n),CUDA_BLOCK_SIZE>>>(n,i,e,o); PX_CUDA_CHECK_LAST(); }
void addGpu(std::size_t n,const float* i,float* o) { copyAddKernel<<<cudaGridsize(n),CUDA_BLOCK_SIZE>>>(n,i,o); PX_CUDA_CHECK_LAST(); }
void selfAttentionProjectGpu(int b,int t,int c,const float* i,const float* qw,const float* qb,const float* kw,const float* kb,const float* vw,const float* vb,float* q,float* k,float* v) { projectKernel<<<cudaGridsize(b*t*c),CUDA_BLOCK_SIZE>>>(b*t*c,t,c,i,qw,qb,kw,kb,vw,vb,q,k,v); PX_CUDA_CHECK_LAST(); }
void selfAttentionForwardGpu(int b,int t,int c,int h,float s,const float* q,const float* k,const float* v,float* a,float* x) { attentionForwardKernel<<<cudaGridsize(b*t*c),CUDA_BLOCK_SIZE>>>(b*t*c,t,c,h,s,q,k,v,a,x); PX_CUDA_CHECK_LAST(); }
void selfAttentionOutputGpu(int b,int t,int c,const float* x,const float* w,const float* bi,float* o) { outputKernel<<<cudaGridsize(b*t*c),CUDA_BLOCK_SIZE>>>(b*t*c,t,c,x,w,bi,o); PX_CUDA_CHECK_LAST(); }
void selfAttentionOutputBackwardGpu(int b,int t,int c,const float* d,const float* x,const float* w,float* xg,float* wu,float* bu) { outputBackwardKernel<<<cudaGridsize(b*t*c),CUDA_BLOCK_SIZE>>>(b*t*c,t,c,d,x,w,xg,wu,bu); PX_CUDA_CHECK_LAST(); }
void selfAttentionAttentionBackwardGpu(int b,int t,int c,int h,float s,const float* a,const float* q,const float* k,const float* v,const float* xg,float* qg,float* kg,float* vg) { attentionBackwardKernel<<<cudaGridsize(b*h*t*t),CUDA_BLOCK_SIZE>>>(b*h*t*t,t,c,h,s,a,q,k,v,xg,qg,kg,vg); PX_CUDA_CHECK_LAST(); }
void selfAttentionProjectBackwardGpu(int b,int t,int c,const float* i,const float* qg,const float* kg,const float* vg,const float* qw,const float* kw,const float* vw,float* ig,float* qu,float* qbu,float* ku,float* kbu,float* vu,float* vbu) { projectBackwardKernel<<<cudaGridsize(b*t*c),CUDA_BLOCK_SIZE>>>(b*t*c,t,c,i,qg,kg,vg,qw,kw,vw,ig,qu,qbu,ku,kbu,vu,vbu); PX_CUDA_CHECK_LAST(); }

} // namespace px
