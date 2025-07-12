/********************************************************************************
* Copyright 2025 Thomas A. Rieck, All Rights Reserved
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

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <hdf5/serial/H5Cpp.h>
#include <hdf5/serial/H5File.h>

#include "Model.h"

using namespace px;
using namespace testing;
using namespace H5;

static std::string readStringAttr(const H5::Attribute& attr)
{
    auto dtype = attr.getDataType();
    if (dtype.getClass() != H5T_STRING) {
        throw std::runtime_error("Attribute is not a string");
    }

    // Read into std::string
    std::string result;
    attr.read(dtype, result);

    return result;
}

static int readIntAttr(const H5::Attribute& attr)
{
    auto dtype = attr.getDataType();
    if (dtype.getClass() != H5T_INTEGER) {
        throw std::runtime_error("Attribute is not an integer");
    }

    // Read into int
    int result;
    attr.read(dtype, &result);

    return result;
}

static bool readBoolAttr(const H5::Attribute& attr)
{
    H5::DataType dtype = attr.getDataType();

    // Read as uint8_t unconditionally
    uint8_t value = 0;
    attr.read(H5::PredType::NATIVE_UINT8, &value);
    return value != 0;
}

struct Tensor
{
    Tensor(PxCpuVector d, std::vector<hsize_t> dim)
            : data(std::move(d)), dims(std::move(dim))
    {}

    Tensor() = default;
    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) = default;

    PxCpuVector data;
    std::vector<hsize_t> dims;
};

static Tensor readTensor(const H5::DataSet& dataset)
{
    const auto dtype = dataset.getDataType();
    if (dtype.getClass() != H5T_FLOAT || dtype.getSize() != sizeof(float)) {
        throw std::runtime_error("Dataset is not a float32 tensor");
    }

    const auto space = dataset.getSpace();
    const auto ndims = space.getSimpleExtentNdims();

    std::vector<hsize_t> dims(ndims);
    space.getSimpleExtentDims(dims.data());

    const auto totalElements = std::accumulate(
            dims.begin(), dims.end(), hsize_t(1), std::multiplies<>()
    );

    PxCpuVector v(totalElements);
    dataset.read(v.data(), H5::PredType::NATIVE_FLOAT);

    Tensor tensor(std::move(v), std::move(dims));

    return tensor;
}

void normalize(PxCpuVector& vec)
{
    auto maxVal = *std::max_element(vec.begin(), vec.end());
    if (maxVal > 1e-8f) {
        for (auto& v: vec) {
            v /= maxVal;
        }
    }
}

void compareTensors(const PxCpuVector& baseline,
                    const PxCpuVector& computed,
                    float meanAllowedDiff = 0.05f)
{
    ASSERT_EQ(baseline.size(), computed.size()) << "Tensor size mismatch";

    PxCpuVector normBaseline = baseline;
    PxCpuVector normComputed = computed;

    normalize(normBaseline);
    normalize(normComputed);

    auto meanDiff = 0.0f;
    for (auto i = 0; i < normBaseline.size(); ++i) {
        auto diff = std::abs(normBaseline[i] - normComputed[i]);
        meanDiff += diff;
    }

    meanDiff /= normBaseline.size();

    EXPECT_LT(meanDiff, meanAllowedDiff) << "Mean diff too large: " << meanDiff;
}

static void runConnLayer(const Group& layerGroup)
{
    auto activation = readStringAttr(layerGroup.openAttribute("activation"));
    auto batchNorm = layerGroup.attrExists("batch_normalize") &&
                     readBoolAttr(layerGroup.openAttribute("batch_normalize")) != 0;

    auto input = readTensor(layerGroup.openDataSet("input"));
    auto output = readTensor(layerGroup.openDataSet("output"));
    auto weights = readTensor(layerGroup.openDataSet("weights"));
    auto bias = readTensor(layerGroup.openDataSet("bias"));

    Model<Device::CPU> model;

    YAML::Node config;
    config["type"] = "connected";
    config["batch"] = input.dims[0];
    config["activation"] = activation;
    config["batch_normalize"] = batchNorm;
    config["inputs"] = input.dims[1];
    config["output"] = weights.dims[0];

    ConnLayer<Device::CPU> layer(model, config);
    layer.copyWeights(weights.data);
    layer.copyBiases(bias.data);

    if (batchNorm) {
        auto mean = readTensor(layerGroup.openDataSet("mean"));
        auto var = readTensor(layerGroup.openDataSet("variance"));
        auto scale = readTensor(layerGroup.openDataSet("scale"));
        layer.copyRollingMean(mean.data);
        layer.copyRollingVariance(var.data);
        layer.copyScales(scale.data);
    }

    layer.forward(input.data);

    compareTensors(output.data, layer.output());
}

static void runConvLayer(const Group& layerGroup)
{
    auto dilation = readIntAttr(layerGroup.openAttribute("dilation"));
    auto kernel = readIntAttr(layerGroup.openAttribute("kernel"));
    auto stride = readIntAttr(layerGroup.openAttribute("stride"));
    auto pad = readIntAttr(layerGroup.openAttribute("padding"));
    auto activation = readStringAttr(layerGroup.openAttribute("activation"));
    auto batchNorm = layerGroup.attrExists("batch_normalize") &&
                     readBoolAttr(layerGroup.openAttribute("batch_normalize")) != 0;

    auto input = readTensor(layerGroup.openDataSet("input"));
    auto output = readTensor(layerGroup.openDataSet("output"));
    auto weights = readTensor(layerGroup.openDataSet("weights"));
    auto bias = readTensor(layerGroup.openDataSet("bias"));

    auto batch = input.dims[0];
    auto channels = input.dims[1];
    auto height = input.dims[2];
    auto width = input.dims[3];

    Model<Device::CPU> model;

    YAML::Node config;
    config["type"] = "conv";
    config["batch"] = batch;
    config["channels"] = channels;
    config["height"] = height;
    config["width"] = width;
    config["filters"] = weights.dims[0];
    config["kernel"] = kernel;
    config["stride"] = stride;
    config["pad"] = pad > 0;
    config["dilation"] = dilation;
    config["activation"] = activation;
    config["batch_normalize"] = batchNorm;

    ConvLayer<Device::CPU> layer(model, config);
    layer.copyWeights(weights.data);
    layer.copyBiases(bias.data);

    if (batchNorm) {
        auto mean = readTensor(layerGroup.openDataSet("mean"));
        auto var = readTensor(layerGroup.openDataSet("variance"));
        auto scale = readTensor(layerGroup.openDataSet("scale"));
        layer.copyRollingMean(mean.data);
        layer.copyRollingVariance(var.data);
        layer.copyScales(scale.data);
    }

    layer.forward(input.data);

    compareTensors(output.data, layer.output());
}

static void runMaxPoolLayer(const Group& layerGroup)
{
    auto dilation = readIntAttr(layerGroup.openAttribute("dilation"));
    auto kernel = readIntAttr(layerGroup.openAttribute("kernel"));
    auto stride = readIntAttr(layerGroup.openAttribute("stride"));
    auto pad = readIntAttr(layerGroup.openAttribute("padding"));

    auto input = readTensor(layerGroup.openDataSet("input"));
    auto output = readTensor(layerGroup.openDataSet("output"));

    auto batch = input.dims[0];
    auto channels = input.dims[1];
    auto height = input.dims[2];
    auto width = input.dims[3];

    Model<Device::CPU> model;

    YAML::Node config;
    config["type"] = "conv";
    config["batch"] = batch;
    config["channels"] = channels;
    config["height"] = height;
    config["width"] = width;
    config["kernel"] = kernel;
    config["stride"] = stride;
    config["padding"] = pad;
    config["dilation"] = dilation;

    MaxPoolLayer<Device::CPU> layer(model, config);

    layer.forward(input.data);

    compareTensors(output.data, layer.output());
}

static void runLayer(const Group& layerGroup)
{
    auto groupName = layerGroup.getObjName();
    auto layerType = readStringAttr(layerGroup.openAttribute("layer_type"));
    if (layerType == "connected") {
        runConnLayer(layerGroup);
    } else if (layerType == "conv") {
        runConvLayer(layerGroup);
    } else if (layerType == "maxpool") {
        runMaxPoolLayer(layerGroup);
    }
}

GTEST_TEST(ModelSuite, RunModel)
{
    H5File file(TESTS_H5_PATH, H5F_ACC_RDONLY);

    auto numLayers = file.getNumObjs();
    ASSERT_GT(numLayers, 0) << "No layers found in the HDF5 file.";

    for (auto i = 0u; i < numLayers; ++i) {
        auto type = file.childObjType(i);
        ASSERT_EQ(type, H5O_TYPE_GROUP);

        auto layerName = file.getObjnameByIdx(i);
        auto layerGroup = file.openGroup(layerName);
        runLayer(layerGroup);
    }
}
