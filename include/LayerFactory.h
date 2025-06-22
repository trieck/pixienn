#pragma once

#include "AvgPoolLayer.h"
#include "BatchNormLayer.h"
#include "ConnLayer.h"
#include "ConvLayer.h"
#include "DetectLayer.h"
#include "DropoutLayer.h"
#include "MaxPoolLayer.h"
#include "RegionLayer.h"
#include "RouteLayer.h"
#include "ShortcutLayer.h"
#include "Singleton.h"
#include "SoftmaxLayer.h"
#include "UpsampleLayer.h"
#include "YoloLayer.h"

namespace px {

template<Device D = Device::CPU>
class LayerFactories : public Singleton<LayerFactories<D>>
{
public:
    using LayerPtr = typename Layer<D>::Ptr;

    LayerFactories();

    template<typename L>
    void registerFactory(const char* name);

    static LayerPtr create(Model<D>& model, YAML::Node layerDef);

private:
    using LayerFactory = std::function<LayerPtr(Model<D>&, YAML::Node)>;
    std::unordered_map<std::string, LayerFactory> factories_;
};


template<Device D>
LayerFactories<D>::LayerFactories()
{
    registerFactory<AvgPoolLayer<D>>("avgpool");
    registerFactory<BatchNormLayer<D>>("batchnorm");
    registerFactory<ConnLayer<D>>("connected");
    registerFactory<ConvLayer<D>>("conv");
    registerFactory<DetectLayer<D>>("detection");
    registerFactory<DropoutLayer<D>>("dropout");
    registerFactory<MaxPoolLayer<D>>("maxpool");
    registerFactory<RegionLayer<D>>("region");
    registerFactory<RouteLayer<D>>("route");
    registerFactory<ShortcutLayer<D>>("shortcut");
    registerFactory<SoftmaxLayer<D>>("softmax");
    registerFactory<UpsampleLayer<D>>("upsample");
    registerFactory<YoloLayer<D>>("yolo");
}

template<Device D>
template<typename L>
void LayerFactories<D>::registerFactory(const char* name)
{
    factories_[name] = [](Model<D>& model, YAML::Node layerDef) {
        return std::make_shared<L>(model, layerDef);
    };
}

template<Device D>
LayerFactories<D>::LayerPtr LayerFactories<D>::create(Model<D>& model, YAML::Node layerDef)
{
    PX_CHECK(layerDef.IsMap(), "Layer definition must be a map");

    auto instance = LayerFactories<D>::instance();

    const auto type = layerDef["type"];
    PX_CHECK(type.IsDefined(), "Layer definition must have a \"type\" property");

    auto stype = type.as<std::string>();

    auto it = instance.factories_.find(stype);
    if (it == std::end(instance.factories_)) {
        PX_ERROR_THROW("Unable to find a layer factory for layer type \"%s\".", stype.c_str());
    }

    auto ptr = it->second(model, layerDef);

    return ptr;
}

}   // px
