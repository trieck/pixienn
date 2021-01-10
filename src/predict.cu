
#include <yaml-cpp/yaml.h>

#include "model_t.h"
#include "NMS.h"

namespace px {

template<typename T>
static void predict_t(const char* cfgFile, const char* imageFile)
{
    model_t<T> model(cfgFile);

    auto detects = model.predict(imageFile, 0.2f);
    nms(detects, 0.4f);

    auto json = model.asJson(std::move(detects));

    std::ofstream ofs("results.geojson", std::ios::out | std::ios::binary);
    PX_CHECK(ofs.good(), "Could not open file \"%s\".", "results.geojson");
    ofs << json << std::flush;
    ofs.close();

    std::cout << "done." << std::endl;
}

void predict(const char* cfgFile, const char* imageFile, bool useGPU)
{
    if (useGPU) {
        predict_t<cuda_array>(cfgFile, imageFile);
    } else {
        predict_t<cpu_array>(cfgFile, imageFile);
    }
}

}   // namespace px