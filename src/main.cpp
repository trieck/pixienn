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

#include <fstream>
#include <iostream>

#include "Common.h"
#include "Error.h"
#include "Image.h"
#include "Model.h"
#include "NMS.h"

namespace po = boost::program_options;

using namespace px;

void predict(const std::string& cfgFile, const std::string& imageFile,
             const po::variables_map& options)
{
    auto model = Model(cfgFile, options);
    auto detects = model.predict(imageFile);

    auto nmsThreshold = options["nms"].as<float>();
    nms(detects, nmsThreshold);

    model.overlay(imageFile, detects);
    auto json = model.asJson(detects);

    std::ofstream ofs("predictions.geojson", std::ios::out | std::ios::binary);
    PX_CHECK(ofs.good(), "Could not open file \"%s\".", "results.geojson");
    ofs << json << std::flush;
    ofs.close();
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "usage: pixienn [options] config-file image-file" << std::endl;
        exit(1);
    }

    po::options_description desc("options");
    po::positional_options_description pod;
    pod.add("config-file", 1);
    pod.add("image-file", 1);

    desc.add_options()
            ("no-gpu", "Use CPU for processing")
            ("confidence", po::value<float>()->default_value(0.2f))
            ("nms", po::value<float>()->default_value(0.3f))
            ("config-file", po::value<std::string>()->required(), "Configuration file")
            ("image-file", po::value<std::string>()->required(), "Image file");

    try {
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(desc).positional(pod).run(), vm);
        po::notify(vm);

        auto config = vm["config-file"].as<std::string>();
        auto image = vm["image-file"].as<std::string>();

        predict(config, image, vm);
    } catch (const px::Error& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }

    return 0;
}

