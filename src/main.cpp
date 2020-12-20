/********************************************************************************
* Copyright 2020 Thomas A. Rieck, All Rights Reserved
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

#include "Error.h"
#include "Model.h"
#include "NMS.h"
#include "Tensor.cuh"

#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>

namespace po = boost::program_options;

using namespace px;

void predict(const char* cfgFile, const char* imageFile)
{
    auto model = Model(cfgFile);

    auto detects = model.predict(imageFile, 0.2f);
    nms(detects, 0.4f);

    auto json = model.asJson(std::move(detects));

    std::ofstream ofs("results.geojson", std::ios::out | std::ios::binary);
    PX_CHECK(ofs.good(), "Could not open file \"%s\".", "results.geojson");
    ofs << json << std::flush;
    ofs.close();

    std::cout << "done." << std::endl;
}

int main(int argc, char* argv[])
{
//    po::options_description desc("options");
//    po::variables_map vm;
//    po::store(po::parse_command_line(argc, argv, desc), vm);
//    po::notify(vm);

//    if (argc < 3) {
//        std::cerr << "usage: pixienn metadata-file image-file" << std::endl;
//        exit(1);
//    }

    try {
        foobar();

        // predict(argv[1], argv[2]);
    } catch (const px::Error& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }

    return 0;
}

