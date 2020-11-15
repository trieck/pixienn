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

#include <fstream>
#include "Error.h"
#include "Model.h"

namespace px {

void loadDarknetWeights(const Model& model, const std::string& filename)
{
    std::ifstream ifs(filename, std::ios::in | std::ios::binary);
    PX_CHECK(ifs.good(), "Could not open file \"%s\".", filename.c_str());

    ifs.seekg(0, ifs.end);
    auto length = ifs.tellg();
    ifs.seekg(0, ifs.beg);

    int major, minor, revision;

    ifs.read((char*) &major, sizeof(int));
    ifs.read((char*) &minor, sizeof(int));
    ifs.read((char*) &revision, sizeof(int));

    if ((major * 10 + minor) >= 2 && major < 1000 && minor < 1000) {
        size_t seen;
        ifs.read((char*) &seen, sizeof(size_t));
    } else {
        int iseen = 0;
        ifs.read((char*) &iseen, sizeof(int));
    }

    for (const auto& layer: model.layers()) {
        layer->loadDarknetWeights(ifs);
    }

    PX_CHECK(ifs.tellg() == length, "Did not fully read weights file.  Model/Weights mismatch?");

    ifs.close();
}

}   // px