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

#include "Common.h"
#include "Error.h"

#include <boost/program_options.hpp>
#include <iostream>

namespace po = boost::program_options;

using namespace px;

namespace px { extern void predict(const char* cfgFile, const char* imageFile, bool useGPU); }

int main(int argc, char* argv[])
{
    std::vector<std::string> extra;

    bool useGPU;
    po::options_description desc("options");
    desc.add_options()
            ("gpu", po::bool_switch(&useGPU), "Run model on GPU.")
            ("extra", po::value(&extra));

    po::positional_options_description p;
    p.add("extra", -1);

    try {
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
        po::notify(vm);

        if (extra.size() < 2) {
            PX_ERROR_THROW("usage: pixienn [--gpu] model-file image-file");
        }

        predict(extra[0].c_str(), extra[1].c_str(), useGPU);
    } catch (const px::Error& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }

    return 0;
}

