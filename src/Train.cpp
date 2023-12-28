/********************************************************************************
* Copyright 2023 Thomas A. Rieck, All Rights Reserved
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
#include "Model.h"

namespace po = boost::program_options;

using namespace px;

namespace px {

void train(const std::string& cfgFile, const std::string& weightsFile,
           const po::variables_map& options)
{
    auto model = Model(cfgFile, options);
    model.train();
}

}
int main(int argc, char* argv[])
{
    po::options_description desc("options");
    po::positional_options_description pod;
    pod.add("config-file", 1);
    pod.add("weights-file", 1);

    desc.add_options()
            ("config-file", po::value<std::string>()->required(), "Configuration file")
            ("weights-file", po::value<std::string>()->required(), "Weights file")
            ("help", "Print program usage")
            ("no-gpu", po::bool_switch()->default_value(true), "Use CPU for processing")
            ("clear-weights", po::bool_switch()->default_value(false), "Clear target weights");

    po::options_description hidden;
    hidden.add_options()("train", po::bool_switch()->default_value(true));

    po::options_description all;
    all.add(desc).add(hidden);

    try {
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(all).positional(pod).run(), vm);
        if (vm.count("help") || argc < 3) {
            std::cerr << "usage: pixienn-train [options] config-file weights-file" << std::endl;
            std::cerr << desc << std::endl;
            exit(1);
        }

        po::notify(vm);

        auto config = vm["config-file"].as<std::string>();
        auto weights = vm["weights-file"].as<std::string>();

        train(config, weights, vm);

    } catch (const px::Error& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }

    return 0;
}
