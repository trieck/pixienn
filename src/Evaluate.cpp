#include "Common.h"
#include "Error.h"
#include "Model.h"

namespace po = boost::program_options;
using namespace px;

int main(int argc, char* argv[])
{
    po::options_description desc("options");
    po::positional_options_description pod;
    pod.add("config-file", 1);
    pod.add("weights-file", 1);
    desc.add_options()
            ("config-file", po::value<std::string>()->required(), "Configuration file")
            ("weights-file", po::value<std::string>()->required(), "Weights/checkpoint file")
            ("all-validation", po::bool_switch()->default_value(false), "Evaluate the full validation manifest")
            ("batch-size", po::value<int>(), "Override evaluation batch size")
            ("help", po::bool_switch()->default_value(false), "Print program usage")
            ("no-gpu", po::bool_switch()->default_value(false), "Use CPU for processing")
            ("train", po::bool_switch()->default_value(true), "Use the configured validation batch shape");

    try {
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(desc).positional(pod).run(), vm);
        if (vm["help"].as<bool>() || argc < 3) {
            std::cerr << "usage: pixienn-eval [options] config-file weights-file\n" << desc << std::endl;
            return 1;
        }
        po::notify(vm);
        auto model = BaseModel::create(vm["config-file"].as<std::string>(), vm);
        model->evaluate();
    } catch (const px::Error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}
