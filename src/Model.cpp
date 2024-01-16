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

#include "Common.h"

#include "Model.h"

namespace px {

BaseModel::Ptr BaseModel::create(const std::string& cfgFile, BaseModel::var_map options)
{
    auto useGpu = !options["no-gpu"].as<bool>();

    return createModel(cfgFile, std::move(options), useGpu);
}

BaseModel::Ptr BaseModel::createModel(const std::string& cfgFile, BaseModel::var_map options, bool useGpu)
{
#ifdef USE_CUDA
    if (useGpu) {
        return std::make_unique<CudaModel>(cfgFile, options);
    }
#endif  // USE_CUDA

    return std::make_unique<CpuModel>(cfgFile, options);
}

}   // px
