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

#ifndef PIXIENN_MODEL_H
#define PIXIENN_MODEL_H

#include "Layer.h"

namespace px {

class Model
{
public:
    Model(const std::string& filename);
    Model(const Model& rhs) = default;
    Model(Model&& rhs) = default;

    Model& operator=(const Model& rhs) = default;
    Model& operator=(Model&& rhs) = default;

    using LayerVec = std::vector<Layer::Ptr>;

    const LayerVec& layers() const;

    const int batch() const;
    const int channels() const;
    const int height() const;
    const int width() const;

private:
    void parse();
    std::string filename_;
    int batch_ = 0, channels_ = 0, height_ = 0, width_ = 0;

    LayerVec layers_;
};

}   // px

#endif // PIXIENN_MODEL_H
