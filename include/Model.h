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

PX_BEGIN

class Model
{
private:
    Model(const std::string& filename);

public:
    using Ptr = std::shared_ptr<Model>;

    static Model::Ptr create(const std::string& filename);

private:
    void parse();
    std::string filename_;
    int batch_ = 0, channels_ = 0, height_ = 0, width_ = 0;

    std::vector<Layer::Ptr> layers_;
};

PX_END

#endif // PIXIENN_MODEL_H
