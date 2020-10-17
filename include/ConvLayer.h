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

#ifndef PIXIENN_CONVLAYER_H
#define PIXIENN_CONVLAYER_H

#include "Layer.h"

PX_BEGIN

class ConvLayer : public Layer
{
protected:
    ConvLayer(const YAML::Node& layerDef);

public:
    virtual ~ConvLayer();

private:
    friend class Layer;

    int dilation_ = 0, filters_ = 0, kernel_ = 0, pad_ = 0, stride_ = 0;
};

PX_END

#endif // PIXIENN_CONVLAYER_H
