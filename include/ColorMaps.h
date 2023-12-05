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

#ifndef PIXIENN_COLORMAPS_H
#define PIXIENN_COLORMAPS_H

namespace px {

class ColorMaps
{
public:
    ColorMaps();
    explicit ColorMaps(const std::string& mapName);
    ~ColorMaps();

    uint32_t color(uint32_t index) const;
    static std::vector<std::string> maps();
private:
    class Iterator;
    Iterator* it_;
};

}

#endif //PIXIENN_COLORMAPS_H
