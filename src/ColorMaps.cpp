/********************************************************************************
* Copyright 2023 Thomas A. Rieck, All Rights Reserved

* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at

*    http://www.apache.org/licenses/LICENSE-2.0

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
********************************************************************************/

#include "Common.h"
#include "ColorMaps.h"
#include "Error.h"
#include "MatplotlibColorMaps.h"

namespace px {

enum class ColorMapKind {
    Qualitative,
    Continuous
};

struct ColorMapEntry {
    const uint32_t* colors;
    std::size_t size;
    ColorMapKind kind;
};

using ColorMap = std::unordered_map<std::string, ColorMapEntry>;

#define MAP(name, kind) { #name, { matplotlib::name, sizeof(matplotlib::name) / sizeof(uint32_t), kind } }

static const ColorMap colorMap = {
        MAP(Accent, ColorMapKind::Qualitative),
        MAP(Dark2, ColorMapKind::Qualitative),
        MAP(Paired, ColorMapKind::Qualitative),
        MAP(Pastel1, ColorMapKind::Qualitative),
        MAP(Pastel2, ColorMapKind::Qualitative),
        MAP(Set1, ColorMapKind::Qualitative),
        MAP(Set2, ColorMapKind::Qualitative),
        MAP(Set3, ColorMapKind::Qualitative),
        MAP(tab10, ColorMapKind::Qualitative),
        MAP(tab20, ColorMapKind::Qualitative),
        MAP(tab20b, ColorMapKind::Qualitative),
        MAP(tab20c, ColorMapKind::Qualitative),
        MAP(Blues, ColorMapKind::Continuous),
        MAP(BuGn, ColorMapKind::Continuous),
        MAP(BuPu, ColorMapKind::Continuous),
        MAP(GnBu, ColorMapKind::Continuous),
        MAP(Greens, ColorMapKind::Continuous),
        MAP(Greys, ColorMapKind::Continuous),
        MAP(Oranges, ColorMapKind::Continuous),
        MAP(OrRd, ColorMapKind::Continuous),
        MAP(PuBu, ColorMapKind::Continuous),
        MAP(PuBuGn, ColorMapKind::Continuous),
        MAP(PuRd, ColorMapKind::Continuous),
        MAP(Purples, ColorMapKind::Continuous),
        MAP(RdPu, ColorMapKind::Continuous),
        MAP(RdYlGn, ColorMapKind::Continuous),
        MAP(Reds, ColorMapKind::Continuous),
        MAP(YlGn, ColorMapKind::Continuous),
        MAP(YlGnBu, ColorMapKind::Continuous),
        MAP(YlOrBr, ColorMapKind::Continuous),
        MAP(YlOrRd, ColorMapKind::Continuous),
        MAP(afmhot, ColorMapKind::Continuous),
        MAP(autumn, ColorMapKind::Continuous),
        MAP(binary, ColorMapKind::Continuous),
        MAP(bone, ColorMapKind::Continuous),
        MAP(cividis, ColorMapKind::Continuous),
        MAP(cool, ColorMapKind::Continuous),
        MAP(coolwarm, ColorMapKind::Continuous),
        MAP(copper, ColorMapKind::Continuous),
        MAP(gist_earth, ColorMapKind::Continuous),
        MAP(gist_gray, ColorMapKind::Continuous),
        MAP(gist_heat, ColorMapKind::Continuous),
        MAP(gist_ncar, ColorMapKind::Continuous),
        MAP(gist_rainbow, ColorMapKind::Continuous),
        MAP(gist_yarg, ColorMapKind::Continuous),
        MAP(gnuplot, ColorMapKind::Continuous),
        MAP(gnuplot2, ColorMapKind::Continuous),
        MAP(gray, ColorMapKind::Continuous),
        MAP(hot, ColorMapKind::Continuous),
        MAP(hsv, ColorMapKind::Continuous),
        MAP(inferno, ColorMapKind::Continuous),
        MAP(jet, ColorMapKind::Continuous),
        MAP(magma, ColorMapKind::Continuous),
        MAP(nipy_spectral, ColorMapKind::Continuous),
        MAP(ocean, ColorMapKind::Continuous),
        MAP(pink, ColorMapKind::Continuous),
        MAP(plasma, ColorMapKind::Continuous),
        MAP(prism, ColorMapKind::Continuous),
        MAP(rainbow, ColorMapKind::Continuous),
        MAP(seismic, ColorMapKind::Continuous),
        MAP(spring, ColorMapKind::Continuous),
        MAP(summer, ColorMapKind::Continuous),
        MAP(terrain, ColorMapKind::Continuous),
        MAP(turbo, ColorMapKind::Continuous),
        MAP(twilight, ColorMapKind::Continuous),
        MAP(twilight_shifted, ColorMapKind::Continuous),
        MAP(viridis, ColorMapKind::Continuous),
        MAP(winter, ColorMapKind::Continuous)
};

#undef MAP

class ColorMaps::Iterator
{
public:
    ColorMap::const_iterator it;
};

ColorMaps::ColorMaps() : it_(new Iterator())
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    it_->it = colorMap.begin();
    std::advance(it_->it, std::rand() % colorMap.size());
}

ColorMaps::ColorMaps(const std::string& name) : it_(new Iterator())
{
    it_->it = colorMap.find(name);
    PX_CHECK(it_->it != colorMap.end(), "Color map \"%s\" not found.", name.c_str());
}

ColorMaps::~ColorMaps()
{
    delete it_;
}

uint32_t ColorMaps::color(uint32_t index) const
{
    const auto& map = it_->it->second;
    index *= 2654435761U;
    return map.colors[index % map.size];
}

uint32_t ColorMaps::sample(float value) const
{
    const auto& map = it_->it->second;
    PX_CHECK(map.kind == ColorMapKind::Continuous,
             "Color map \"%s\" is not a continuous Matplotlib colormap.",
             it_->it->first.c_str());

    if (!std::isfinite(value)) {
        value = 0.0f;
    }
    value = std::max(0.0f, std::min(1.0f, value));

    // This matches Matplotlib's 256-entry normalized lookup behavior:
    // values in [i/256, (i+1)/256) select entry i, with 1.0 selecting 255.
    const auto index = std::min(map.size - 1,
                                static_cast<std::size_t>(value * map.size));
    return map.colors[index];
}

std::vector<std::string> ColorMaps::maps()
{
    std::vector<std::string> names;
    names.reserve(colorMap.size());
    for (const auto& entry: colorMap) {
        names.emplace_back(entry.first);
    }
    std::sort(names.begin(), names.end());
    return names;
}

bool ColorMaps::isContinuous(const std::string& mapName)
{
    const auto it = colorMap.find(mapName);
    PX_CHECK(it != colorMap.end(), "Color map \"%s\" not found.", mapName.c_str());
    return it->second.kind == ColorMapKind::Continuous;
}

} // namespace px
