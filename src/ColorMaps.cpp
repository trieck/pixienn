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
#include "ColorMaps.h"
#include "Error.h"

namespace px {

static constexpr uint32_t tab20c[] = {
        0x3182bd,  // Blue
        0x6baed6,  // Light Blue
        0x9ecae1,  // Lighter Blue
        0xc6dbef,  // Lightest Blue
        0xe6550d,  // Red-Orange
        0xfd8d3c,  // Light Orange
        0xfdae6b,  // Darker Orange
        0xfdd0a2,  // Lightest Orange
        0x31a354,  // Green
        0x74c476,  // Light Green
        0xa1d99b,  // Lighter Green
        0xc7e9c0,  // Lightest Green
        0x756bb1,  // Purple
        0x9e9ac8,  // Light Purple
        0xbcbddc,  // Lighter Purple
        0xdadaeb,  // Lightest Purple
        0x636363,  // Dark Gray
        0x969696,  // Medium Gray
        0xbdbdbd,  // Light Gray
        0xd9d9d9   // Lightest Gray
};

static constexpr uint32_t Set1[] = {
        0xe41a1c,  // Red
        0x377eb8,  // Blue
        0x4daf4a,  // Green
        0x984ea3,  // Purple
        0xff7f00,  // Orange
        0xffff33,  // Yellow
        0xa65628,  // Brown
        0xf781bf,  // Pink
        0x999999   // Gray
};

static constexpr uint32_t Paired[] = {
        0xa6cee3,  // Light Blue
        0x1f78b4,  // Dark Blue
        0xb2df8a,  // Light Green
        0x33a02c,  // Dark Green
        0xfb9a99,  // Light Red
        0xe31a1c,  // Dark Red
        0xfdbf6f,  // Light Orange
        0xff7f00,  // Dark Orange
        0xcab2d6,  // Light Purple
        0x6a3d9a,  // Dark Purple
        0xffff99,  // Light Yellow
        0xb15928   // Dark Brown
};

static constexpr uint32_t Accent[] = {
        0x7fc97f,  // Green
        0xbeaed4,  // Purple
        0xfdc086,  // Orange
        0xffff99,  // Yellow
        0x386cb0,  // Blue
        0xf0027f,  // Pink
        0xbf5b17,  // Brown
        0x666666   // Gray
};

static constexpr uint32_t tab10[] = {
        0x1f77b4,  // Blue
        0xff7f0e,  // Orange
        0x2ca02c,  // Green
        0xd62728,  // Red
        0x9467bd,  // Purple
        0x8c564b,  // Brown
        0xe377c2,  // Pink
        0x7f7f7f,  // Gray
        0xbcbd22,  // Olive
        0x17becf   // Cyan
};

static constexpr uint32_t tab20[] = {
        0x1f77b4,  // Blue
        0xaec7e8,  // Light Blue
        0xff7f0e,  // Orange
        0xffbb78,  // Light Orange
        0x2ca02c,  // Green
        0x98df8a,  // Light Green
        0xd62728,  // Red
        0xff9896,  // Light Red
        0x9467bd,  // Purple
        0xc5b0d5,  // Light Purple
        0x8c564b,  // Brown
        0xc49c94,  // Light Brown
        0xe377c2,  // Pink
        0xf7b6d2,  // Light Pink
        0x7f7f7f,  // Gray
        0xc7c7c7,  // Light Gray
        0xbcbd22,  // Olive
        0xdbdb8d,  // Light Olive
        0x17becf,  // Cyan
        0x9edae5   // Light Cyan
};

static constexpr uint32_t crayola16[] = {
        0xed0a3f,  // Radical Red
        0xff681f,  // Outrageous Orange
        0xff8833,  // Sunset Orange
        0xffae42,  // Selective Yellow
        0xfbe870,  // Maximum Yellow Red
        0xc5e17a,  // Granny Smith Apple
        0x3aa655,  // Tropical Rain Forest
        0x0095b7,  // Pacific Blue
        0x0066ff,  // Blue
        0x6456b7,  // Blue Violet
        0x8359a3,  // Royal Purple
        0xbb3385,  // Jazzberry Jam
        0xffa6c9,  // Pink Sherbet
        0xaf593e,  // Burnt Sienna
        0x000000,  // Black
        0xffffff   // White
};


static constexpr uint32_t darknet[] = {
        0xff00ff,  // Magenta
        0x0000ff,  // Blue
        0x00ffff,  // Cyan
        0x00ff00,  // Green
        0xffff00,  // Yellow
        0xff0000   // Red
};

static constexpr uint32_t fluorescent[] = {
        0xff355e,  // Radical Red
        0xfd5b78,  // Wild Watermelon
        0xff6037,  // Outrageous Orange
        0xff9966,  // Atomic Tangerine
        0xff9933,  // Neon Carrot
        0xffcc33,  // Sunglow
        0xffff66,  // Laser Lemon
        0xccff00,  // Electric Lime
        0x66ff66,  // Screamin' Green
        0xaaf0d1,  // Magic Mint
        0x50bfe6,  // Blizzard Blue
        0xff6eff,  // Ultra Pink
        0xee34d2,  // Shocking Pink
};

static constexpr uint32_t vivid[] = {
        0xFF0000,  // Red
        0xFFA500,  // Orange
        0xFFFF00,  // Yellow
        0x00FF00,  // Lime
        0x00FFFF,  // Cyan
        0x0000FF,  // Blue
        0xFF00FF,  // Magenta
        0xFF1493,  // Deep Pink
        0x8B4513,  // Saddle Brown
        0x32CD32,  // Lime Green
        0x9932CC,  // Dark Orchid
        0xFF4500,  // Orange Red
        0x228B22,  // Forest Green
        0x8A2BE2,  // Blue Violet
        0x20B2AA,  // Light Sea Green
        0xFF6347   // Tomato
};

static constexpr uint32_t off_the_path[] = {
        0xff00cc,  // Electric Violet
        0xff9933,  // Sunburst Orange
        0x33ccff,  // Lagoon Blue
        0xff3300,  // Blaze Red
        0x66ff33,  // Neon Lime
        0xff3399,  // Fuchsia Flash
        0x00cc66,  // Jungle Green
        0xffcc00,  // Solar Yellow
        0x9966ff,  // Cosmic Lavender
        0xff6600,  // Radiant Amber
        0x00ffcc,  // Turbo Teal
        0xcc66ff   // Mystical Magenta
};

static constexpr uint32_t vga[256] = {
        0x000000, 0x0000AA, 0x00AA00, 0x00AAAA, 0xAA0000, 0xAA00AA, 0xAA5500, 0xAAAAAA,
        0x555555, 0x5555FF, 0x55FF55, 0x55FFFF, 0xFF5555, 0xFF55FF, 0xFFFF55, 0xFFFFFF,
        0x000000, 0x00005F, 0x000087, 0x0000AF, 0x0000D7, 0x0000FF, 0x005F00, 0x005F5F,
        0x005F87, 0x005FAF, 0x005FD7, 0x005FFF, 0x008700, 0x00875F, 0x008787, 0x0087AF,
        0x0087D7, 0x0087FF, 0x00AF00, 0x00AF5F, 0x00AF87, 0x00AFAF, 0x00AFD7, 0x00AFFF,
        0x00D700, 0x00D75F, 0x00D787, 0x00D7AF, 0x00D7D7, 0x00D7FF, 0x00FF00, 0x00FF5F,
        0x00FF87, 0x00FFAF, 0x00FFD7, 0x00FFFF, 0x5F0000, 0x5F005F, 0x5F0087, 0x5F00AF,
        0x5F00D7, 0x5F00FF, 0x5F5F00, 0x5F5F5F, 0x5F5F87, 0x5F5FAF, 0x5F5FD7, 0x5F5FFF,
        0x5F8700, 0x5F875F, 0x5F8787, 0x5F87AF, 0x5F87D7, 0x5F87FF, 0x5FAF00, 0x5FAF5F,
        0x5FAF87, 0x5FAFAF, 0x5FAFD7, 0x5FAFFF, 0x5FD700, 0x5FD75F, 0x5FD787, 0x5FD7AF,
        0x5FD7D7, 0x5FD7FF, 0x5FFF00, 0x5FFF5F, 0x5FFF87, 0x5FFFAF, 0x5FFFD7, 0x5FFFFF,
        0x870000, 0x87005F, 0x870087, 0x8700AF, 0x8700D7, 0x8700FF, 0x875F00, 0x875F5F,
        0x875F87, 0x875FAF, 0x875FD7, 0x875FFF, 0x878700, 0x87875F, 0x878787, 0x8787AF,
        0x8787D7, 0x8787FF, 0x87AF00, 0x87AF5F, 0x87AF87, 0x87AFAF, 0x87AFD7, 0x87AFFF,
        0x87D700, 0x87D75F, 0x87D787, 0x87D7AF, 0x87D7D7, 0x87D7FF, 0x87FF00, 0x87FF5F,
        0x87FF87, 0x87FFAF, 0x87FFD7, 0x87FFFF, 0xAF0000, 0xAF005F, 0xAF0087, 0xAF00AF,
        0xAF00D7, 0xAF00FF, 0xAF5F00, 0xAF5F5F, 0xAF5F87, 0xAF5FAF, 0xAF5FD7, 0xAF5FFF,
        0xAF8700, 0xAF875F, 0xAF8787, 0xAF87AF, 0xAF87D7, 0xAF87FF, 0xAFAF00, 0xAFAF5F,
        0xAFAF87, 0xAFAFAF, 0xAFAFD7, 0xAFAFFF, 0xAFD700, 0xAFD75F, 0xAFD787, 0xAFD7AF,
        0xAFD7D7, 0xAFD7FF, 0xAFFF00, 0xAFFF5F, 0xAFFF87, 0xAFFFAF, 0xAFFFD7, 0xAFFFFF,
        0xD70000, 0xD7005F, 0xD70087, 0xD700AF, 0xD700D7, 0xD700FF, 0xD75F00, 0xD75F5F,
        0xD75F87, 0xD75FAF, 0xD75FD7, 0xD75FFF, 0xD78700, 0xD7875F, 0xD78787, 0xD787AF,
        0xD787D7, 0xD787FF, 0xD7AF00, 0xD7AF5F, 0xD7AF87, 0xD7AFAF, 0xD7AFD7, 0xD7AFFF,
        0xD7D700, 0xD7D75F, 0xD7D787, 0xD7D7AF, 0xD7D7D7, 0xD7D7FF, 0xD7FF00, 0xD7FF5F,
        0xD7FF87, 0xD7FFAF, 0xD7FFD7, 0xD7FFFF, 0xFF0000, 0xFF005F, 0xFF0087, 0xFF00AF,
        0xFF00D7, 0xFF00FF, 0xFF5F00, 0xFF5F5F, 0xFF5F87, 0xFF5FAF, 0xFF5FD7, 0xFF5FFF,
        0xFF8700, 0xFF875F, 0xFF8787, 0xFF87AF, 0xFF87D7, 0xFF87FF, 0xFFAF00, 0xFFAF5F,
        0xFFAF87, 0xFFAFAF, 0xFFAFD7, 0xFFAFFF, 0xFFD700, 0xFFD75F, 0xFFD787, 0xFFD7AF,
        0xFFD7D7, 0xFFD7FF, 0xFFFF00, 0xFFFF5F, 0xFFFF87, 0xFFFFAF, 0xFFFFD7, 0xFFFFFF
};

// Vivid Red Color Map
static constexpr uint32_t reds[] = {
        0xFF0000,  // Red
        0xFF1A1A,
        0xFF3333,
        0xFF4D4D,
        0xFF6666,
        0xFF8080,
        0xFF9999,
        0xFFB3B3,
        0xFFCCCC,
        0xFFE6E6,
        0xFFFF00,  // Yellow
        0xFFFF1A,
        0xFFFF33,
        0xFFFF4D,
        0xFFFF66,
        0xFFFF80,
        0xFFFF99,
        0xFFFFB3,
        0xFFFFCC,
        0xFFFFE6,
        0xFF0000,  // Repeating for variety
        0xFF1A1A,
        0xFF3333,
        0xFF4D4D,
        0xFF6666,
        0xFF8080,
        0xFF9999,
        0xFFB3B3,
        0xFFCCCC,
        0xFFE6E6
};

static constexpr uint32_t viridis[] = {
        0x3B4992,
        0x406C8E,
        0x487F8F,
        0x55908B,
        0x629A8E,
        0x6FAA94,
        0x7CBBA0,
        0x89CBB0,
        0x96DCC2,
        0xA3EED7,
        0xB1F6E4,
        0xC1F9ED,
        0xD2F6F7,
        0xE5F0F7,
        0xF8EBF5,
        0xFCDAE8,
        0xFCBDBF,
        0xFB9983,
        0xF8775E,
        0xF45B47,
        0xF34538,
        0xF1372D,
        0xEF2625,
        0xED141D,
        0xEA0214,
        0xE50009,
        0xDB0001,
        0xC40000,
        0xAB0000,
        0x900000,
        0x750000
};

static constexpr uint32_t interesting32[] = {
        0xFF0000,  // Red
        0x00FF00,  // Green
        0x0000FF,  // Blue
        0xFFA500,  // Orange
        0xFFFF00,  // Yellow
        0xFF00FF,  // Magenta
        0x00FFFF,  // Cyan
        0x800080,  // Purple
        0xFF4500,  // OrangeRed
        0x8B4513,  // SaddleBrown
        0x2E8B57,  // SeaGreen
        0x00CED1,  // DarkTurquoise
        0x9932CC,  // DarkOrchid
        0xBDB76B,  // DarkKhaki
        0x8A2BE2,  // BlueViolet
        0x20B2AA,  // LightSeaGreen
        0xFF69B4,  // HotPink
        0x00FA9A,  // MediumSpringGreen
        0xADFF2F,  // GreenYellow
        0x00BFFF,  // DeepSkyBlue
        0x228B22,  // ForestGreen
        0xFF6347,  // Tomato
        0x40E0D0,  // Turquoise
        0xFFD700,  // Gold
        0x4682B4,  // SteelBlue
        0x2F4F4F,  // DarkSlateGray
        0x800000,  // Maroon
        0xDC143C,  // Crimson
        0x7B68EE,  // MediumSlateBlue
        0xB8860B,  // DarkGoldenrod
        0x556B2F,  // DarkOliveGreen
        0x8B008B,  // DarkMagenta
        0xFF8C00   // DarkOrange
};

static constexpr uint32_t vibrant32[] = {
        0xFF5733,  // Vermilion
        0xFFD700,  // Gold
        0x4CAF50,  // Emerald
        0x00BCD4,  // Turquoise
        0xE040FB,  // Purple
        0xFF4081,  // Pink
        0x03A9F4,  // Cerulean
        0x9C27B0,  // Violet
        0xFF9800,  // Orange
        0x8BC34A,  // Lime
        0x2196F3,  // Azure
        0x673AB7,  // Grape
        0xFFEB3B,  // Yellow
        0x009688,  // Teal
        0xCDDC39,  // Lime Green
        0xFF5722,  // Deep Orange
        0x009688,  // Green
        0xE91E63,  // Raspberry
        0xFFC107,  // Amber
        0x795548,  // Mocha
        0x00BCD4,  // Cyan
        0x8BC34A,  // Apple Green
        0x2196F3,  // Dodger Blue
        0x673AB7,  // Byzantium
        0xFF4081,  // Fuchsia
        0x4CAF50,  // Malachite
        0xFF5733,  // Orange Red
        0x03A9F4,  // Maya Blue
        0xFFEB3B,  // Maize
        0xE91E63,  // Red Violet
        0x795548   // Redwood
};

static constexpr uint32_t sunset32[] = {
        0xFF4500,  // Orange Red
        0xFF6347,  // Tomato
        0xFF7F50,  // Coral
        0xFFA07A,  // Light Salmon
        0xFFB6C1,  // Light Pink
        0xFFC0CB,  // Pink
        0xFFD700,  // Gold
        0xFFDAB9,  // Peach Puff
        0xFFE4B5,  // Moccasin
        0xFFE4C4,  // Bisque
        0xFFE4E1,  // Misty Rose
        0xFFEBCD,  // Blanched Almond
        0xFFEC8B,  // Light Goldenrod Yellow
        0xFFEFD5,  // Papaya Whip
        0xFFF0F5,  // Lavender Blush
        0xFFF5EE,  // Sea Shell
        0xFFF8DC,  // Cornsilk
        0xFFFACD,  // Lemon Chiffon
        0xFFFAF0,  // Floral White
        0xFFFAFA,  // Snow
        0xFFFF00,  // Yellow
        0xFFFFE0,  // Light Yellow
        0xFFFFF0,  // Ivory
        0xFFFFFF,  // White
        0xFAEBD7,  // Antique White
        0xFDF5E6,  // Old Lace
        0xFFFAF0,  // Floral White
        0xF5FFFA,  // Mint Cream
        0xF8F8FF,  // Ghost White
        0xF0FFF0,  // Honeydew
        0xF0F8FF,  // Alice Blue
        0xF5F5F5   // White Smoke
};

static constexpr uint32_t deep_ocean32[] = {
        0x001F3F,  // Dark Blue
        0x003366,  // Deep Blue
        0x004080,  // Navy Blue
        0x00509E,  // Sapphire
        0x0066CC,  // Royal Blue
        0x0077B5,  // Ocean Blue
        0x0088CC,  // Sky Blue
        0x0099CC,  // Cerulean
        0x00AAD4,  // Deep Sky Blue
        0x00BBE4,  // Steel Blue
        0x00CCE5,  // Light Steel Blue
        0x00D4E6,  // Powder Blue
        0x00E6F3,  // Light Sky Blue
        0x00F0F8,  // Alice Blue
        0x00F5FA,  // Light Cyan
        0x00FAFF,  // Azure
        0x00FFFF,  // Cyan
        0x66CDAA,  // Medium Aquamarine
        0x77DDAA,  // Aquamarine
        0x88EEAA,  // Medium Spring Green
        0x99FFAA,  // Medium Sea Green
        0xAAFFBB,  // Medium Turquoise
        0xBBFFCC,  // Light Sea Green
        0xCCFFDD,  // Dark Sea Green
        0xDDFFEE,  // Sea Green
        0xEEFFFA,  // Mint Cream
        0xF5FFFA,  // Mint Cream
        0xF0FFF0,  // Honeydew
        0xE0FFF0,  // Turquoise
        0xD0FFF0,  // Dark Turquoise
        0xC0FFF0   // Light Cyan
};

static constexpr uint32_t frogs32[] = {
        0x004529,  // Dark Forest Green
        0x005A32,  // Deep Green
        0x007142,  // Jungle Green
        0x008C53,  // Emerald
        0x00A266,  // Shamrock
        0x00B877,  // Forest Green
        0x00CE8A,  // Mint Green
        0x00E49D,  // Pistachio
        0x00F9B0,  // Lime Green
        0x00FFC2,  // Spring Green
        0x66FFCC,  // Medium Spring Green
        0x77FFD3,  // Aquamarine
        0x88FFDB,  // Medium Aquamarine
        0x99FFE2,  // Light Green
        0xAAFFEA,  // Honeydew
        0xBBFFF1,  // Pale Green
        0xCCFFF8,  // Mint Cream
        0xDDFFFE,  // Light Cyan
        0xEEFFFF,  // Cyan
        0xFFFFAA,  // Pale Goldenrod
        0xFFFFBB,  // Light Yellow
        0xFFFFCC,  // Light Goldenrod Yellow
        0xFFFFDD,  // Lemon Chiffon
        0xFFFFEE,  // Light Yellow
        0xFFFFFA,  // Snow
        0xFFF0E6,  // Linen
        0xFFE4C4,  // Bisque
        0xFFDAB9,  // Peachpuff
        0xFFC0CB,  // Pink
        0xFFB6C1,  // Light Pink
        0xFFA07A,  // Light Salmon
        0xFF8C69   // Salmon
};

using ColorMap = std::unordered_map<std::string, std::pair<const uint32_t*, std::size_t>>;

static const ColorMap colorMap = {
        { "Accent",        { Accent,        sizeof(Accent) } },
        { "Paired",        { Paired,        sizeof(Paired) } },
        { "Paired",        { Paired,        sizeof(Paired) } },
        { "Set1",          { Set1,          sizeof(Set1) } },
        { "crayola16",     { crayola16,     sizeof(crayola16) } },
        { "darknet",       { darknet,       sizeof(darknet) } },
        { "deep_ocean32",  { deep_ocean32,  sizeof(deep_ocean32) } },
        { "fluorescent",   { fluorescent,   sizeof(fluorescent) } },
        { "frogs32",       { frogs32,       sizeof(frogs32), } },
        { "interesting32", { interesting32, sizeof(interesting32) } },
        { "off_the_path",  { off_the_path,  sizeof(off_the_path) } },
        { "Paired",        { Paired,        sizeof(Paired) } },
        { "reds",          { reds,          sizeof(reds) } },
        { "Set1",          { Set1,          sizeof(Set1) } },
        { "sunset32",      { sunset32,      sizeof(sunset32), } },
        { "tab10",         { tab10,         sizeof(tab10) } },
        { "tab20",         { tab20,         sizeof(tab20) } },
        { "tab20c",        { tab20c,        sizeof(tab20c) } },
        { "vga",           { vga,           sizeof(vga) } },
        { "vibrant32",     { vibrant32,     sizeof(vibrant32) } },
        { "viridis",       { viridis,       sizeof(viridis) } },
        { "vivid",         { vivid,         sizeof(vivid) } },
};

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
    const auto* table = it_->it->second.first;
    const auto sz = it_->it->second.second;

    index *= 2654435761U;

    return table[index % sz];
}

}   // px
