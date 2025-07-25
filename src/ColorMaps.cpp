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
        0x000000, 0x800000, 0x008000, 0x808000, 0x000080, 0x800080, 0x008080, 0xc0c0c0,
        0x808080, 0xff0000, 0x00ff00, 0xffff00, 0x0000ff, 0xff00ff, 0x00ffff, 0xffffff,
        0x000000, 0x00005f, 0x000087, 0x0000af, 0x0000d7, 0x0000ff, 0x005f00, 0x005f5f,
        0x005f87, 0x005faf, 0x005fd7, 0x005fff, 0x008700, 0x00875f, 0x008787, 0x0087af,
        0x0087d7, 0x0087ff, 0x00af00, 0x00af5f, 0x00af87, 0x00afaf, 0x00afd7, 0x00afff,
        0x00d700, 0x00d75f, 0x00d787, 0x00d7af, 0x00d7d7, 0x00d7ff, 0x00ff00, 0x00ff5f,
        0x00ff87, 0x00ffaf, 0x00ffd7, 0x00ffff, 0x5f0000, 0x5f005f, 0x5f0087, 0x5f00af,
        0x5f00d7, 0x5f00ff, 0x5f5f00, 0x5f5f5f, 0x5f5f87, 0x5f5faf, 0x5f5fd7, 0x5f5fff,
        0x5f8700, 0x5f875f, 0x5f8787, 0x5f87af, 0x5f87d7, 0x5f87ff, 0x5faf00, 0x5faf5f,
        0x5faf87, 0x5fafaf, 0x5fafd7, 0x5fafff, 0x5fd700, 0x5fd75f, 0x5fd787, 0x5fd7af,
        0x5fd7d7, 0x5fd7ff, 0x5fff00, 0x5fff5f, 0x5fff87, 0x5fffaf, 0x5fffd7, 0x5fffff,
        0x870000, 0x87005f, 0x870087, 0x8700af, 0x8700d7, 0x8700ff, 0x875f00, 0x875f5f,
        0x875f87, 0x875faf, 0x875fd7, 0x875fff, 0x878700, 0x87875f, 0x878787, 0x8787af,
        0x8787d7, 0x8787ff, 0x87af00, 0x87af5f, 0x87af87, 0x87afaf, 0x87afd7, 0x87afff,
        0x87d700, 0x87d75f, 0x87d787, 0x87d7af, 0x87d7d7, 0x87d7ff, 0x87ff00, 0x87ff5f,
        0x87ff87, 0x87ffaf, 0x87ffd7, 0x87ffff, 0xaf0000, 0xaf005f, 0xaf0087, 0xaf00af,
        0xaf00d7, 0xaf00ff, 0xaf5f00, 0xaf5f5f, 0xaf5f87, 0xaf5faf, 0xaf5fd7, 0xaf5fff,
        0xaf8700, 0xaf875f, 0xaf8787, 0xaf87af, 0xaf87d7, 0xaf87ff, 0xafaf00, 0xafaf5f,
        0xafaf87, 0xafafaf, 0xafafd7, 0xafafff, 0xafd700, 0xafd75f, 0xafd787, 0xafd7af,
        0xafd7d7, 0xafd7ff, 0xafff00, 0xafff5f, 0xafff87, 0xafffaf, 0xafffd7, 0xafffff,
        0xd70000, 0xd7005f, 0xd70087, 0xd700af, 0xd700d7, 0xd700ff, 0xd75f00, 0xd75f5f,
        0xd75f87, 0xd75faf, 0xd75fd7, 0xd75fff, 0xd78700, 0xd7875f, 0xd78787, 0xd787af,
        0xd787d7, 0xd787ff, 0xd7af00, 0xd7af5f, 0xd7af87, 0xd7afaf, 0xd7afd7, 0xd7afff,
        0xd7d700, 0xd7d75f, 0xd7d787, 0xd7d7af, 0xd7d7d7, 0xd7d7ff, 0xd7ff00, 0xd7ff5f,
        0xd7ff87, 0xd7ffaf, 0xd7ffd7, 0xd7ffff, 0xff0000, 0xff005f, 0xff0087, 0xff00af,
        0xff00d7, 0xff00ff, 0xff5f00, 0xff5f5f, 0xff5f87, 0xff5faf, 0xff5fd7, 0xff5fff,
        0xff8700, 0xff875f, 0xff8787, 0xff87af, 0xff87d7, 0xff87ff, 0xffaf00, 0xffaf5f,
        0xffaf87, 0xffafaf, 0xffafd7, 0xffafff, 0xffd700, 0xffd75f, 0xffd787, 0xffd7af,
        0xffd7d7, 0xffd7ff, 0xffff00, 0xffff5f, 0xffff87, 0xffffaf, 0xffffd7, 0xffffff,
        0x080808, 0x121212, 0x1c1c1c, 0x262626, 0x303030, 0x3a3a3a, 0x444444, 0x4e4e4e,
        0x585858, 0x626262, 0x6c6c6c, 0x767676, 0x808080, 0x8a8a8a, 0x949494, 0x9e9e9e,
        0xa8a8a8, 0xb2b2b2, 0xbcbcbc, 0xc6c6c6, 0xd0d0d0, 0xdadada, 0xe4e4e4, 0xeeeeee
};

static constexpr uint32_t reds[] = {
        0xFF0000,
        0xFF1A1A,
        0xFF3333,
        0xFF4D4D,
        0xFF6666,
        0xFF8080,
        0xFF9999,
        0xFFB3B3,
        0xFFCCCC,
        0xFFE6E6,
        0xFFFF00,
        0xFFFF1A,
        0xFFFF33,
        0xFFFF4D,
        0xFFFF66,
        0xFFFF80,
        0xFFFF99,
        0xFFFFB3,
        0xFFFFCC,
        0xFFFFE6,
        0xFF0000,
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

constexpr uint32_t plasma[] = {
        0x0d0887,
        0x100788,
        0x130789,
        0x16078a,
        0x19068c,
        0x1b068d,
        0x1d068e,
        0x20068f,
        0x220690,
        0x240691,
        0x260591,
        0x280592,
        0x2a0593,
        0x2c0594,
        0x2e0595,
        0x2f0596,
        0x310597,
        0x330597,
        0x350498,
        0x370499,
        0x38049a,
        0x3a049a,
        0x3c049b,
        0x3e049c,
        0x3f049c,
        0x41049d,
        0x43039e,
        0x44039e,
        0x46039f,
        0x48039f,
        0x4903a0,
        0x4b03a1,
        0x4c02a1,
        0x4e02a2,
        0x5002a2,
        0x5102a3,
        0x5302a3,
        0x5502a4,
        0x5601a4,
        0x5801a4,
        0x5901a5,
        0x5b01a5,
        0x5c01a6,
        0x5e01a6,
        0x6001a6,
        0x6100a7,
        0x6300a7,
        0x6400a7,
        0x6600a7,
        0x6700a8,
        0x6900a8,
        0x6a00a8,
        0x6c00a8,
        0x6e00a8,
        0x6f00a8,
        0x7100a8,
        0x7201a8,
        0x7401a8,
        0x7501a8,
        0x7701a8,
        0x7801a8,
        0x7a02a8,
        0x7b02a8,
        0x7d03a8,
        0x7e03a8,
        0x8004a8,
        0x8104a7,
        0x8305a7,
        0x8405a7,
        0x8606a6,
        0x8707a6,
        0x8808a6,
        0x8a09a5,
        0x8b0aa5,
        0x8d0ba5,
        0x8e0ca4,
        0x8f0da4,
        0x910ea3,
        0x920fa3,
        0x9410a2,
        0x9511a1,
        0x9613a1,
        0x9814a0,
        0x99159f,
        0x9a169f,
        0x9c179e,
        0x9d189d,
        0x9e199d,
        0xa01a9c,
        0xa11b9b,
        0xa21d9a,
        0xa31e9a,
        0xa51f99,
        0xa62098,
        0xa72197,
        0xa82296,
        0xaa2395,
        0xab2494,
        0xac2694,
        0xad2793,
        0xae2892,
        0xb02991,
        0xb12a90,
        0xb22b8f,
        0xb32c8e,
        0xb42e8d,
        0xb52f8c,
        0xb6308b,
        0xb7318a,
        0xb83289,
        0xba3388,
        0xbb3488,
        0xbc3587,
        0xbd3786,
        0xbe3885,
        0xbf3984,
        0xc03a83,
        0xc13b82,
        0xc23c81,
        0xc33d80,
        0xc43e7f,
        0xc5407e,
        0xc6417d,
        0xc7427c,
        0xc8437b,
        0xc9447a,
        0xca457a,
        0xcb4679,
        0xcc4778,
        0xcc4977,
        0xcd4a76,
        0xce4b75,
        0xcf4c74,
        0xd04d73,
        0xd14e72,
        0xd24f71,
        0xd35171,
        0xd45270,
        0xd5536f,
        0xd5546e,
        0xd6556d,
        0xd7566c,
        0xd8576b,
        0xd9586a,
        0xda5a6a,
        0xda5b69,
        0xdb5c68,
        0xdc5d67,
        0xdd5e66,
        0xde5f65,
        0xde6164,
        0xdf6263,
        0xe06363,
        0xe16462,
        0xe26561,
        0xe26660,
        0xe3685f,
        0xe4695e,
        0xe56a5d,
        0xe56b5d,
        0xe66c5c,
        0xe76e5b,
        0xe76f5a,
        0xe87059,
        0xe97158,
        0xe97257,
        0xea7457,
        0xeb7556,
        0xeb7655,
        0xec7754,
        0xed7953,
        0xed7a52,
        0xee7b51,
        0xef7c51,
        0xef7e50,
        0xf07f4f,
        0xf0804e,
        0xf1814d,
        0xf1834c,
        0xf2844b,
        0xf3854b,
        0xf3874a,
        0xf48849,
        0xf48948,
        0xf58b47,
        0xf58c46,
        0xf68d45,
        0xf68f44,
        0xf79044,
        0xf79143,
        0xf79342,
        0xf89441,
        0xf89540,
        0xf9973f,
        0xf9983e,
        0xf99a3e,
        0xfa9b3d,
        0xfa9c3c,
        0xfa9e3b,
        0xfb9f3a,
        0xfba139,
        0xfba238,
        0xfca338,
        0xfca537,
        0xfca636,
        0xfca835,
        0xfca934,
        0xfdab33,
        0xfdac33,
        0xfdae32,
        0xfdaf31,
        0xfdb130,
        0xfdb22f,
        0xfdb42f,
        0xfdb52e,
        0xfeb72d,
        0xfeb82c,
        0xfeba2c,
        0xfebb2b,
        0xfebd2a,
        0xfebe2a,
        0xfec029,
        0xfdc229,
        0xfdc328,
        0xfdc527,
        0xfdc627,
        0xfdc827,
        0xfdca26,
        0xfdcb26,
        0xfccd25,
        0xfcce25,
        0xfcd025,
        0xfcd225,
        0xfbd324,
        0xfbd524,
        0xfbd724,
        0xfad824,
        0xfada24,
        0xf9dc24,
        0xf9dd25,
        0xf8df25,
        0xf8e125,
        0xf7e225,
        0xf7e425,
        0xf6e626,
        0xf6e826,
        0xf5e926,
        0xf5eb27,
        0xf4ed27,
        0xf3ee27,
        0xf3f027,
        0xf2f227,
        0xf1f426,
        0xf1f525,
        0xf0f724,
        0xf0f921
};

constexpr uint32_t tailwind[] = {
        0xffbe0b, // Orange
        0xfb5607, // Red Orange
        0xff006e, // Pink
        0x8338ec, // Purple
        0x3a86ff  // Blue
};

constexpr uint32_t sunset[] = {
        0x001219, // Midnight Blue
        0x005f73, // Teal Blue
        0x0a9396, // Turquoise Blue
        0x94d2bd, // Pastel Green
        0xe9d8a6, // Light Tan
        0xee9b00, // Sunset Orange
        0xca6702, // Rust Orange
        0xbb3e03, // Burnt Orange
        0xae2012, // Reddish Brown
        0x9b2226  // Brick Red
};

constexpr uint32_t vibrant[] = {
        0xff595e, // Reddish
        0xffca3a, // Amber
        0x8ac926, // Green
        0x1982c4, // Blue
        0x6a4c93  // Purple
};

constexpr uint32_t retro[16] = {
        0x000000, // black
        0x800000, // dark red
        0x008000, // dark green
        0x808000, // dark yellow (olive)
        0x000080, // dark blue
        0x800080, // dark magenta
        0x008080, // dark cyan
        0xc0c0c0, // light gray

        0x808080, // dark gray
        0xff0000, // bright red
        0x00ff00, // bright green
        0xffff00, // bright yellow
        0x0000ff, // bright blue
        0xff00ff, // bright magenta
        0x00ffff, // bright cyan
        0xffffff  // white
};

#define TABLE_SIZE(tbl) (sizeof(tbl)/sizeof(uint32_t))

using ColorMap = std::unordered_map<std::string, std::pair<const uint32_t*, std::size_t>>;

static const ColorMap colorMap = {
        { "Accent",        { Accent,        TABLE_SIZE(Accent) } },
        { "Paired",        { Paired,        TABLE_SIZE(Paired) } },
        { "Set1",          { Set1,          TABLE_SIZE(Set1) } },
        { "crayola16",     { crayola16,     TABLE_SIZE(crayola16) } },
        { "darknet",       { darknet,       TABLE_SIZE(darknet) } },
        { "deep_ocean32",  { deep_ocean32,  TABLE_SIZE(deep_ocean32) } },
        { "fluorescent",   { fluorescent,   TABLE_SIZE(fluorescent) } },
        { "frogs32",       { frogs32,       TABLE_SIZE(frogs32), } },
        { "interesting32", { interesting32, TABLE_SIZE(interesting32) } },
        { "off_the_path",  { off_the_path,  TABLE_SIZE(off_the_path) } },
        { "plasma",        { plasma,        TABLE_SIZE(plasma) } },
        { "reds",          { reds,          TABLE_SIZE(reds) } },
        { "retro",         { retro,         TABLE_SIZE(retro) } },
        { "sunset",        { sunset,        TABLE_SIZE(sunset) } },
        { "sunset32",      { sunset32,      TABLE_SIZE(sunset32), } },
        { "tab10",         { tab10,         TABLE_SIZE(tab10) } },
        { "tab20",         { tab20,         TABLE_SIZE(tab20) } },
        { "tab20c",        { tab20c,        TABLE_SIZE(tab20c) } },
        { "tailwind",      { tailwind,      TABLE_SIZE(tailwind) } },
        { "vga",           { vga,           TABLE_SIZE(vga) } },
        { "vibrant",       { vibrant,       TABLE_SIZE(vibrant) } },
        { "vibrant32",     { vibrant32,     TABLE_SIZE(vibrant32) } },
        { "viridis",       { viridis,       TABLE_SIZE(viridis) } },
        { "vivid",         { vivid,         TABLE_SIZE(vivid) } }
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

    auto color = table[index % sz];

    auto gamma = 1.2f; // gamma correction
    auto r = static_cast<float>((color >> 16) & 0xFF) / 255.0f;
    auto g = static_cast<float>((color >> 8) & 0xFF) / 255.0f;
    auto b = static_cast<float>(color & 0xFF) / 255.0f;

    r = std::pow(r, gamma);
    g = std::pow(g, gamma);
    b = std::pow(b, gamma);

    color = ((static_cast<uint32_t>(r * 255.0f) & 0xFF) << 16) |
            ((static_cast<uint32_t>(g * 255.0f) & 0xFF) << 8) |
            (static_cast<uint32_t>(b * 255.0f) & 0xFF);

    return color;
}

std::vector<std::string> ColorMaps::maps()
{
    std::vector<std::string> colors;

    for (auto const& entry: colorMap) {
        colors.emplace_back(entry.first);
    }

    std::sort(colors.begin(), colors.end());

    return colors;
}

}   // px
