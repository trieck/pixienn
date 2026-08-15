#!/usr/bin/env python3
"""Generate the C++ lookup tables used by ColorMaps from Matplotlib."""

from pathlib import Path

import matplotlib
import numpy as np


QUALITATIVE = (
    "Accent", "Dark2", "Paired", "Pastel1", "Pastel2",
    "Set1", "Set2", "Set3", "tab10", "tab20",
)

CONTINUOUS = (
    "Blues", "BuGn", "BuPu", "GnBu", "Greens", "Greys", "Oranges",
    "OrRd", "PuBu", "PuBuGn", "PuRd", "Purples", "RdPu", "RdYlGn", "Reds",
    "YlGn", "YlGnBu", "YlOrBr", "YlOrRd", "afmhot", "autumn", "binary",
    "bone", "cividis", "cool", "coolwarm", "copper", "gist_earth",
    "gist_gray", "gist_heat", "gist_ncar", "gist_rainbow", "gist_yarg",
    "gnuplot", "gnuplot2", "gray", "hot", "hsv", "inferno", "jet",
    "magma", "nipy_spectral", "ocean", "pink", "plasma", "prism",
    "rainbow", "seismic", "spring", "summer", "tab20b", "tab20c",
    "terrain", "turbo", "twilight", "twilight_shifted", "viridis", "winter",
)

SEQUENTIAL = {
    "Blues", "BuGn", "BuPu", "GnBu", "Greens", "Greys", "OrRd", "Oranges",
    "PuBu", "PuBuGn", "PuRd", "Purples", "RdPu", "Reds", "YlGn", "YlGnBu",
    "YlOrBr", "YlOrRd", "afmhot", "autumn", "binary", "bone", "cividis",
    "copper", "gist_gray", "gist_yarg", "gray", "hot", "inferno", "magma",
    "pink", "plasma", "viridis", "winter",
}

DIVERGING = {"coolwarm", "RdYlGn", "seismic"}


def rgb(value):
    channels = np.asarray(value[:3]) * 255.0
    return f"0x{int(channels[0] + 0.5):02x}{int(channels[1] + 0.5):02x}{int(channels[2] + 0.5):02x}"


def hex_rgb(value):
    return "#" + rgb(value)[2:]


def swatch(color):
    return f'<span style="display:inline-block;width:3em;height:1.2em;background:{color};border:1px solid #999;vertical-align:middle"></span>'


def write_markdown():
    output = Path(__file__).resolve().parents[1] / "docs" / "colormaps.md"
    lines = [
        "# PixieNN colormaps",
        "",
        "These maps are generated from Matplotlib and are available through `--color-map`.",
        "",
        "Confidence coloring uses a continuous map:",
        "",
        "```bash",
        "pixienn --color-map=viridis --color-by-confidence model.yml image.jpg",
        "pixienn --color-map=viridis --color-by-confidence --stretch-confidence model.yml image.jpg",
        "```",
        "",
        "With `--stretch-confidence`, the lowest and highest confidence values in the image map to 0.0 and 1.0.",
        "",
        "## Continuous maps",
        "",
        "`Continuous` means the map accepts a normalized scalar value. `Sequential / ordered` means it is designed to communicate low-to-high magnitude and is appropriate for confidence. This is a semantic visual-design property, not a claim that every RGB channel is mathematically monotonic.",
        "",
    ]

    positions = np.linspace(0.0, 1.0, 11)
    for name in CONTINUOUS:
        cmap = matplotlib.colormaps[name]
        ordered = name in SEQUENTIAL
        family = "diverging" if name in DIVERGING else "non-sequential"
        recommendation = "recommended" if ordered else (
            f"usable with caution ({family}; not strictly sequential)"
            if name in DIVERGING else f"not recommended ({family})"
        )
        lines.extend([
            f"### `{name}`", "",
            f"**Continuous:** yes  \n**Sequential / ordered:** {'yes' if ordered else 'no'}  \n**Confidence use:** {recommendation}",
            "", "| value | color | hex |", "|---:|:---:|:---|"
        ])
        for position in positions:
            color = hex_rgb(cmap(float(position)))
            lines.append(f"| {position:.1f} | {swatch(color)} | `{color}` |")
        lines.append("")

    lines.extend(["## Qualitative maps", "", "These maps assign discrete colors to classes and are not continuous or ordered for confidence.", ""])
    for name in QUALITATIVE:
        lines.extend([
            f"### `{name}`", "",
            "**Continuous:** no  \n**Sequential / ordered:** no  \n**Confidence use:** not recommended",
            "", "| index | color | hex |", "|---:|:---:|:---|"
        ])
        for index, value in enumerate(matplotlib.colormaps[name].colors):
            color = hex_rgb(value)
            lines.append(f"| {index} | {swatch(color)} | `{color}` |")
        lines.append("")

    output.write_text("\n".join(lines) + "\n")


def main():
    output = Path(__file__).resolve().parents[1] / "include" / "MatplotlibColorMaps.h"
    lines = [
        "// Generated from Matplotlib %s; do not edit by hand." % matplotlib.__version__,
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace px::matplotlib {",
        "",
    ]

    for name in QUALITATIVE:
        values = matplotlib.colormaps[name].colors
        lines.append(f"inline constexpr uint32_t {name}[] = {{")
        lines.extend(f"        {rgb(value)}," for value in values)
        lines.append("};")
        lines.append("")

    for name in CONTINUOUS:
        values = matplotlib.colormaps[name](np.linspace(0.0, 1.0, 256))
        lines.append(f"inline constexpr uint32_t {name}[] = {{")
        lines.extend(f"        {rgb(value)}," for value in values)
        lines.append("};")
        lines.append("")

    lines.append("} // namespace px::matplotlib")
    output.write_text("\n".join(lines) + "\n")
    write_markdown()


if __name__ == "__main__":
    main()
