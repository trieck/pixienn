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
    "OrRd", "PuBu", "PuBuGn", "PuRd", "Purples", "RdPu", "Reds",
    "YlGn", "YlGnBu", "YlOrBr", "YlOrRd", "afmhot", "autumn", "binary",
    "bone", "cividis", "cool", "coolwarm", "copper", "gist_earth",
    "gist_gray", "gist_heat", "gist_ncar", "gist_rainbow", "gist_yarg",
    "gnuplot", "gnuplot2", "gray", "hot", "hsv", "inferno", "jet",
    "magma", "nipy_spectral", "ocean", "pink", "plasma", "prism",
    "rainbow", "seismic", "spring", "summer", "tab20b", "tab20c",
    "terrain", "turbo", "twilight", "twilight_shifted", "viridis", "winter",
)


def rgb(value):
    channels = np.asarray(value[:3]) * 255.0
    return f"0x{int(channels[0] + 0.5):02x}{int(channels[1] + 0.5):02x}{int(channels[2] + 0.5):02x}"


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


if __name__ == "__main__":
    main()
