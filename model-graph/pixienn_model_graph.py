#!/usr/bin/env python3
"""Render a PixieNN YAML model as a Netron-style interactive graph."""

from __future__ import annotations

import argparse
import colorsys
import html
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import yaml


DISPLAY_NAMES = {
    "positional-encoding": "Positional Encoding",
    "layernorm": "LayerNorm",
    "self-attention": "Self-Attention",
}


def palette_color(index: int) -> str:
    """Return a deterministic, distinct color for a layer type."""
    hue = (index * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.58, 0.78)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def apply_type_colors(layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    types = list(dict.fromkeys(layer["type"] for layer in layers))
    colors = {layer_type: palette_color(index) for index, layer_type in enumerate(types)}
    for layer in layers:
        layer["color"] = colors[layer["type"]]
    return layers


def shape_text(shape: tuple[int | str, int | str, int | str, int | str]) -> str:
    return " × ".join(str(value) for value in shape)


def resolve(index: int, reference: int) -> int:
    return index + reference if reference < 0 else reference


def references(index: int, layer: dict[str, Any]) -> list[int]:
    if layer.get("type") == "route":
        return [resolve(index, int(value)) for value in layer.get("layers", [])]
    if layer.get("type") == "shortcut":
        return [resolve(index, int(layer.get("from", -1)))]
    return []


def conv_shape(shape: tuple[int | str, int | str, int | str, int | str], layer: dict[str, Any]):
    batch, _, height, width = shape
    filters = int(layer.get("filters", shape[1]))
    kernel = int(layer.get("kernel", 1))
    stride = int(layer.get("stride", 1))
    padding = kernel // 2 if layer.get("pad", False) else 0

    def output(size: int | str) -> int | str:
        if not isinstance(size, int):
            return "?"
        return max(1, math.floor((size + 2 * padding - kernel) / stride) + 1)

    return batch, filters, output(height), output(width)


def infer_layers(model: dict[str, Any]) -> list[dict[str, Any]]:
    definitions = model["layers"]
    input_shape = (int(model.get("batch", 1)), int(model.get("channels", 3)),
                   int(model.get("height", 0)), int(model.get("width", 0)))
    outputs = [input_shape]
    result = [{"index": -1, "type": "input", "name": "Input",
               "input_shape": shape_text(input_shape),
               "input_shapes": [shape_text(input_shape)],
               "shape": shape_text(input_shape), "references": [], "params": {}}]

    for index, raw in enumerate(definitions):
        layer = dict(raw)
        layer_type = str(layer.get("type", "unknown")).lower()
        input_shape = outputs[-1]
        refs = references(index, layer)
        source_shapes = [outputs[ref + 1] for ref in refs if 0 <= ref < len(outputs) - 1]
        shape = input_shape

        if layer_type == "route" and source_shapes:
            first = source_shapes[0]
            channels = sum(int(source[1]) for source in source_shapes if isinstance(source[1], int))
            shape = first[0], channels, first[2], first[3]
        elif layer_type == "shortcut" and source_shapes:
            shape = source_shapes[0]
        elif layer_type == "conv":
            shape = conv_shape(outputs[-1], layer)
        elif layer_type in {"upsample", "upsample2d"}:
            stride = int(layer.get("stride", 2))
            shape = (outputs[-1][0], outputs[-1][1],
                     outputs[-1][2] * stride if isinstance(outputs[-1][2], int) else "?",
                     outputs[-1][3] * stride if isinstance(outputs[-1][3], int) else "?")
        elif layer_type in {"maxpool", "avgpool", "pool"}:
            size = int(layer.get("size", layer.get("kernel", 2)))
            shape = conv_shape(outputs[-1], {"filters": outputs[-1][1], "kernel": size,
                                             "stride": int(layer.get("stride", size)),
                                             "pad": layer.get("pad", False)})

        if layer_type == "route" and source_shapes:
            first = source_shapes[0]
            input_shape_text = f"{first[0]} × C × {first[2]} × {first[3]}"
            input_shapes = list(dict.fromkeys(shape_text(source) for source in source_shapes))
        elif layer_type == "shortcut" and source_shapes:
            input_shape_text = (
                f"previous: {shape_text(input_shape)} + "
                f"#{refs[0]}: {shape_text(source_shapes[0])}"
            )
            input_shapes = [shape_text(input_shape)]
        else:
            input_shape_text = shape_text(input_shape)
            input_shapes = [input_shape_text]

        outputs.append(shape)
        result.append({"index": index, "type": layer_type,
                       "name": DISPLAY_NAMES.get(layer_type, layer_type.replace("_", " ").title()),
                       "input_shape": input_shape_text,
                       "input_shapes": input_shapes,
                       "shape": shape_text(shape), "references": refs,
                       "params": {key: value for key, value in layer.items() if key != "type"}})
    return apply_type_colors(result)


def read_model(path: Path) -> tuple[str, list[dict[str, Any]]]:
    document = yaml.safe_load(path.read_text()) or {}
    model = document.get("model", document)
    if not isinstance(model, dict) or not isinstance(model.get("layers"), list):
        raise ValueError(f"{path} does not contain a model.layers sequence")
    return path.name, infer_layers(model)


def dot_label(layer: dict[str, Any], color: str) -> str:
    index = layer["index"]
    name = html.escape(layer["name"], quote=False)
    input_shapes = layer.get("input_shapes", [layer["input_shape"]])
    input_rows = "<BR ALIGN=\"LEFT\"/>".join(
        f"input: {html.escape(value, quote=False)}" for value in input_shapes
    )
    shape = html.escape(layer["shape"], quote=False)
    inputs = ", ".join(f"#{value}" for value in layer["references"]) or "sequential input"
    return (f'<<TABLE STYLE="ROUNDED" BORDER="2" COLOR="{color}" CELLBORDER="0" '
            f'CELLSPACING="0" CELLPADDING="7" BGCOLOR="#ffffff">'
            f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9" COLOR="{color}">'
            f'<B>#{index}  {name}</B></FONT></TD></TR>'
            f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="7" COLOR="#475569"><B>{input_rows}</B></FONT></TD></TR>'
            f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="7" COLOR="#334155"><B>output: {shape}</B></FONT></TD></TR>'
            f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="6" COLOR="#64748b">'
            f'inputs: {html.escape(inputs, quote=False)}</FONT></TD></TR></TABLE>>')


def graph_edges(layers: list[dict[str, Any]]) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for layer in layers:
        index = layer["index"]
        if index < 0:
            continue
        if layer["type"] == "route":
            sources = layer["references"]
        elif layer["type"] == "shortcut":
            sources = [index - 1, *layer["references"]]
        else:
            sources = [index - 1] if index > 0 else [-1]
        for source in sources:
            if (source, index) not in edges:
                edges.append((source, index))
    return edges


def render_svg(layers: list[dict[str, Any]]) -> str:
    lines = [
        "digraph PixieNN {",
        'graph [rankdir=TB, bgcolor="transparent", pad="0.25", nodesep="0.35", ranksep="0.55", splines=spline, overlap=false, outputorder=edgesfirst];',
        'node [shape=plain, fontname="Arial"];',
        'edge [color="#64748b", arrowsize="0.65", penwidth="1.2"];',
    ]
    for layer in layers:
        node_id = "input" if layer["index"] < 0 else f"n{layer['index']}"
        color = layer["color"]
        lines.append(f'{node_id} [id="{node_id}", label={dot_label(layer, color)}];')
    for source, target in graph_edges(layers):
        source_id = "input" if source < 0 else f"n{source}"
        lines.append(f"{source_id} -> n{target};")
    lines.append("}")
    completed = subprocess.run(["dot", "-Tsvg"], input="\n".join(lines), text=True,
                               capture_output=True, check=True)
    svg = completed.stdout.replace('<svg ', '<svg id="model-svg" ', 1)
    svg = svg.replace('<svg id="model-svg" ', '<svg id="model-svg" ', 1)
    svg = svg.replace('<g id="input" class="node">',
                      '<g id="input" class="node" filter="url(#node-shadow)">')
    svg = svg.replace('class="node">', 'class="node" filter="url(#node-shadow)">')
    defs = ('<defs><filter id="node-shadow" x="-30%" y="-30%" width="180%" height="210%">'
            '<feDropShadow dx="0" dy="2" stdDeviation="1.2" flood-color="#ffffff" flood-opacity=".75"/>'
            '<feDropShadow dx="0" dy="7" stdDeviation="4" flood-color="#0f172a" flood-opacity=".52"/>'
            '</filter></defs>')
    return svg.replace('xmlns:xlink="http://www.w3.org/1999/xlink">',
                       'xmlns:xlink="http://www.w3.org/1999/xlink">' + defs, 1)


def render_html(model_name: str, layers: list[dict[str, Any]]) -> str:
    payload = json.dumps(layers, separators=(",", ":"))
    title = html.escape(model_name)
    svg = render_svg(layers)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PixieNN model graph — {title}</title>
<style>
:root {{ color-scheme: dark; --bg:#0b1020; --panel:#121a2e; --ink:#e5edf8; --muted:#91a0b9; --line:#33415f; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:radial-gradient(circle at 20% 0%,#18284b 0,#0b1020 45%); color:var(--ink); font:14px/1.4 system-ui,sans-serif; }}
header {{ position:sticky; top:0; z-index:5; padding:18px 24px 14px; background:rgba(11,16,32,.94); border-bottom:1px solid var(--line); backdrop-filter:blur(12px); }}
h1 {{ margin:0 0 10px; font-size:20px; }} .toolbar {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; }}
input {{ width:min(430px,80vw); padding:9px 12px; color:var(--ink); background:var(--panel); border:1px solid var(--line); border-radius:7px; outline:none; }}
button {{ padding:8px 11px; color:var(--ink); background:var(--panel); border:1px solid var(--line); border-radius:7px; cursor:pointer; }} button:hover {{ background:#17223b; }}
.count {{ color:var(--muted); }} main {{ display:grid; grid-template-columns:minmax(700px,1fr) 340px; gap:18px; padding:20px 24px 40px; }}
.graph {{ min-height:720px; overflow:auto; padding:12px; background:#f8fafc; border:1px solid var(--line); border-radius:10px; cursor:grab; }} .graph:active {{ cursor:grabbing; }}
#model-svg {{ display:block; min-width:760px; width:100%; height:auto; transform-origin:top left; }}
#model-svg g.node {{ cursor:pointer; transition:opacity .12s; }} #model-svg g.node:hover {{ filter:drop-shadow(0 0 7px #38bdf899); }}
#model-svg g.node polygon {{ stroke:#64748b; stroke-width:1.1; }}
#model-svg g.node.selected polygon {{ stroke:#0ea5e9; stroke-width:2.2; }}
#model-svg g.node.dim {{ opacity:.12; }}
#model-svg g.edge path {{ stroke:#64748b; stroke-width:1.7; }} #model-svg g.edge polygon {{ fill:#64748b; stroke:#64748b; }}
.side {{ position:sticky; top:110px; align-self:start; padding:16px; background:rgba(18,26,46,.9); border:1px solid var(--line); border-radius:10px; min-height:220px; }}
.side h2 {{ margin:0 0 12px; font-size:16px; }} .side dl {{ display:grid; grid-template-columns:95px 1fr; gap:7px 10px; margin:0; }} .side dt {{ color:var(--muted); }}
.side dd {{ margin:0; overflow-wrap:anywhere; font-family:ui-monospace,monospace; font-size:12px; }} .param {{ margin-top:14px; padding-top:12px; border-top:1px solid var(--line); color:#cbd5e1; white-space:pre-wrap; font:12px/1.5 ui-monospace,monospace; }}
@media (max-width:900px) {{ main {{ grid-template-columns:1fr; }} .side {{ position:relative; top:auto; }} }}
</style></head><body>
<header><h1>PixieNN model graph — {title}</h1><div class="toolbar">
<input id="search" type="search" placeholder="Search layer number, type, or shape…" aria-label="Search layers">
<button id="reset" type="button">Show all</button><button id="zoom-out" type="button">−</button><button id="zoom-reset" type="button">100%</button><button id="zoom-in" type="button">+</button><span id="count" class="count"></span>
</div></header>
<main><section id="graph" class="graph" aria-label="Model topology graph">{svg}</section>
<aside id="detail" class="side"><h2>Layer details</h2><p class="count">Select a node to inspect its inputs, output shape, and YAML properties.</p></aside></main>
<script>
const layers = {payload}; const detail = document.getElementById('detail'); const search = document.getElementById('search'); const count = document.getElementById('count'); const svg = document.getElementById('model-svg'); let selected = null; let zoom = 1;
function matches(layer, query) {{ return !query || [layer.index, layer.type, layer.name, layer.shape, JSON.stringify(layer.params)].join(' ').toLowerCase().includes(query); }}
function escapeHtml(value) {{ return value.replace(/[&<>"']/g, character => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[character])); }}
function showDetail(layer) {{ selected = layer.index; document.querySelectorAll('#model-svg g.node').forEach(node => node.classList.toggle('selected', node.id === (layer.index < 0 ? 'input' : 'n' + layer.index))); const refs = layer.references.length ? layer.references.map(value => '#' + value).join(', ') : 'sequential input'; detail.innerHTML = `<h2>#${{layer.index}} ${{layer.name}}</h2><dl><dt>Type</dt><dd>${{layer.type}}</dd><dt>Output</dt><dd>${{layer.shape}}</dd><dt>Inputs</dt><dd>${{refs}}</dd></dl><div class="param">${{escapeHtml(JSON.stringify(layer.params, null, 2))}}</div>`; }}
function draw() {{ const query = search.value.trim().toLowerCase(); let visible = 0; layers.forEach(layer => {{ const node = document.getElementById(layer.index < 0 ? 'input' : 'n' + layer.index); if (node) node.classList.toggle('dim', !matches(layer, query)); if (matches(layer, query)) visible++; }}); count.textContent = `${{visible}} of ${{layers.length}} nodes`; }}
document.querySelectorAll('#model-svg g.node').forEach(node => {{ const key = node.id === 'input' ? -1 : Number(node.id.slice(1)); const layer = layers.find(value => value.index === key); if (layer) node.addEventListener('click', () => showDetail(layer)); }});
document.getElementById('reset').addEventListener('click', () => {{ search.value = ''; draw(); }}); search.addEventListener('input', draw);
function setZoom(value) {{ zoom = Math.max(.35, Math.min(2.5, value)); svg.style.transform = `scale(${{zoom}})`; document.getElementById('zoom-reset').textContent = `${{Math.round(zoom * 100)}}%`; }}
document.getElementById('zoom-in').addEventListener('click', () => setZoom(zoom + .15)); document.getElementById('zoom-out').addEventListener('click', () => setZoom(zoom - .15)); document.getElementById('zoom-reset').addEventListener('click', () => setZoom(1)); draw();
</script></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="PixieNN model YAML file")
    parser.add_argument("-o", "--output", type=Path, help="HTML output path (default: <model>.html)")
    args = parser.parse_args()
    model_name, layers = read_model(args.model)
    output = args.output or args.model.with_suffix(".html")
    output.write_text(render_html(model_name, layers))
    print(f"Wrote {output} ({len(layers) - 1} model layers)")


if __name__ == "__main__":
    main()
