#!/usr/bin/env python3
"""Serve the PixieNN React model graph viewer and parse uploaded YAML models."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

MODEL_GRAPH = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_GRAPH))
from pixienn_model_graph import infer_layers, render_svg  # noqa: E402


class Handler(SimpleHTTPRequestHandler):
    default_model: Path | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(MODEL_GRAPH / "dist"), **kwargs)

    def send_json(self, value: object, status: int = 200) -> None:
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def inspect(name: str, text: str) -> dict:
        document = yaml.safe_load(text) or {}
        model = document.get("model", document)
        if not isinstance(model, dict) or not isinstance(model.get("layers"), list):
            raise ValueError("The YAML must contain a model.layers sequence.")
        layers = infer_layers(model)
        return {"name": name, "layers": layers, "svg": render_svg(layers)}

    def do_GET(self) -> None:
        if self.path == "/api/model":
            if self.default_model is None:
                self.send_json({"error": "No default model was configured."}, 404)
                return
            try:
                self.send_json(self.inspect(self.default_model.name, self.default_model.read_text()))
            except Exception as error:
                self.send_json({"error": str(error)}, 400)
            return
        if self.path == "/" or self.path == "":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/model":
            self.send_json({"error": "Not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            self.send_json(self.inspect(str(payload.get("name", "model.yml")), str(payload["text"])))
        except Exception as error:
            self.send_json({"error": str(error)}, 400)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("resources/models/centernet-prosopo.yml"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5179)
    args = parser.parse_args()
    Handler.default_model = args.model.resolve()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"PixieNN model graph: http://{args.host}:{args.port}/")
    print(f"Default model: {Handler.default_model}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
