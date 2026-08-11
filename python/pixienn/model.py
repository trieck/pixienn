"""Python-first model graph construction backed by PixieNN layer factories.

The graph is deliberately kept as ordinary Python data.  This makes model
definitions composable and inspectable while the native engine remains the
source of truth for layer shape inference and execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ._native import Model as _NativeModel, Tensor
try:
    from ._native import CudaModel as _NativeCudaModel
    from ._native import CudaTensor
except ImportError:
    _NativeCudaModel = None
    CudaTensor = None


class Model:
    """Build a PixieNN model without hand-writing a static YAML graph."""

    def __init__(self, channels: int, height: int, width: int, batch: int = 1,
                 *, device: str = "cpu", **options: Any):
        self._spec: dict[str, Any] = {
            "channels": channels, "height": height, "width": width,
            "batch": batch, "layers": [], **options,
        }
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'")
        if device == "cuda":
            if _NativeCudaModel is None:
                raise RuntimeError("PixieNN was built without CUDA Python bindings")
            self._native = _NativeCudaModel(channels, height, width, batch)
        else:
            self._native = _NativeModel(channels, height, width, batch)
        self._device = device
        # Python bindings are quiet by default; callers may opt into native
        # layer/timing/detection logging with verbose=True.
        native_options = {"verbose": False, **options}
        self._native.set_options(native_options)

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "Model":
        required = ("channels", "height", "width")
        missing = [key for key in required if key not in spec]
        if missing:
            raise ValueError(f"model spec missing {', '.join(missing)}")
        model = cls(spec["channels"], spec["height"], spec["width"], spec.get("batch", 1),
                    **{k: v for k, v in spec.items() if k not in (*required, "batch", "layers")})
        for layer in spec.get("layers", []):
            model.add_layer(layer)
        return model

    @classmethod
    def from_config(cls, path: str | Path) -> "Model":
        """Load a JSON-compatible native configuration emitted by ``save_config``."""
        document = json.loads(Path(path).read_text())
        if not isinstance(document, Mapping) or not isinstance(document.get("model"), Mapping):
            raise ValueError("configuration must contain a model mapping")
        model = cls.from_spec(document["model"])
        if isinstance(document.get("training"), Mapping):
            model.configure_training(**document["training"])
        return model

    @property
    def spec(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._spec))

    @property
    def training_config(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._spec.get("training", {})))

    @property
    def built(self) -> bool:
        return self._native.built

    @property
    def layer_count(self) -> int:
        return self._native.layer_count

    @property
    def input_shape(self) -> list[int]:
        return [self.channels, self.height, self.width]

    @property
    def output_shape(self) -> list[int]:
        return self._native.output_shape

    @property
    def device(self) -> str:
        return self._device

    def add_layer(self, definition: Mapping[str, Any]) -> "Model":
        if not isinstance(definition, Mapping) or "type" not in definition:
            raise ValueError("layer definition must be a mapping with a type")
        layer = dict(definition)
        self._native.add_layer(layer)
        self._spec["layers"].append(json.loads(json.dumps(layer)))
        return self

    def layer(self, layer_type: str, **kwargs: Any) -> "Model":
        return self.add_layer({"type": layer_type, **kwargs})

    def conv(self, filters: int, kernel: int, stride: int = 1, *, pad: bool = True,
             activation: str = "linear", batch_normalize: bool = False,
             dilation: int = 1) -> "Model":
        return self.layer("conv", filters=filters, kernel=kernel, stride=stride,
                          pad=pad, activation=activation, batch_normalize=batch_normalize,
                          dilation=dilation)

    def maxpool(self, kernel: int, stride: int) -> "Model":
        return self.layer("maxpool", kernel=kernel, stride=stride)

    def avgpool(self, kernel: int = 0, stride: int = 1) -> "Model":
        args = {"stride": stride}
        if kernel:
            args["kernel"] = kernel
        return self.layer("avgpool", **args)

    def batchnorm(self) -> "Model":
        return self.layer("batchnorm")

    def shortcut(self, from_: int, activation: str = "linear") -> "Model":
        return self.layer("shortcut", **{"from": from_, "activation": activation})

    def route(self, layers: list[int]) -> "Model":
        return self.layer("route", layers=layers)

    def upsample(self, stride: int) -> "Model":
        return self.layer("upsample", stride=stride)

    def dropout(self, probability: float) -> "Model":
        return self.layer("dropout", probability=probability)

    def connected(self, output: int, activation: str = "logistic",
                  batch_normalize: bool = False) -> "Model":
        return self.layer("connected", output=output, activation=activation,
                          batch_normalize=batch_normalize)

    def softmax(self, groups: int = 1, temperature: float = 1.0,
                detector: bool = True) -> "Model":
        return self.layer("softmax", groups=groups, temperature=temperature,
                          detector=detector)

    def yolo(self, anchors: list[int], mask: list[int], **kwargs: Any) -> "Model":
        return self.layer("yolo", anchors=anchors, mask=mask, **kwargs)

    def region(self, anchors: list[float], num: int, **kwargs: Any) -> "Model":
        return self.layer("region", anchors=anchors, num=num, **kwargs)

    def detection(self, side: int = 7, num: int = 2, **kwargs: Any) -> "Model":
        return self.layer("detection", side=side, num=num, **kwargs)

    def centernet(self, **kwargs: Any) -> "Model":
        return self.layer("centernet", **kwargs)

    def build(self) -> "Model":
        self._native.build()
        return self

    def configure_training(self, **settings: Any) -> "Model":
        """Attach native trainer settings before building the graph."""
        self._spec["training"] = json.loads(json.dumps(settings))
        self._native.configure_training(settings)
        return self

    def forward(self, input_tensor: Any):
        """Run one CPU batch and return a native :class:`Tensor`."""
        if not hasattr(input_tensor, "values"):
            view = memoryview(input_tensor)
            if not view.c_contiguous or view.format not in ("f", "d"):
                raise TypeError("forward input must be a float32/float64 C-contiguous buffer")
            shape = list(view.shape or (view.nbytes // view.itemsize,))
            values = list(view) if view.ndim == 1 else list(view.cast("B").cast(view.format))
            if view.format == "d":
                values = [float(value) for value in values]
            input_tensor = Tensor(shape, values)
        return self._native.forward(input_tensor)

    def set_labels(self, labels: list[str]) -> "Model":
        self._native.set_labels(labels)
        self._spec["labels"] = list(labels)
        return self

    def evaluate(self) -> None:
        self._native.evaluate()

    def train(self) -> "Model":
        """Train a configured CPU graph and return the model.

        Training must be configured before ``build()``.  The CUDA Python
        binding currently exposes inference but not the native trainer.
        """
        native_train = getattr(self._native, "train", None)
        if native_train is None:
            raise RuntimeError("training is not exposed by the selected native device binding")
        native_train()
        return self

    def load_weights(self, path: str | Path) -> "Model":
        self._native.load_weights(str(path))
        return self

    def save_weights(self, path: str | Path) -> "Model":
        self._native.save_weights(str(path))
        return self

    def save_training_state(self, path: str | Path) -> "Model":
        self._native.save_training_state(str(path))
        return self

    def set_mode(self, mode: str) -> "Model":
        self._native.set_mode(mode)
        return self

    @property
    def mode(self) -> str:
        return self._native.mode

    @property
    def labels(self) -> list[str]:
        return self._native.labels

    def predict_json(self, image_file: str | Path, nms_threshold: float = 0.3) -> str:
        return self._native.predict_json(str(image_file), nms_threshold)

    def predict_image(self, image_file: str | Path, *, nms_threshold: float = 0.3,
                      geojson_path: str | Path | None = None) -> str:
        """Run inference, render ``predictions.jpg``, and optionally save GeoJSON."""
        result = self._native.predict_image(str(image_file), nms_threshold)
        if geojson_path is not None:
            Path(geojson_path).write_text(result)
        return result

    @property
    def cost(self) -> float:
        return self._native.cost

    @property
    def learning_rate(self) -> float:
        return self._native.learning_rate

    @property
    def seen(self) -> int:
        return self._native.seen

    @property
    def batch(self) -> int:
        return self._native.batch

    @property
    def layers(self) -> list[dict[str, Any]]:
        return json.loads(json.dumps(self._spec["layers"]))

    @property
    def layer_shapes(self) -> list[list[int]]:
        return self._native.layer_shapes

    @property
    def threshold(self) -> float:
        return self._native.threshold

    @property
    def channels(self) -> int:
        return self._native.channels

    @property
    def height(self) -> int:
        return self._native.height

    @property
    def width(self) -> int:
        return self._native.width

    def set_threshold(self, threshold: float) -> "Model":
        self._native.set_threshold(threshold)
        return self

    @property
    def adam_enabled(self) -> bool:
        return self._native.adam_enabled

    @property
    def adam_beta1(self) -> float:
        return self._native.adam_beta1

    @property
    def adam_beta2(self) -> float:
        return self._native.adam_beta2

    @property
    def adam_epsilon(self) -> float:
        return self._native.adam_epsilon

    def save_spec(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"model": self.spec}, indent=2) + "\n")

    def config(self, *, configuration: Mapping[str, Any], training: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Return a complete portable configuration document.

        The architecture is still authored in Python. The native runner's
        historical two-file layout can be emitted with ``save_native_files``.
        """
        result: dict[str, Any] = {"configuration": dict(configuration), "model": self.spec}
        if training is not None:
            result["training"] = dict(training)
        return result

    def save_config(self, path: str | Path, *, configuration: Mapping[str, Any],
                    training: Mapping[str, Any] | None = None) -> None:
        Path(path).write_text(json.dumps(self.config(configuration=configuration, training=training), indent=2) + "\n")

    def save_native_files(self, directory: str | Path, *, configuration: Mapping[str, Any],
                          training: Mapping[str, Any] | None = None) -> Path:
        """Write the model and runner documents used by the native CLI."""
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        model_path = root / "model.json"
        model_path.write_text(json.dumps({"model": self.spec}, indent=2) + "\n")
        runner = dict(configuration)
        runner["model"] = model_path.name
        config_path = root / "config.json"
        config_path.write_text(json.dumps(self.config(configuration=runner, training=training), indent=2) + "\n")
        return config_path


__all__ = ["Model"]
