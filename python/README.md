# PixieNN Python interface

Version `0.1.0` is the first versioned Python interface release. The source
of truth for the version is `pixienn.__version__`.

## Project and Darknet compatibility

PixieNN is a C++ inference/training engine derived from the Darknet model and
weight conventions. Its parser and layer implementations preserve the
Darknet-style graph, tensor layout, detector heads, and binary weight format;
Darknet `.cfg` files and `.weights` files can therefore be used as model
references and, where the graph is supported, loaded directly. PixieNN is a
separate implementation, not a copy of the Darknet executable.

The upstream Darknet project is documented at
<https://pjreddie.com/darknet/> and the YOLOv3 COCO weights are available from
the original release page:
<https://pjreddie.com/media/files/yolov3.weights>.
The matching Darknet configuration and labels are:

```bash
wget https://pjreddie.com/media/files/yolov3.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
```

The canonical Darknet check is:

```bash
./darknet detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights data/dog.jpg
```

PixieNN should produce the same objects and broadly the same boxes and
confidence values, subject to implementation and floating-point differences.
The PixieNN CLI equivalent is:

```bash
pixienn --confidence=0.25 --nms=0.45 \
  --weights=yolov3.weights resources/cfg/yolov3-cfg.yml resources/images/dog.jpg
```

The Python API uses the same graph and weights, but the graph must be created
with Python layer definitions (or a generated Python-compatible spec); YAML is
not required by the Python-facing model API. The YOLOv1-tiny test in
`python/tests/test_model.py` is the executable reference for this workflow.

## Training

Training uses the same Darknet-style configuration concepts: dataset image and
label lists, validation lists, batch/subdivisions, optimizer, learning-rate
policy, checkpoint weights, and validation thresholds. Configure these before
building the graph, then run:

```python
model.configure_training(
    **{
        "train-images": "data/train.txt",
        "train-labels": "data/labels",
        "val-images": "data/val.txt",
        "val-labels": "data/labels",
    }
).build()
model.train()
```

For a complete training run, use the repository's native CLI configuration
until the Python training graph for the desired detector is defined. See the
project source and tests at <https://github.com/trieck/pixienn>.

## Packaging status

The current repository build produces the native extension through CMake. A
PyPI release must package that compiled extension into platform-specific
wheels; a pure Python wheel would not be functional. CPU and CUDA wheels must
be built separately because CUDA/cuDNN and GPU compatibility are platform
constraints. The intended release names are `pixienn` for CPU and
`pixienn-cuda` for CUDA.

The native extension is built from the C++ source at
`src/python/pixienn_module.cpp`. Enable it with:

```bash
cmake -S . -B build-python \
  -DPIXIENN_BUILD_PYTHON_BINDINGS=ON \
  -DUSE_CUDA=OFF
cmake --build build-python --target pixienn_native
cmake --install build-python --prefix /path/to/prefix
```

The build produces a self-contained package directory at
`build-python/python/pixienn`. Use it with:

```bash
PYTHONPATH=build-python/python python -c \
  "from pixienn import Tensor; print(Tensor([2, 3]).shape)"
```

`Tensor` currently provides CPU-backed construction, indexing, filling,
shape/stride metadata, and the Python buffer protocol.

## Python-first model graphs

Model graphs are authored as Python data and use the native PixieNN layer
factories for shape inference and execution:

```python
from pixienn import Model, Tensor

model = (Model(3, 416, 416, batch=1)
         .conv(32, 3, activation="leaky", batch_normalize=True)
         .maxpool(2, 2)
         .build())

output = model.forward(Tensor([1, 3, 416, 416]))
```

Every native layer can be supplied with `model.layer(type, **properties)`;
helpers are provided for common layers. Use `model.config(...)` for a
portable document or `model.save_native_files(...)` for the native runner's
two-file layout. Training settings can be attached before
`build()` with `configure_training(...)`, then started with `train()` or
`evaluate()`. Detector inference is available as `predict_json(image_path)`.
Learned state can be persisted with `save_weights(path)` and restored with
`load_weights(path)` after building the same graph. Training-control state is
available through `save_training_state(path)`, and execution mode can be
selected with `set_mode("inference"|"training"|"validation")`.
