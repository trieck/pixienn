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

Inference takes confidence explicitly; NMS is a separate secondary filter:

```python
result = model.predict_image(
    "image.jpg", confidence=0.25, nms_threshold=0.4)

# Run one native batch and write predictions.jpg as a mosaic.
result = model.predict_batch_image_list(
    "validation-images.txt", confidence=0.25, nms_threshold=0.4)
```

`validation-images.txt` contains one image path per line. The native model
loads the list in batches matching the model's configured batch size, returns
all detections with their `batch_id`, and writes a single `predictions.jpg`
containing the annotated images.

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

## Native CenterNet architecture

`python/examples/centernet_demo.py` is a CenterNet experiment built through the native
PixieNN Python binding. Python describes the graph and orchestrates the run;
convolution, routing, upsampling, target generation, focal loss, decoding,
CUDA execution, checkpoints, and event writing remain in C++.

For a `320 x 320` input, the model follows this encoder-decoder progression:

```text
320x320 input
    ↓ stride 2, 32 channels       160x160
    ↓ stride 2, 64 channels         80x80
    ↓ stride 2, 128 channels        40x40
    ↓ stride 2, 256 channels        20x20
    ↓ stride 2, 384 channels        10x10
    ↓ 1x1, 256 channels              10x10
    ↑ upsample                       20x20
    + route from the 20x20 encoder feature
    ↓ 3x3 and 1x1 refinement         20x20
    ↑ upsample                       40x40
    + route from the 40x40 encoder feature
    ↓ 3x3 and 1x1 refinement         40x40
    ↑ upsample                       80x80
    + route from the 80x80 encoder feature
    ↓ 3x3 and 3x3 refinement         80x80
    ↓ 1x1 prediction                  80x80
    ↓ CenterNet decoding
```

The five downsampling stages increase receptive field and image context. The
decoder restores a stride-4 feature map, giving CenterNet a reasonably fine
grid for locating smaller objects.

The route layers are skip connections. They join deep semantic features with
earlier, higher-resolution features: the deep path supplies context while the
earlier path preserves spatial detail lost during downsampling. The three
routes correspond to the `20x20`, `40x40`, and `80x80` feature maps.

The final prediction has 24 channels for the 20 VOC classes:

```text
20 class heatmaps + 2 width/height values + 2 center offsets = 24 channels
```

The CenterNet head uses focal-loss parameters `alpha=2` and `beta=4`, with a
negative heatmap bias to make the initial prediction sparse. Width/height
regression is weighted separately because it has a different scale from the
dense heatmap loss.

Training uses Adam, horizontal flips, moderate geometric/color augmentation,
gradient rescaling, and cosine annealing from `0.0002` toward `0.000002`.
These choices address the instability and limited representation capacity of
the original tiny prototype.

Changing this graph makes old CenterNet weights incompatible. Start a new
experiment with:

```bash
python3 python/examples/centernet_demo.py \
  --fresh \
  --batch-size=32 \
  --epochs=5000 \
  --image-size=320 \
  --output runs/centernet
```

Validation evaluates all selected validation images at every validation
interval. Use `--validation-count` to choose a smaller fixed validation subset
for a quick smoke test; the native validator recreates its deterministic
validation loader at each interval so the same images are evaluated repeatedly.

`--fresh` removes the existing CenterNet checkpoints and event file for that
output directory. Later runs can resume using the native weights, optimizer
state, and training-control sidecars produced by PixieNN.
