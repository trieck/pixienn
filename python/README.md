# PixieNN Python interface

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
