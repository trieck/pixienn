import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torchvision import transforms


def run_spec(h5_file_path):
    """
    Load a single test case from an HDF5 file and print its tensors and metadata.
    :param h5_file_path:
    :return: tensors, metadata
    """
    tensors = {}
    metadata = {}

    with h5py.File(h5_file_path, "r") as f:
        if len(f.keys()) == 0:
            raise ValueError(f"No test cases found in {h5_file_path}")

        test_name = list(f.keys())[0]  # Load the first test case

        grp = f[test_name]

        # Load tensors
        for name, dset in grp.items():
            tensors[name] = torch.tensor(dset[()])  # full read

        # Load metadata attributes
        for key, attr in grp.attrs.items():
            metadata[key] = attr if not isinstance(attr, np.ndarray) else attr.tolist()

    print(f"[INFO] Loaded test case '{test_name}' from {h5_file_path}")

    for name, tensor in tensors.items():
        print(f"  tensor: {name:>24} | shape: {tuple(tensor.shape)} | dtype: {tensor.dtype}")

    for key, value in metadata.items():
        print(f"  metadata: {key:>22} | value: {value}")

    return tensors, metadata


def save_layer(h5_file: h5py.File, layer_name: str, tensors: dict, metadata: dict):
    """
    Save a single test layer in an HDF5 file.

    Parameters:
        h5_file    : h5py file
        layer_name : name of layer
        tensors    : dict of {name: torch.Tensor}
        metadata   : dict of layer config and operation info
    """
    grp = h5_file.create_group(layer_name)

    # Write tensors
    for name, tensor in tensors.items():
        data = tensor.detach().cpu().numpy()
        grp.create_dataset(name, data=data)

    # Write metadata
    for key, value in metadata.items():
        grp.attrs[key] = value


def load_spec(path):
    """
    Load a specification from a YAML file.
    :param path:
    :return: dict
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_image_tensor(spec):
    image_filename = spec["image_filename"]
    image_channels = spec["image_channels"]
    image_size = spec["image_size"]

    img = cv2.imread(image_filename)
    if img.shape[2] == 3:  # RGB image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = letterbox(img, tuple(image_size))

    if image_channels == 1:
        img = img.mean(axis=2).astype(img.dtype)

    tensor = transforms.ToTensor()(img)

    return tensor.unsqueeze(0)  # NCHW


def letterbox(image, size):
    """
    Resize an image to fit within a target size while maintaining aspect ratio,
    :param image: numpy array of the image
    :param size: tuple of (width, height)
    :return: numpy array of the resized image
    """
    h, w = image.shape[:2]
    target_w, target_h = size

    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)

    # Determine dtype and midpoint color
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        midpoint = (info.min + info.max) // 2
    elif np.issubdtype(image.dtype, np.floating):
        midpoint = 0.5
    else:
        raise TypeError(f"Unsupported image dtype: {image.dtype}")

    channels = image.shape[2] if image.ndim == 3 else 1
    color = [midpoint] * channels
    boxed = np.full((target_h, target_w, channels), color, dtype=image.dtype)

    top = (target_h - nh) // 2
    left = (target_w - nw) // 2

    boxed[top:top + nh, left:left + nw] = resized

    return boxed


def apply_activation(x, act):
    """
    Apply the specified activation function to the input tensor.
    :param x: torch.Tensor
    :param act: str, name of the activation function
    :return: torch.Tensor
    """
    if act == "hardtanh":
        return F.hardtanh(x)
    if act == "relu":
        return F.relu(x)
    if act == "linear":
        return x
    if act == "logistic":
        return torch.sigmoid(x)
    if act == "mish":
        return x * torch.tanh(F.softplus(x))
    if act == "leaky":
        return F.leaky_relu(x, negative_slope=0.1)
    if act == "loggy":
        return 2 * torch.sigmoid(x) - 1
    if act == "softplus":
        return F.softplus(x)
    if act == "swish":
        return x * torch.sigmoid(x)
    if act == "tanh":
        return torch.tanh(x)
    if act == "selu":
        return F.selu(x)
    if act == "rrelu":
        return F.rrelu(x)
    if act == "hardshrink":
        return F.hardshrink(x)
    if act == "hardsigmoid":
        return F.hardsigmoid(x)
    if act == "gelu":
        return F.gelu(x, approximate='tanh')

    raise ValueError(f"Unknown activation: \"{act}\"")


def display_tensors(tensor: torch.Tensor, title: str = ""):
    """
    Display each of the channels of a tensor as an image.
    :param tensor: torch.Tensor, expected shape (1, C, H, W)
    :param title: str, optional
    """

    len = min(20, tensor.shape[1])

    for channel in range(len):
        img = tensor[0, channel, :, :]
        img = img.detach().numpy()

        img_min = img.min()
        img_max = img.max()

        if img_max > img_min:  # avoid divide by zero
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

        img = (img * 255).astype(np.uint8)  # Scale to 0-255

        title = f"{title}: Channel {channel} - Shape: {img.shape}"
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def compile_conv(layer_idx: int, layer: dict, input: torch.Tensor, h5_file: h5py.File):
    filters = layer["filters"]
    kernel = [layer["kernel"], layer["kernel"]]
    pad = layer["pad"] or 0
    stride = layer["stride"] or 1
    dilation = layer["dilation"] or 1
    activation = layer["activation"] or "linear"

    weights = torch.empty(filters, input.shape[1], *kernel, dtype=input.dtype).uniform_(-1.0, 1.0)
    bias = torch.empty(filters).uniform_(-0.5, 0.5)

    conv = torch.nn.Conv2d(in_channels=input.shape[1], out_channels=filters, kernel_size=kernel, stride=stride,
                           padding=pad, bias=True)
    with torch.no_grad():
        conv.weight.copy_(weights)
        conv.bias.copy_(bias)

    input.requires_grad_(True)

    pre_activation = conv(input)
    output = apply_activation(pre_activation, activation)

    tensors = {
        "bias": bias,
        "input": input,
        "output": output,
        "pre_activation": pre_activation,
        "weights": weights,
    }

    metadata = {
        "activation": activation,
        "dilation": dilation,
        "input_channels": input.shape[1],
        "input_shape": list(input.shape),
        "kernel": kernel,
        "layer_type": "conv",
        "operation": "forward",
        "output_channels": output.shape[1],
        "output_shape": list(output.shape),
        "padding": pad,
        "stride": stride
    }

    layer_name = "Conv_{}".format(layer_idx + 1)

    save_layer(h5_file=h5_file, layer_name=layer_name, tensors=tensors, metadata=metadata)

    return output


def compile_maxpool(layer_idx: int, layer: dict, input: torch.Tensor, h5_file: h5py.File):
    kernel = layer["kernel"]
    stride = layer["stride"]
    pad = layer["pad"] or 0
    dilation = layer["dilation"] or 1

    maxpool = torch.nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=pad, dilation=dilation)

    output = maxpool(input)

    tensors = {
        "input": input,
        "output": output,
    }

    metadata = {
        "input_shape": list(input.shape),
        "output_shape": list(output.shape),
        "kernel": kernel,
        "stride": stride,
        "padding": pad,
        "dilation": dilation,
    }

    layer_name = "MaxPool_{}".format(layer_idx + 1)

    save_layer(h5_file=h5_file, layer_name=layer_name, tensors=tensors, metadata=metadata)

    return output


def compile_fc(layer_idx: int, layer: dict, input: torch.Tensor, h5_file: h5py.File):
    activation = layer["activation"]
    out_features = layer["output"]

    input = input.view(input.size(0), -1)  # Flatten
    in_features = input.shape[1]

    weights = torch.empty(out_features, in_features, dtype=input.dtype).uniform_(-1.0, 1.0)
    bias = torch.empty(out_features).uniform_(-0.5, 0.5)

    fc = torch.nn.Linear(in_features=input.shape[1], out_features=out_features)

    with torch.no_grad():
        fc.weight.copy_(weights)
        fc.bias.copy_(bias)

    pre_activation = fc(input)
    output = apply_activation(pre_activation, activation)

    tensors = {
        "input": input,
        "pre_activation": pre_activation,
        "output": output,
        "weights": weights,
        "bias": bias
    }

    metadata = {
        "activation": activation,
        "in_features": in_features,
        "out_features": out_features
    }

    layer_name = "Connected_{}".format(layer_idx + 1)

    save_layer(h5_file=h5_file, layer_name=layer_name, tensors=tensors, metadata=metadata)

    return output


def compile_layer(layer_idx: int, layer: dict, input: torch.Tensor, h5_file: h5py.File):
    type = layer["type"]

    if type == "conv":
        return compile_conv(layer_idx, layer, input, h5_file)
    elif type == "maxpool":
        return compile_maxpool(layer_idx, layer, input, h5_file)
    elif type == "connected":
        return compile_fc(layer_idx, layer, input, h5_file)
    else:
        raise TypeError(f"Unknown layer type: {type}")


def compile_spec(spec_file: str, output_file: str):
    """
    Compile a specification from a YAML file into an HDF5 test case.
    :param spec_file: path to the YAML specification file
    :param output_file: path to the output HDF5 file
    """
    h5_file = h5py.File(output_file, "w")

    spec = load_spec(spec_file)

    torch.manual_seed(spec.get("seed", 42))

    layers = spec.get("layers", [])
    input = load_image_tensor(spec)

    for i in range(0, len(layers)):
        input = compile_layer(i, layers[i], input, h5_file)

    h5_file.close()


if __name__ == "__main__":
    compile_spec("specs/test_1.yaml", "tests.h5")
