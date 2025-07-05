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


def save_test_case(file_path, test_name, tensors, metadata):
    """
    Save a single test case under the group /<test_name>/ in an HDF5 file.

    Parameters:
        file_path  : path to .h5 file
        test_name  : name of the test case, used as group name
        tensors    : dict of {name: torch.Tensor}
        metadata   : dict of layer config and operation info
    """
    with h5py.File(file_path, "w") as f:
        if test_name in f:
            raise ValueError(f"Test case '{test_name}' already exists in {file_path}")

        grp = f.create_group(test_name)

        # Write tensors
        for name, tensor in tensors.items():
            data = tensor.detach().cpu().numpy()
            grp.create_dataset(name, data=data)

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


def load_image_tensor(image_path, size, in_channels):
    """
    Load an image from the given path, resize it to the specified size,
    :param image_path: path to the image file
    :param size:  tuple of (width, height)
    :param in_channels: number of input channels (e.g., 3 for RGB, 1 for grayscale)
    :return: torch.Tensor
    """
    img = cv2.imread(image_path)
    if img.shape[2] == 3:  # RGB image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = letterbox(img, tuple(size))
    tensor = transforms.ToTensor()(img)

    tensor = tensor[:in_channels, :, :]

    return tensor.unsqueeze(0)


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
    if act == "relu":
        return F.relu(x)
    if act == "linear":
        return x
    if act == "logistic":
        return torch.sigmoid(x)
    if act == "mish":
        return x * torch.tanh(F.softplus(x))
    if act == "leaky":
        return F.leaky_relu(x, negative_slope=0.01)

    raise ValueError(f"Unknown activation: {act}")


def display_tensors(tensor: torch.Tensor):
    """
    Display each of the channels of a tensor as an image.
    :param tensor: torch.Tensor, expected shape (1, C, H, W)
    """ 

    for channel in range(tensor.shape[1]):
        img = tensor[0, channel, :, :]
        img = img.detach().numpy()
        img = (img * 255).astype(np.uint8)  # Scale to 0-255

        title = f"Channel {channel} - Shape: {img.shape}"
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def compile_spec(spec_path, output_file):
    """
    Compile a specification from a YAML file into an HDF5 test case.
    :param spec_path: path to the YAML specification file
    :param output_file: path to the output HDF5 file
    """
    spec = load_spec(spec_path)

    torch.manual_seed(spec.get("seed", 42))

    c_in = spec['input_channels']
    c_out = spec['conv']['out_channels']
    k_h, k_w = spec['conv']['kernel_size']
    padding = spec['conv']['padding']
    stride = spec['conv']['stride']
    dilation = spec['conv']['dilation']

    weights = torch.empty(c_out, c_in, k_h, k_w).uniform_(-1.0, 1.0)
    bias = torch.empty(c_out).uniform_(-0.5, 0.5)

    conv = torch.nn.Conv2d(c_in, c_out, (k_h, k_w),
                           stride=stride,
                           padding=padding,
                           dilation=dilation,
                           bias=True)
    with torch.no_grad():
        conv.weight.copy_(weights)
        conv.bias.copy_(bias)

    input_image = spec['input_image']
    resize = spec['resize']

    input_tensor = load_image_tensor(input_image, resize, c_in)

    display_tensors(input_tensor)

    input_tensor.requires_grad_(True)
    out = conv(input_tensor)

    display_tensors(out)

    activation = spec.get('activation', 'relu')
    act_out = apply_activation(out, activation)

    display_tensors(act_out)

    tensors = {
        "input_tensor": input_tensor,
        "pre_activation_output": out,
        "post_activation_output": act_out,
        "weights": weights,
        "bias": bias,
    }

    metadata = {
        "layer_type": "conv",
        "operation": "forward",
        "activation": "relu",
        "input_shape": list(input_tensor.shape)[1:],  # CHW
        "output_shape": list(act_out.shape)[1:],  # CHW
        "kernel_size": [k_h, k_w],
        "output_channels": c_out,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
    }

    test_name = spec.get("name")

    save_test_case(output_file, test_name, tensors, metadata)


if __name__ == "__main__":
    compile_spec("specs/basic_conv_test.yaml", "tests.h5")
    run_spec("tests.h5")
