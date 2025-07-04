# test_conv_layer.py
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# === 1. Create a conv layer with explicit params ===
def make_conv(weights, bias, stride=1, padding=0, dilation=1):
    out_channels, in_channels, kh, kw = weights.shape
    conv = torch.nn.Conv2d(
        in_channels, out_channels,
        kernel_size=(kh, kw),
        stride=stride, padding=padding,
        dilation=dilation, bias=True
    )
    with torch.no_grad():
        conv.weight.copy_(weights)
        conv.bias.copy_(bias)
    return conv


# === 2. Load and letterbox the image ===
def load_letterboxed_image(path, size=(32, 32)):
    img = Image.open(path).convert("RGB")
    img = letterbox(img, size, color=(128, 128, 128))
    tensor = transforms.ToTensor()(img)  # 0..1 float
    return tensor.unsqueeze(0)  # Add batch dim


def letterbox(image, target_size, color=(128, 128, 128)):
    # Resize keeping aspect ratio and pad with color to fit target size
    orig_w, orig_h = image.size
    target_w, target_h = target_size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = image.resize((new_w, new_h), Image.BICUBIC)

    letterboxed = Image.new("RGB", (target_w, target_h), color)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    letterboxed.paste(resized, (paste_x, paste_y))
    return letterboxed


# === 3. Activation function map ===
def apply_activation(x, act):
    if act == "linear":
        return x
    elif act == "relu":
        return F.relu(x)
    elif act == "logistic":
        return torch.sigmoid(x)
    elif act == "mish":
        return x * torch.tanh(F.softplus(x))
    else:
        raise ValueError(f"Unsupported activation: {act}")


# === 4. The actual test ===
@pytest.mark.parametrize("activation", ["linear", "relu", "logistic", "mish"])
def test_conv_forward_output(activation):
    # (C_out, C_in, H_k, W_k)
    weights = torch.tensor([[[[1.0, 0.0],
                              [0.0, -1.0]]]], dtype=torch.float32)
    bias = torch.tensor([0.0], dtype=torch.float32)

    conv = make_conv(weights, bias, stride=1, padding=0)

    # Load input image
    input_tensor = load_letterboxed_image("/data/imagery/dog.jpg", size=(448, 448))

    # Reduce to 1 channel if needed (match weights)
    input_tensor = input_tensor[:, 0:1, :, :]

    # Enable grad for training-mode check
    input_tensor.requires_grad_(True)

    # === Forward pass
    preact = conv(input_tensor)
    out = apply_activation(preact, activation)

    # === Assertions
    assert out.shape[1:] == torch.Size([1, 447, 447]), f"Unexpected output shape: {out.shape}"

    # === Optional: verify specific output region
    expected = out.detach().numpy()[0, 0, :3, :3]  # small patch
    print("Activation:", activation)
    print("Top-left output patch:\n", expected)

    # === Optional: Save for PixieNN test
    # np.save("test_vectors/input.npy", input_tensor.detach().numpy())
    # np.save("test_vectors/output.npy", out.detach().numpy())
