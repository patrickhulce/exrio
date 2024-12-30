# exrio

Fast and dependency-free python library for reading and writing OpenEXR files.

## Installation

```bash
pip install exrio
```

## Usage

### Read an EXR file from disk (sRGB)

```python
from exrio import load

image = load("path/to/image.exr")
pixels = image.to_pixels()
```

### Read an EXR file from memory (ACES 2065-1)

```python
import boto3
from exrio import ExrImage

s3 = boto3.client("s3")
buffer = s3.get_object(Bucket="bucket-name", Key="path/to/image.exr")["Body"].read()
image = ExrImage.from_buffer(buffer)
pixels = image.to_pixels()

print(f"Red Chromaticity: {image.chromaticities.red}")
print(f"Green Chromaticity: {image.chromaticities.green}")
print(f"Blue Chromaticity: {image.chromaticities.blue}")
```

### Write an EXR file (sRGB)

```python
image = ExrImage.from_pixels(pixels)
image.to_path("path/to/output.exr")
```

### Write an EXR to S3 (ACEScg)

```python
image = ExrImage.from_pixels_ACEScg(pixels)
s3 = boto3.client("s3")
s3.put_object(Bucket="bucket-name", Key="path/to/output.exr", Body=image.to_buffer())
```

### Read a Custom Multi-Layer EXR

```python
from exrio import load

image = load("path/to/image.exr")
print(f"Attributes: {image.attributes}")
print(f"Chromaticities: {image.chromaticities}")

for layer in image.layers:
    print(f"Layer: {layer.name}")
    print(f"  Width: {layer.width}")
    print(f"  Height: {layer.height}")
    print(f"  Attributes: {layer.attributes}")

    for channel in layer.channels:
        print(f"    Channel: {channel.name}")
        print(f"      Pixels: {channel.pixels.shape}, dtype: {channel.pixels.dtype}")
```

### Write a Custom Multi-Layer EXR

```python
from exrio import ExrImage, ExrLayer, ExrChannel

pixels = np.random.rand(1920 * 1080).astype(np.float32)

channels = [
    ExrChannel(name="R", width=1920, height=1080, pixels=pixels),
    ExrChannel(name="G", width=1920, height=1080, pixels=pixels),
    ExrChannel(name="B", width=1920, height=1080, pixels=pixels),
]
mask = ExrChannel(name="A", width=1920, height=1080, pixels=pixels)
layer = ExrLayer(name="color", width=1920, height=1080, channels=channels)
mask_layer = ExrLayer(name="mask", width=1920, height=1080, channels=[mask])
chromaticities = Chromaticities(red=(0.68, 0.32), green=(0.265, 0.70), blue=(0.15, 0.06))
image = ExrImage(layers=[layer, mask_layer], chromaticities=chromaticities)
image.to_path("path/to/output.exr")
```

## Development

### Install Tools

- [Rust](https://www.rust-lang.org/tools/install)
- [Python](https://www.python.org/downloads/)
- [maturin](https://maturin.rs)

### Run Tests

```bash
uv sync
uv run pytest
```

### Run Examples

```bash
uv sync
uv run examples/test.py
```

### Publish

Create an account on [test.pypi.org](https://test.pypi.org) and register your GitHub OIDC provider on [Publishing](https://test.pypi.org/manage/account/publishing/).
