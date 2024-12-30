import json
import re
import traceback
import zipfile
from pathlib import Path

import numpy as np
import PIL.Image
import requests

from exrio import ExrChannel, ExrLayer, load

EXAMPLES_ZIP_URL = "https://github.com/patrickhulce/exrio/releases/download/v0.0.2/examples-v20241230.zip"
EXAMPLES_DATA_PATH = Path(__file__).parent.parent / ".data" / "examples-v20241230"


def download_examples() -> None:
    if EXAMPLES_DATA_PATH.exists():
        print("Examples already downloaded, skipping...")
        return

    print("Downloading examples...")
    response = requests.get(EXAMPLES_ZIP_URL)
    response.raise_for_status()
    EXAMPLES_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    zip_path = EXAMPLES_DATA_PATH.with_suffix(".zip")
    with open(zip_path, "wb") as file:
        file.write(response.content)

    print("Extracting examples...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(EXAMPLES_DATA_PATH)
    zip_path.unlink()
    print("Examples downloaded and extracted.")


def linear_to_srgb(pixels: np.ndarray) -> np.ndarray:
    return np.clip(pixels**2.2, 0, 1)


def write_channel(layer: ExrLayer, channel: ExrChannel, exr_path: Path) -> None:
    layer_name = layer.name or "unknown_layer"
    clean_layer_name = re.sub(r"[^a-zA-Z0-9]+", "_", layer_name)
    channel_name = channel.name or "unknown_channel"
    clean_channel_name = re.sub(r"[^a-zA-Z0-9]+", "_", channel_name)
    channel_path = exr_path.with_suffix(".png")
    channel_path = channel_path.with_stem(
        f"{channel_path.stem}_{clean_layer_name}_{clean_channel_name}"
    )
    channel_pixels = channel.pixels.astype(np.float32)
    max_pixel_value = np.max(channel_pixels)
    min_pixel_value = np.min(channel_pixels)
    if min_pixel_value >= -0.1 and max_pixel_value <= 1:
        normalized_pixels = channel_pixels * 255
        normalized_pixels = normalized_pixels.clip(0, 255).astype(np.uint8)
    else:
        # It's probably linear, so we need to do a rough gamma correction.
        normalized_pixels = linear_to_srgb(channel_pixels)
        normalized_pixels = (normalized_pixels * 255).clip(0, 255).astype(np.uint8)

    print(f"Saving channel {channel_name} in {layer_name} to {channel_path}...")
    channel_image = PIL.Image.fromarray(normalized_pixels)
    channel_image.save(channel_path)


def run_example(example_exr_path: Path) -> None:
    print(f"Loading example {example_exr_path}...")
    try:
        image = load(example_exr_path)
    except Exception as e:
        print(f"Error loading example {example_exr_path}: {e}")
        return

    metadata_path = example_exr_path.with_suffix(".json")
    metadata_path = metadata_path.with_stem(f"{metadata_path.stem}_metadata")
    metadata = {
        "image": {
            "attributes": image.attributes,
        },
        "layers": [],
    }

    for layer in image.layers:
        metadata["layers"].append(
            {
                "name": layer.name,
                "width": layer.width,
                "height": layer.height,
                "attributes": layer.attributes,
                "channels": [
                    {
                        "name": channel.name,
                        "dtype": str(channel.pixels.dtype),
                        "shape": str(channel.pixels.shape),
                    }
                    for channel in layer.channels
                ],
            }
        )

    print(f"Saving metadata to {metadata_path}...")
    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=2)

    for layer in image.layers:
        for channel in layer.channels:
            try:
                write_channel(layer, channel, example_exr_path)
            except Exception as e:
                stacktrace = traceback.format_exc()
                print(f"Error writing channel {channel.name} in {layer.name}: {e}")
                print(f"Stack trace: {stacktrace}")

    print(f"Done with example {example_exr_path}")


def run_all_examples() -> None:
    for example_exr_path in EXAMPLES_DATA_PATH.rglob("*.exr"):
        run_example(example_exr_path)


if __name__ == "__main__":
    download_examples()
    run_all_examples()
