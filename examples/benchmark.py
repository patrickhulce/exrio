import os
import tempfile
import time
from typing import Callable

import numpy as np

from exrio._rust import ExrImage as RustImage
from exrio._rust import ExrLayer as RustLayer


def benchmark(label: str, func: Callable[[], RustImage]):
    start = time.time()
    for _ in range(10):
        func()
    end = time.time()
    duration = (end - start) * 1000 / 10
    print(f"{label}: {duration:.1f}ms")


def benchmark_pypath_based_fn():
    with tempfile.NamedTemporaryFile(suffix=".exr") as f:
        image = create_test_image()
        buffer = image.save_to_buffer()
        with open(f.name, "wb") as f:
            f.write(buffer)

        with open(f.name, "rb") as f:
            RustImage.load_from_buffer(f.read())
        print(f"size: {os.path.getsize(f.name) / 1024 / 1024:.1f}MB")


def benchmark_buffer_based_fn():
    image = create_test_image()
    buffer = image.save_to_buffer()
    RustImage.load_from_buffer(buffer)
    print(f"size: {len(buffer) / 1024 / 1024:.1f}MB")


def create_test_image():
    pixels = np.random.rand(1024, 1024, 1).reshape(-1).astype(np.float32).copy()
    layer = RustLayer("test")
    layer.with_width(1024)
    layer.with_height(1024)
    layer.with_channel_f32("R", pixels)
    layer.with_channel_f32("G", pixels)
    layer.with_channel_f32("B", pixels)

    image = RustImage()
    image.with_layer(layer)
    return image


if __name__ == "__main__":
    benchmark("buffer_based", benchmark_buffer_based_fn)
    benchmark("pypath_based", benchmark_pypath_based_fn)
