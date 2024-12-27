from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np

from exrio._rust import ExrImage as RustImage
from exrio._rust import ExrLayer as RustLayer


def _pixels_from_layer(layer: RustLayer) -> list[np.ndarray]:
    pixels = layer.pixels_f32()
    assert pixels is not None
    return [
        pixels[i].reshape(layer.height(), layer.width()) for i in range(len(pixels))
    ]


@dataclass
class ExrChannel:
    name: str
    width: int
    height: int
    pixels: np.ndarray

    @staticmethod
    def _from_rust(
        name: str, width: int, height: int, pixels: np.ndarray
    ) -> "ExrChannel":
        return ExrChannel(
            name=name,
            width=width,
            height=height,
            pixels=pixels,
        )


@dataclass
class ExrLayer:
    name: str
    width: int
    height: int
    channels: list[ExrChannel]
    attributes: dict[str, Any]

    @staticmethod
    def _from_rust(rust_layer: RustLayer) -> "ExrLayer":
        name = rust_layer.name() or "unknown"

        width = rust_layer.width()
        assert width is not None

        height = rust_layer.height()
        assert height is not None

        channel_pixels = _pixels_from_layer(rust_layer)
        assert len(rust_layer.channels()) == len(
            channel_pixels
        ), f"expected {len(rust_layer.channels())} channels, got {len(channel_pixels)}"

        channels = [
            ExrChannel._from_rust(channel, width, height, pixels)
            for channel, pixels in zip(rust_layer.channels(), channel_pixels)
        ]

        return ExrLayer(
            name=name,
            width=width,
            height=height,
            channels=channels,
            attributes=rust_layer.attributes(),
        )


@dataclass
class ExrImage:
    layers: list[ExrLayer]
    attributes: dict[str, Any]

    @staticmethod
    def _from_rust(rust_image: RustImage) -> "ExrImage":
        return ExrImage(
            layers=[ExrLayer._from_rust(layer) for layer in rust_image.layers()],
            attributes=rust_image.attributes(),
        )


def load(buffer: BytesIO) -> ExrImage:
    return ExrImage._from_rust(RustImage.load_from_buffer(buffer.getvalue()))


def load_from_path(path: str) -> ExrImage:
    with open(path, "rb") as file:
        buffer = BytesIO(file.read())
        return load(buffer)
