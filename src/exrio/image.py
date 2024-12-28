from dataclasses import dataclass
from io import BytesIO
from typing import Any, Union

import numpy as np

from exrio._rust import ExrImage as RustImage
from exrio._rust import ExrLayer as RustLayer


def _pixels_from_layer(layer: RustLayer) -> list[np.ndarray]:
    pixels = layer.pixels()
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

    def _to_rust(self) -> RustLayer:
        layer = RustLayer(name=self.name)
        layer.with_width(self.width)
        layer.with_height(self.height)
        layer.with_attributes(self.attributes)
        for channel in self.channels:
            assert channel.pixels.dtype in [np.float16, np.float32, np.uint32]
            pixels = channel.pixels.flatten()
            layer.with_channel(channel=channel.name, pixels=pixels.copy(order="C"))
        return layer

    @staticmethod
    def _from_rust(rust_layer: RustLayer) -> "ExrLayer":
        name = rust_layer.name() or "unknown"

        width = rust_layer.width()
        assert width is not None

        height = rust_layer.height()
        assert height is not None

        channel_names = rust_layer.channels()
        channel_pixels = _pixels_from_layer(rust_layer)
        assert len(channel_names) == len(
            channel_pixels
        ), f"expected {len(channel_names)} channels, got {len(channel_pixels)}"

        channels = [
            ExrChannel._from_rust(channel, width, height, pixels)
            for channel, pixels in zip(channel_names, channel_pixels)
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

    def save(self) -> bytes:
        return self._to_rust().save_to_buffer()

    def _to_rust(self) -> RustImage:
        image = RustImage()
        image.with_attributes(self.attributes)
        for layer in self.layers:
            image.with_layer(layer._to_rust())
        return image

    @staticmethod
    def _from_rust(rust_image: RustImage) -> "ExrImage":
        return ExrImage(
            layers=[ExrLayer._from_rust(layer) for layer in rust_image.layers()],
            attributes=rust_image.attributes(),
        )


def load(buffer: Union[BytesIO, bytes]) -> ExrImage:
    if isinstance(buffer, bytes):
        buffer = BytesIO(buffer)
    return ExrImage._from_rust(RustImage.load_from_buffer(buffer.getvalue()))


def load_from_path(path: str) -> ExrImage:
    with open(path, "rb") as file:
        buffer = BytesIO(file.read())
        return load(buffer)
