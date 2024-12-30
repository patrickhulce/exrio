from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from exrio._rust import ExrImage as RustImage
from exrio._rust import ExrLayer as RustLayer


def _pixels_from_layer(layer: RustLayer) -> list[np.ndarray]:
    pixels = layer.pixels()
    assert pixels is not None
    return [
        pixels[i].reshape(layer.height(), layer.width()) for i in range(len(pixels))
    ]


class Colorspace(str, Enum):
    sRGB = "sRGB"
    ACES = "ACES 2065-1"
    ACEScg = "ACEScg"
    ACEScct = "ACEScct"


@dataclass
class Chromaticities:
    red: tuple[float, float]
    green: tuple[float, float]
    blue: tuple[float, float]
    white: tuple[float, float]


PRIMARY_CHROMATICITIES = {
    # https://pub.smpte.org/pub/st2065-1/st2065-1-2021.pdf
    "AP0": Chromaticities(
        red=(0.7347, 0.2653),
        green=(0.0000, 1.0000),
        blue=(0.0001, -0.0770),
        white=(0.32168, 0.33767),
    ),
    # https://docs.acescentral.com/specifications/acescg/
    # https://docs.acescentral.com/specifications/acescct/
    "AP1": Chromaticities(
        red=(0.713, 0.293),
        green=(0.165, 0.830),
        blue=(0.128, 0.044),
        white=(0.32168, 0.33767),
    ),
    # https://www.color.org/chardata/rgb/srgb.xalter
    "sRGB": Chromaticities(
        red=(0.64, 0.33),
        green=(0.3, 0.6),
        blue=(0.15, 0.06),
        white=(0.3127, 0.329),
    ),
}


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
    width: int
    height: int
    channels: list[ExrChannel]
    name: Optional[str] = None
    attributes: dict[str, Any] = field(default_factory=dict)
    chromaticities: Optional[Chromaticities] = None

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
            name=rust_layer.name(),
            width=width,
            height=height,
            channels=channels,
            attributes=rust_layer.attributes(),
        )


@dataclass
class ExrImage:
    layers: list[ExrLayer]
    attributes: dict[str, Any]

    def to_buffer(self) -> bytes:
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


def load(path_or_buffer: Union[BytesIO, bytes, str, Path]) -> ExrImage:
    if isinstance(path_or_buffer, str) or isinstance(path_or_buffer, Path):
        return load_from_path(path_or_buffer)
    elif isinstance(path_or_buffer, bytes) or isinstance(path_or_buffer, BytesIO):
        return load_from_buffer(path_or_buffer)
    else:
        raise ValueError(f"Unsupported type: {type(path_or_buffer)}")


def load_from_buffer(buffer: Union[BytesIO, bytes]) -> ExrImage:
    if isinstance(buffer, bytes):
        buffer = BytesIO(buffer)
    return ExrImage._from_rust(RustImage.load_from_buffer(buffer.getvalue()))


def load_from_path(path: Union[str, Path]) -> ExrImage:
    with open(path, "rb") as file:
        buffer = BytesIO(file.read())
        return load(buffer)
