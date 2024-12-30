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

    def to_pixels(self) -> np.ndarray:
        num_layers = len(self.layers)
        assert num_layers == 1, f"ambiguous reference, image has {num_layers} layers"
        layer = self.layers[0]
        channel_pixels = [
            channel.pixels.reshape(layer.height, layer.width)
            for channel in layer.channels
        ]
        return np.stack(channel_pixels, axis=-1)

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

    @staticmethod
    def from_buffer(buffer: Union[BytesIO, bytes]) -> "ExrImage":
        if isinstance(buffer, BytesIO):
            buffer = buffer.getvalue()
        return ExrImage._from_rust(RustImage.load_from_buffer(buffer))

    @staticmethod
    def from_path(path: Union[str, Path]) -> "ExrImage":
        with open(path, "rb") as file:
            buffer = BytesIO(file.read())
            return ExrImage.from_buffer(buffer)

    @staticmethod
    def _from_pixels(
        pixels: np.ndarray,
        chromaticities: Chromaticities,
    ) -> "ExrImage":
        debug_msg = f"expected float16, float32, or uint32, got {pixels.dtype}"
        assert pixels.dtype in [np.float16, np.float32, np.uint32], debug_msg

        height, width, channel_count = pixels.shape
        assert channel_count in [3, 4], f"expected 3 or 4 channels, got {channel_count}"

        channels: list[ExrChannel] = []
        channel_names = "RGBA"[:channel_count]
        for idx, channel_name in enumerate(channel_names):
            channels.append(
                ExrChannel(
                    name=channel_name,
                    width=width,
                    height=height,
                    pixels=pixels[..., idx].flatten(),
                )
            )
        layer = ExrLayer(
            width=width,
            height=height,
            channels=channels,
            chromaticities=chromaticities,
        )

        return ExrImage(layers=[layer], attributes={"Aces Image Container Flag": 1})

    @staticmethod
    def from_pixels_aces(pixels: np.ndarray) -> "ExrImage":
        """
        Creates an EXR image from a set of RGB/RGBA ACES2065-1 pixels in float16/float32 HWC layout.

        The output image will have a single layer with those RGB/RGBA channels,
        the ACES container attribute, and the chromaticity values of the AP0 primaries.

        @see https://pub.smpte.org/pub/st2065-1/st2065-1-2021.pdf
        """
        assert pixels.dtype in [np.float16, np.float32]
        return ExrImage._from_pixels(pixels, PRIMARY_CHROMATICITIES["AP0"])

    @staticmethod
    def from_pixels_acescg(pixels: np.ndarray) -> "ExrImage":
        """
        Creates an EXR image from a set of RGB/RGBA ACEScg pixels in float16/float32 HWC layout.

        The output image will have a single layer with those RGB/RGBA channels,
        the ACEScg container attribute, and the chromaticity values of the AP1 primaries.

        @see https://docs.acescentral.com/specifications/acescg/
        """
        assert pixels.dtype in [np.float16, np.float32]
        return ExrImage._from_pixels(pixels, PRIMARY_CHROMATICITIES["AP1"])

    @staticmethod
    def from_pixels_acescct(pixels: np.ndarray) -> "ExrImage":
        """
        Creates an EXR image from a set of RGB/RGBA ACEScct pixels in float16/float32 HWC layout.

        The output image will have a single layer with those RGB/RGBA channels,
        the ACEScct container attribute, and the chromaticity values of the AP1 primaries.

        @see https://docs.acescentral.com/specifications/acescct/
        """
        assert pixels.dtype in [np.float16, np.float32]
        return ExrImage._from_pixels(pixels, PRIMARY_CHROMATICITIES["AP1"])

    @staticmethod
    def from_pixels_srgb(pixels: np.ndarray) -> "ExrImage":
        """
        Creates an EXR image from a set of RGB/RGBA sRGB pixels in HWC layout.

        The output image will have a single layer with those RGB/RGBA channels,
        the sRGB container attribute, and the chromaticity values of the sRGB primaries.

        @see https://www.color.org/chardata/rgb/srgb.xalter
        """
        if pixels.dtype == np.uint8:
            pixels = pixels.astype(np.uint32)
        return ExrImage._from_pixels(pixels, PRIMARY_CHROMATICITIES["sRGB"])

    @staticmethod
    def from_pixels(
        pixels: np.ndarray, colorspace: Colorspace = Colorspace.sRGB
    ) -> "ExrImage":
        if colorspace == Colorspace.ACES:
            return ExrImage.from_pixels_aces(pixels)
        elif colorspace == Colorspace.ACEScg:
            return ExrImage.from_pixels_acescg(pixels)
        elif colorspace == Colorspace.ACEScct:
            return ExrImage.from_pixels_acescct(pixels)
        elif colorspace == Colorspace.sRGB:
            return ExrImage.from_pixels_srgb(pixels)
        else:
            raise ValueError(f"Unsupported colorspace: {colorspace}")


def load(path_or_buffer: Union[BytesIO, bytes, str, Path, np.ndarray]) -> ExrImage:
    if isinstance(path_or_buffer, np.ndarray):
        return ExrImage.from_pixels(path_or_buffer)
    elif isinstance(path_or_buffer, str) or isinstance(path_or_buffer, Path):
        return ExrImage.from_path(path_or_buffer)
    elif isinstance(path_or_buffer, bytes) or isinstance(path_or_buffer, BytesIO):
        return ExrImage.from_buffer(path_or_buffer)
    else:
        raise ValueError(f"Unsupported type: {type(path_or_buffer)}")
