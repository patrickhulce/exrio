import json
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from exrio._rust import ExrImage as RustImage
from exrio._rust import ExrLayer as RustLayer

ACES_IMAGE_CONTAINER_FLAG = "acesImageContainerFlag"
EXRIO_COLORSPACE_KEY = "py/exrio/Colorspace"


def _pixels_from_layer(layer: RustLayer) -> list[np.ndarray]:
    pixels = layer.pixels()
    assert pixels is not None
    return [
        pixels[i].reshape(layer.height(), layer.width()) for i in range(len(pixels))
    ]


class Colorspace(str, Enum):
    sRGB = "sRGB"
    LinearRGB = "Linear Rec.709 (sRGB)"
    ACES = "ACES 2065-1"
    ACEScg = "ACEScg"
    ACEScc = "ACEScc"
    ACEScct = "ACEScct"

    @staticmethod
    def from_dict(dict: dict[str, Any]) -> Optional["Colorspace"]:
        if EXRIO_COLORSPACE_KEY not in dict:
            return None
        value = dict[EXRIO_COLORSPACE_KEY]
        if not isinstance(value, str):
            return None

        return next((color for color in Colorspace if color.value == value), None)


@dataclass
class Chromaticities:
    red: tuple[float, float] = field(default=(0.64, 0.33))
    green: tuple[float, float] = field(default=(0.3, 0.6))
    blue: tuple[float, float] = field(default=(0.15, 0.06))
    white: tuple[float, float] = field(default=(0.3127, 0.329))

    def to_list(self) -> list[float]:
        return [*self.red, *self.green, *self.blue, *self.white]

    def is_close_to(self, other: "Chromaticities") -> bool:
        return np.allclose(self.to_list(), other.to_list(), atol=1e-3)

    @staticmethod
    def _from_rust(chromaticities: str) -> "Chromaticities":
        assert chromaticities.startswith("chroma:{")
        parsed = json.loads(chromaticities.removeprefix("chroma:"))
        assert isinstance(parsed, dict)
        assert "values_f32" in parsed
        values = parsed["values_f32"]
        assert len(values) == 8
        return Chromaticities(
            red=(values[0], values[1]),
            green=(values[2], values[3]),
            blue=(values[4], values[5]),
            white=(values[6], values[7]),
        )

    def _to_rust(self) -> str:
        values_dict = {"values_f32": self.to_list()}
        return f"chroma:{json.dumps(values_dict)}"


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
    attributes: dict[str, Any] = field(default_factory=dict)
    chromaticities: Optional[Chromaticities] = None

    @property
    def inferred_colorspace(self) -> Optional[Colorspace]:
        """
        Makes a best-effort guess at the colorspace of the image using the exrio-specific
        colorspace attribute, the chromaticities attribute, and the maximum pixel value.

        This is not fool-proof and should not be a substitute for proper colorspace tagging.
        But it can be useful for dealing with sources outside of your control.
        """

        image_colorspace = Colorspace.from_dict(self.attributes)
        if image_colorspace is not None:
            return image_colorspace

        layer_colorspaces = [
            Colorspace.from_dict(layer.attributes) for layer in self.layers
        ]
        if any(layer_colorspaces):
            return next(color for color in layer_colorspaces if color is not None)

        if self.chromaticities is None:
            return None

        if self.chromaticities.is_close_to(PRIMARY_CHROMATICITIES["sRGB"]):
            # We can't easily determine if the image is sRGB or LinearRGB,
            # so we'll default to the more common sRGB.
            return Colorspace.sRGB
        elif self.chromaticities.is_close_to(PRIMARY_CHROMATICITIES["AP0"]):
            # Thankfully, ACES2065-1 is the only supported colorspace that uses the AP0 primaries.
            return Colorspace.ACES
        elif self.chromaticities.is_close_to(PRIMARY_CHROMATICITIES["AP1"]):
            if not self.first_layer:
                return None

            pixel_arrays = [c.pixels for c in self.first_layer.channels]
            max_pixel = max([np.max(pixels) for pixels in pixel_arrays], default=0.0)
            if max_pixel > 1.0:
                # ACEScc/cct are log-encoded with the vast majority of values being <1.0 (~222 in Linear AP1).
                # ACEScg is linear-encoded with any highlights easily being >1.0.
                # If the maximum pixel value is >1.0, we'll assume it's ACEScg.
                return Colorspace.ACEScg
            else:
                # It's not really possible to determine if the image is ACEScc or ACEScct,
                # so we'll just pick the one more common in VFX.
                return Colorspace.ACEScct

        return None

    @property
    def first_layer(self) -> Optional[ExrLayer]:
        if len(self.layers) == 0:
            return None
        return self.layers[0]

    def to_buffer(self) -> bytes:
        return self._to_rust().save_to_buffer()

    def to_path(self, path: Union[str, Path]) -> None:
        with open(path, "wb") as file:
            file.write(self.to_buffer())

    def to_pixels(self) -> np.ndarray:
        num_layers = len(self.layers)
        assert num_layers == 1, f"ambiguous reference, image has {num_layers} layers"
        layer = self.layers[0]
        channel_names = set([c.name for c in layer.channels])
        channel_pixels = {
            channel.name: channel.pixels.reshape(layer.height, layer.width)
            for channel in layer.channels
        }

        rgb_pixels = []
        for channel in ["R", "G", "B"]:
            assert (
                channel in channel_names
            ), f"missing {channel} channel, {channel_names} available"
            rgb_pixels.append(channel_pixels[channel])

        if "A" in channel_names:
            rgb_pixels.append(channel_pixels["A"])

        return np.stack(rgb_pixels, axis=-1)

    def _to_rust(self) -> RustImage:
        image = RustImage()

        attributes = self.attributes.copy()
        if self.chromaticities is not None:
            attributes["chromaticities"] = self.chromaticities._to_rust()
        image.with_attributes(attributes)

        for layer in self.layers:
            image.with_layer(layer._to_rust())

        return image

    @staticmethod
    def _from_rust(rust_image: RustImage) -> "ExrImage":
        attributes = rust_image.attributes()
        chromaticities = attributes.get("chromaticities")
        if chromaticities is not None:
            chromaticities = Chromaticities._from_rust(chromaticities)
        return ExrImage(
            layers=[ExrLayer._from_rust(layer) for layer in rust_image.layers()],
            attributes=attributes,
            chromaticities=chromaticities,
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
        )

        return ExrImage(layers=[layer], chromaticities=chromaticities)

    @staticmethod
    def from_pixels_ACES(pixels: np.ndarray) -> "ExrImage":
        """
        Creates an EXR image from a set of RGB/RGBA ACES2065-1 pixels in float16/float32 HWC layout.

        The output image will have a single layer with those RGB/RGBA channels,
        the ACES container attribute, and the chromaticity values of the AP0 primaries.

        @see https://pub.smpte.org/pub/st2065-1/st2065-1-2021.pdf
        """
        assert pixels.dtype in [np.float16, np.float32]
        image = ExrImage._from_pixels(pixels, PRIMARY_CHROMATICITIES["AP0"])
        image.attributes[ACES_IMAGE_CONTAINER_FLAG] = 1
        image.attributes[EXRIO_COLORSPACE_KEY] = Colorspace.ACES
        return image

    @staticmethod
    def from_pixels_ACEScg(pixels: np.ndarray) -> "ExrImage":
        """
        Creates an EXR image from a set of RGB/RGBA ACEScg pixels in float16/float32 HWC layout.

        The output image will have a single layer with those RGB/RGBA channels,
        and the chromaticity values of the AP1 primaries.

        @see https://docs.acescentral.com/specifications/acescg/
        """
        assert pixels.dtype in [np.float16, np.float32]
        image = ExrImage._from_pixels(pixels, PRIMARY_CHROMATICITIES["AP1"])
        image.attributes[EXRIO_COLORSPACE_KEY] = Colorspace.ACEScg
        return image

    @staticmethod
    def from_pixels_ACEScc(pixels: np.ndarray) -> "ExrImage":
        """
        Creates an EXR image from a set of RGB/RGBA ACEScc pixels in float16/float32 HWC layout.

        The output image will have a single layer with those RGB/RGBA channels,
        and the chromaticity values of the AP1 primaries.

        @see https://docs.acescentral.com/specifications/acescc/
        """
        assert pixels.dtype in [np.float16, np.float32]
        image = ExrImage._from_pixels(pixels, PRIMARY_CHROMATICITIES["AP1"])
        image.attributes[EXRIO_COLORSPACE_KEY] = Colorspace.ACEScc
        return image

    @staticmethod
    def from_pixels_ACEScct(pixels: np.ndarray) -> "ExrImage":
        """
        Creates an EXR image from a set of RGB/RGBA ACEScct pixels in float16/float32 HWC layout.

        The output image will have a single layer with those RGB/RGBA channels,
        and the chromaticity values of the AP1 primaries.

        @see https://docs.acescentral.com/specifications/acescct/
        """
        assert pixels.dtype in [np.float16, np.float32]
        image = ExrImage._from_pixels(pixels, PRIMARY_CHROMATICITIES["AP1"])
        image.attributes[EXRIO_COLORSPACE_KEY] = Colorspace.ACEScct
        return image

    @staticmethod
    def from_pixels_sRGB(pixels: np.ndarray) -> "ExrImage":
        """
        Creates an EXR image from a set of RGB/RGBA sRGB pixels in HWC layout.

        The output image will have a single layer with those RGB/RGBA channels,
        and the chromaticity values of the sRGB/Rec.709 primaries.

        @see https://www.color.org/chardata/rgb/srgb.xalter
        """
        if pixels.dtype == np.uint8:
            pixels = pixels.astype(np.float16) / 255.0
        image = ExrImage._from_pixels(pixels, PRIMARY_CHROMATICITIES["sRGB"])
        image.attributes[EXRIO_COLORSPACE_KEY] = Colorspace.sRGB
        return image

    @staticmethod
    def from_pixels_LinearRGB(pixels: np.ndarray) -> "ExrImage":
        """
        Creates an EXR image from a set of RGB/RGBA Linear RGB pixels in HWC layout.

        The output image will have a single layer with those RGB/RGBA channels,
        and the chromaticity values of the sRGB/Rec.709 primaries.

        @see https://facelessuser.github.io/coloraide/colors/srgb_linear/
        """
        assert pixels.dtype in [np.float16, np.float32]
        image = ExrImage._from_pixels(pixels, PRIMARY_CHROMATICITIES["sRGB"])
        image.attributes[EXRIO_COLORSPACE_KEY] = Colorspace.LinearRGB
        return image

    @staticmethod
    def from_pixels(
        pixels: np.ndarray, colorspace: Colorspace = Colorspace.sRGB
    ) -> "ExrImage":
        if colorspace == Colorspace.ACES:
            return ExrImage.from_pixels_ACES(pixels)
        elif colorspace == Colorspace.ACEScg:
            return ExrImage.from_pixels_ACEScg(pixels)
        elif colorspace == Colorspace.ACEScct:
            return ExrImage.from_pixels_ACEScct(pixels)
        elif colorspace == Colorspace.ACEScc:
            return ExrImage.from_pixels_ACEScc(pixels)
        elif colorspace == Colorspace.sRGB:
            return ExrImage.from_pixels_sRGB(pixels)
        elif colorspace == Colorspace.LinearRGB:
            return ExrImage.from_pixels_LinearRGB(pixels)
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
