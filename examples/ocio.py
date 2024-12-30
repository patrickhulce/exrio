from pathlib import Path

import numpy as np
import PyOpenColorIO as OCIO

from exrio import ExrImage

ACES_CONFIG = "studio-config-v1.0.0_aces-v1.3_ocio-v2.1"


def apply_transform(processor: OCIO.Processor, pixels: np.ndarray):
    if pixels.dtype.name != "float32":
        raise ValueError("Image must be float32 not " + pixels.dtype.name)
    cpu = processor.getDefaultCPUProcessor()
    _, _, channels = pixels.shape
    if channels == 3:
        cpu.applyRGB(pixels)
    if channels == 4:
        cpu.applyRGBA(pixels)


def convert_to_acescct(pixels: np.ndarray, output_path: Path):
    ocio_config = OCIO.Config().CreateFromBuiltinConfig(ACES_CONFIG)
    from_transform = "ACES2065-1"
    to_transform = "ACEScct"

    acesscct_pixels = pixels.copy()
    processor = ocio_config.getProcessor(from_transform, to_transform)
    apply_transform(processor, acesscct_pixels)

    acesscct_image = ExrImage.from_pixels_ACEScct(acesscct_pixels)
    acesscct_image.to_path(output_path)


def convert_to_srgb(pixels: np.ndarray, output_path: Path):
    ocio_config = OCIO.Config().CreateFromBuiltinConfig(ACES_CONFIG)
    from_transform = "ACES2065-1"
    to_transform = "sRGB - Display"
    view = "ACES 1.0 - SDR Video"

    srgb_pixels = pixels.copy()
    processor = ocio_config.getProcessor(
        from_transform, to_transform, view, OCIO.TRANSFORM_DIR_FORWARD
    )
    apply_transform(processor, srgb_pixels)

    srgb_image = ExrImage.from_pixels(srgb_pixels)
    srgb_image.to_path(output_path)


def main():
    examples_dir = Path(__file__).parent.parent / ".data" / "examples-v20241230"
    output_dir = Path(__file__).parent.parent / ".data" / "out"
    image_path = examples_dir / "ACES" / "DigitalLAD.2048x1556.exr"
    image = ExrImage.from_path(image_path)
    pixels = image.to_pixels().astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    image.to_path(output_dir / "out_original.exr")
    convert_to_acescct(pixels, output_dir / "out_acescct.exr")
    convert_to_srgb(pixels, output_dir / "out_srgb.exr")


if __name__ == "__main__":
    main()
