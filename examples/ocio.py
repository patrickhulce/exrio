from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import PyOpenColorIO as OCIO
from scipy.ndimage import zoom

from exrio import Colorspace, ExrImage

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


def convert_to_acescg(pixels: np.ndarray, output_path: Path):
    ocio_config = OCIO.Config().CreateFromBuiltinConfig(ACES_CONFIG)
    from_transform = "ACES2065-1"
    to_transform = "ACEScg"

    acesscg_pixels = pixels.copy()
    processor = ocio_config.getProcessor(from_transform, to_transform)
    apply_transform(processor, acesscg_pixels)

    acesscg_image = ExrImage.from_pixels_ACEScg(acesscg_pixels)
    acesscg_image.to_path(output_path)
    write_thumbnail(acesscg_image, output_path, Colorspace.ACEScg)
    return acesscg_pixels


def convert_to_acescct(pixels: np.ndarray, output_path: Path):
    ocio_config = OCIO.Config().CreateFromBuiltinConfig(ACES_CONFIG)
    from_transform = "ACES2065-1"
    to_transform = "ACEScct"

    acesscct_pixels = pixels.copy()
    processor = ocio_config.getProcessor(from_transform, to_transform)
    apply_transform(processor, acesscct_pixels)

    acesscct_image = ExrImage.from_pixels_ACEScct(acesscct_pixels)
    acesscct_image.to_path(output_path)
    write_thumbnail(acesscct_image, output_path, Colorspace.ACEScct)
    return acesscct_pixels


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
    write_thumbnail(srgb_image, output_path, Colorspace.sRGB)
    return srgb_pixels


def plot_image_and_histogram(images_and_labels: list[tuple[np.ndarray, str]]):
    """Display multiple images and their histograms stacked vertically.

    Args:
        images_and_labels: List of (image_array, label) tuples to display
    """
    n_images = len(images_and_labels)
    fig, axes = plt.subplots(n_images, 2, figsize=(15, 5 * n_images))

    for idx, (pixels, label) in enumerate(images_and_labels):
        ax1, ax2 = axes[idx]

        # Plot the image
        ax1.imshow(pixels)
        ax1.set_title(f"{label} - Image")
        ax1.axis("off")

        # Plot histogram for each channel
        colors = ["red", "green", "blue"]
        for i, color in enumerate(colors):
            ax2.hist(
                pixels[:, :, i].ravel(),
                bins=256,
                range=(-0.1, 1),
                color=color,
                alpha=0.5,
                label=color.upper(),
            )

        ax2.set_title(f"{label} - Channel Distribution")
        ax2.set_xlabel("Pixel Value")
        ax2.set_ylabel("Frequency")
        ax2.legend()

    plt.tight_layout(h_pad=0.0)
    plt.show()


def write_thumbnail(image: ExrImage, original_path: Path, colorspace: Colorspace):
    """Create a 64x64 thumbnail version of an EXR image.

    Args:
        image: The ExrImage to create a thumbnail from
        original_path: Path to the original file, used to generate thumbnail path
        colorspace: The colorspace of the image ('ACEScg', 'ACEScct', or None for default)
    """
    pixels = image.to_pixels()
    # Ensure pixels are float32 before zooming
    pixels = pixels.astype(np.float32)

    height, width, channels = pixels.shape
    zoom_y = 64.0 / height
    zoom_x = 64.0 / width
    min_zoom = min(zoom_y, zoom_x)
    thumb_pixels = zoom(pixels, (min_zoom, min_zoom, 1), order=1)

    thumb_image = ExrImage.from_pixels(thumb_pixels, colorspace)
    thumb_path = original_path.parent / (original_path.stem + ".thumb.exr")
    thumb_image.to_path(thumb_path)


def main():
    examples_dir = Path(__file__).parent.parent / ".data" / "examples-v20241230"
    output_dir = Path(__file__).parent.parent / ".data" / "out"
    image_path = examples_dir / "ACES" / "DigitalLAD.2048x1556.exr"
    image = ExrImage.from_path(image_path)
    pixels = image.to_pixels().astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    image.to_path(output_dir / "out_original.exr")
    write_thumbnail(image, output_dir / "out_original.exr", Colorspace.ACES)

    acescg_pixels = convert_to_acescg(pixels, output_dir / "out_acescg.exr")
    acescct_pixels = convert_to_acescct(pixels, output_dir / "out_acescct.exr")
    srgb_pixels = convert_to_srgb(pixels, output_dir / "out_srgb.exr")
    plot_image_and_histogram(
        [
            (pixels, "ACES2065-1"),
            (acescg_pixels, "ACEScg"),
            (acescct_pixels, "ACEScct"),
            (srgb_pixels, "sRGB"),
        ]
    )


if __name__ == "__main__":
    main()
