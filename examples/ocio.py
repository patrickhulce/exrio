from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import PyOpenColorIO as OCIO
from scipy.ndimage import zoom

from exrio import Colorspace, ExrImage

ACES_CONFIG = "ocio://studio-config-v2.2.0_aces-v1.3_ocio-v2.4"


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
    ocio_config = OCIO.Config().CreateFromFile(ACES_CONFIG)
    from_transform = "ACES2065-1"
    to_transform = "ACEScg"

    acescg_pixels = pixels.copy()
    processor = ocio_config.getProcessor(from_transform, to_transform)
    apply_transform(processor, acescg_pixels)

    back_to_aces = acescg_pixels.copy()
    reverse_processor = ocio_config.getProcessor(to_transform, from_transform)
    apply_transform(reverse_processor, back_to_aces)

    acescg_image = ExrImage.from_pixels_ACEScg(acescg_pixels)
    acescg_image.to_path(output_path)
    write_thumbnail(acescg_image, output_path, Colorspace.ACEScg)

    return acescg_pixels, back_to_aces


def convert_to_acescct(pixels: np.ndarray, output_path: Path):
    ocio_config = OCIO.Config().CreateFromFile(ACES_CONFIG)
    from_transform = "ACES2065-1"
    to_transform = "ACEScct"

    acescct_pixels = pixels.copy()
    processor = ocio_config.getProcessor(from_transform, to_transform)
    apply_transform(processor, acescct_pixels)

    back_to_aces = acescct_pixels.copy()
    reverse_processor = ocio_config.getProcessor(to_transform, from_transform)
    apply_transform(reverse_processor, back_to_aces)

    acescct_image = ExrImage.from_pixels_ACEScct(acescct_pixels)
    acescct_image.to_path(output_path)
    write_thumbnail(acescct_image, output_path, Colorspace.ACEScct)

    return acescct_pixels, back_to_aces


def convert_to_srgb(pixels: np.ndarray, output_path: Path):
    ocio_config = OCIO.Config().CreateFromFile(ACES_CONFIG)
    from_transform = "ACES2065-1"
    to_transform = "sRGB Encoded Rec.709 (sRGB)"

    srgb_pixels = pixels.copy()
    processor = ocio_config.getProcessor(from_transform, to_transform)
    apply_transform(processor, srgb_pixels)

    back_to_aces = srgb_pixels.copy()
    reverse_processor = ocio_config.getProcessor(to_transform, from_transform)
    apply_transform(reverse_processor, back_to_aces)

    srgb_image = ExrImage.from_pixels(srgb_pixels)
    srgb_image.to_path(output_path)
    write_thumbnail(srgb_image, output_path, Colorspace.sRGB)

    return srgb_pixels, back_to_aces


def convert_to_srgb_rrt(pixels: np.ndarray, output_path: Path):
    ocio_config = OCIO.Config().CreateFromFile(ACES_CONFIG)
    from_transform = "ACES2065-1"
    display = "sRGB - Display"
    view = "ACES 1.0 - SDR Video"

    srgb_pixels = pixels.copy()
    processor = ocio_config.getProcessor(
        from_transform, display, view, OCIO.TRANSFORM_DIR_FORWARD
    )
    apply_transform(processor, srgb_pixels)

    back_to_aces = srgb_pixels.copy()
    reverse_processor = ocio_config.getProcessor(
        from_transform, display, view, OCIO.TRANSFORM_DIR_INVERSE
    )
    apply_transform(reverse_processor, back_to_aces)

    srgb_image = ExrImage.from_pixels(srgb_pixels)
    srgb_image.to_path(output_path)
    write_thumbnail(srgb_image, output_path, Colorspace.sRGB)

    return srgb_pixels, back_to_aces


def convert_to_srgb_linear(pixels: np.ndarray, output_path: Path):
    ocio_config = OCIO.Config().CreateFromFile(ACES_CONFIG)
    from_transform = "ACES2065-1"
    to_transform = "Linear Rec.709 (sRGB)"

    srgb_linear_pixels = pixels.copy()
    processor = ocio_config.getProcessor(from_transform, to_transform)
    apply_transform(processor, srgb_linear_pixels)

    back_to_aces = srgb_linear_pixels.copy()
    reverse_processor = ocio_config.getProcessor(to_transform, from_transform)
    apply_transform(reverse_processor, back_to_aces)

    srgb_linear_image = ExrImage.from_pixels(srgb_linear_pixels)
    srgb_linear_image.to_path(output_path)
    write_thumbnail(srgb_linear_image, output_path, Colorspace.LinearRGB)

    return srgb_linear_pixels, back_to_aces


def plot_image_and_histogram(
    images_and_labels: list[tuple[np.ndarray, np.ndarray, str]],
):
    """Display multiple image pairs and their histograms stacked vertically.

    Args:
        images_and_labels: List of (transformed_image, reverse_transformed_image, label) tuples
    """
    n_images = len(images_and_labels)
    fig, axes = plt.subplots(n_images, 3, figsize=(20, 5 * n_images))

    for idx, (transformed, reverse_transformed, label) in enumerate(images_and_labels):
        ax1, ax2, ax3 = axes[idx]

        # Plot the transformed image
        ax1.imshow(transformed)
        ax1.set_title(f"{label}")
        ax1.axis("off")

        # Plot the reverse-transformed image
        ax2.imshow(reverse_transformed)
        ax2.set_title(f"{label} (Back to ACES)")
        ax2.axis("off")

        # Plot histograms for just the transformed image
        colors = ["red", "green", "blue"]
        for i, color in enumerate(colors):
            # Transformed image histogram (solid lines)
            ax3.hist(
                transformed[:, :, i].ravel(),
                bins=256,
                range=(-0.1, 1),
                color=color,
                alpha=0.5,
                label=f"{color.upper()} transformed",
                histtype="step",
                linewidth=2,
            )

        ax3.set_title(f"{label} - Channel Distribution")
        ax3.set_xlabel("Pixel Value")
        ax3.set_ylabel("Frequency")
        ax3.legend()

    plt.tight_layout(h_pad=2.0)
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

    acescg = convert_to_acescg(pixels, output_dir / "out_acescg.exr")
    acescct = convert_to_acescct(pixels, output_dir / "out_acescct.exr")
    srgb = convert_to_srgb(pixels, output_dir / "out_srgb.exr")
    srgb_rrt = convert_to_srgb_rrt(pixels, output_dir / "out_srgb_rrt.exr")
    linear = convert_to_srgb_linear(pixels, output_dir / "out_srgb_linear.exr")
    plot_image_and_histogram(
        [
            (pixels, pixels, "ACES2065-1"),
            (*acescg, "ACEScg"),
            (*acescct, "ACEScct"),
            (*srgb, "sRGB"),
            (*srgb_rrt, "sRGB RRT"),
            (*linear, "sRGB Linear"),
        ]
    )


if __name__ == "__main__":
    main()
