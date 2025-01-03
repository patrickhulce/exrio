import tempfile
from typing import Any, Optional

import numpy as np

from exrio.image import Colorspace, ExrChannel, ExrImage, ExrLayer, load


def _create_image(
    pixels: np.ndarray[Any, Any], attributes: Optional[dict[str, str]] = None
) -> ExrImage:
    width = pixels.shape[1]
    height = pixels.shape[0]
    channel = ExrChannel(name="testc", width=width, height=height, pixels=pixels)
    layer = ExrLayer(
        name="testl",
        width=width,
        height=height,
        channels=[channel],
        attributes=attributes or {},
    )
    return ExrImage(layers=[layer], attributes={})


def test_load_from_path():
    image = load("tests/fixtures/AllHalfValues.exr")
    assert image.layers[0].channels[0].pixels.shape == (256, 256)
    assert image.layers[0].channels[0].pixels.dtype == np.float16


def test_roundtrip_f32():
    image = _create_image(np.zeros((256, 256), dtype=np.float32))

    buffer = image.to_buffer()
    rt_image = load(buffer)
    assert rt_image.layers[0].channels[0].pixels.dtype == np.float32
    assert rt_image.layers[0].channels[0].pixels.shape == (256, 256)


def test_roundtrip_f16():
    image = _create_image(np.zeros((320, 240), dtype=np.float16))
    buffer = image.to_buffer()
    rt_image = load(buffer)
    assert rt_image.layers[0].channels[0].pixels.dtype == np.float16
    assert rt_image.layers[0].channels[0].pixels.shape == (320, 240)


def test_roundtrip_u32():
    image = _create_image(np.zeros((320, 240), dtype=np.uint32))
    buffer = image.to_buffer()
    rt_image = load(buffer)
    assert rt_image.layers[0].channels[0].pixels.dtype == np.uint32
    assert rt_image.layers[0].channels[0].pixels.shape == (320, 240)


def test_roundtrip_pixels():
    input_pixels = np.random.rand(320, 240, 3).astype(np.float32)
    image = load(input_pixels)
    assert image.layers[0].channels[0].pixels.dtype == np.float32
    assert image.layers[0].channels[0].pixels.shape == (320, 240)

    output_pixels = image.to_pixels()
    assert output_pixels.shape == (1, 320, 240, 3)
    assert output_pixels.dtype == np.float32

    np.testing.assert_allclose(output_pixels[0], input_pixels)


def test_roundtrip_pixels_with_layers():
    input_pixels = np.random.rand(3, 320, 240, 3).astype(np.float32)
    image = load(input_pixels)
    assert len(image.layers) == 3
    for layer in image.layers:
        assert layer.channels[0].pixels.dtype == np.float32
        assert layer.channels[0].pixels.shape == (320, 240)

    output_pixels = image.to_pixels()
    assert output_pixels.shape == (3, 320, 240, 3)
    assert output_pixels.dtype == np.float32

    np.testing.assert_allclose(output_pixels, input_pixels)


def test_roundtrip_pixels_with_layer_names():
    input_pixels = np.random.rand(3, 320, 240, 1).astype(np.float32)
    image = ExrImage.from_pixels(input_pixels, layer_names=["mask", "depth", "color"])
    assert len(image.layers) == 3
    for layer in image.layers:
        assert layer.name in ["mask", "depth", "color"]

    output_pixels = image.to_pixels()
    assert output_pixels.shape == (3, 320, 240, 1)
    assert output_pixels.dtype == np.float32

    buffer = image.to_buffer()
    rt_image = load(buffer)
    layer_names = [layer.name for layer in rt_image.layers]
    assert len(rt_image.layers) == 3
    assert layer_names == ["mask", "depth", "color"]
    for i, layer in enumerate(rt_image.layers):
        assert layer.channels[0].pixels.dtype == np.float32
        assert layer.channels[0].pixels.shape == (320, 240)
        np.testing.assert_allclose(layer.channels[0].pixels, input_pixels[i, :, :, 0])


def test_roundtrip_chromaticities():
    image = load("tests/fixtures/ACES-2065-1.exr")
    assert image.chromaticities is not None
    assert image.chromaticities.red == (0.7347, 0.2653)
    assert image.chromaticities.green == (0.0000, 1.0000)
    assert image.chromaticities.blue == (0.0001, -0.0770)
    assert image.chromaticities.white == (0.32168, 0.33767)

    with tempfile.NamedTemporaryFile(suffix=".exr") as f:
        image.chromaticities.red = (0.5, 0.5)
        image.to_path(f.name)
        image_out = load(f.name)

        assert image_out.chromaticities is not None
        assert image_out.chromaticities.red == (0.5, 0.5)
        assert image_out.chromaticities.blue == (0.0001, -0.0770)


def test_infer_colorspace():
    image = load("tests/fixtures/ACES-2065-1.exr")
    assert image.inferred_colorspace == Colorspace.ACES

    image = load("tests/fixtures/ACEScg.exr")
    assert image.inferred_colorspace == Colorspace.ACEScg

    image = load("tests/fixtures/ACEScct.exr")
    assert image.inferred_colorspace == Colorspace.ACEScct

    image = load("tests/fixtures/sRGB.exr")
    assert image.inferred_colorspace == Colorspace.sRGB

    image = load("tests/fixtures/AllHalfValues.exr")
    assert image.inferred_colorspace is None


def test_infer_ACEScc_from_exrio_attribute():
    image = ExrImage.from_pixels_ACEScc(np.zeros((256, 256, 3), dtype=np.float32))
    assert image.inferred_colorspace == Colorspace.ACEScc

    buffer = image.to_buffer()
    image_out = load(buffer)
    assert image_out.inferred_colorspace == Colorspace.ACEScc


def test_infer_LinearRGB_from_exrio_attribute():
    image = ExrImage.from_pixels_LinearRGB(np.zeros((256, 256, 3), dtype=np.float32))
    assert image.inferred_colorspace == Colorspace.LinearRGB

    buffer = image.to_buffer()
    image_out = load(buffer)
    assert image_out.inferred_colorspace == Colorspace.LinearRGB
