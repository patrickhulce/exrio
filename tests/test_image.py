from typing import Optional

import numpy as np

from exrio.image import ExrChannel, ExrImage, ExrLayer, load, load_from_path


def _create_image(
    pixels: np.ndarray, attributes: Optional[dict[str, str]] = None
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
    image = load_from_path("tests/fixtures/AllHalfValues.exr")
    assert image.layers[0].channels[0].pixels.shape == (256, 256)
    assert image.layers[0].channels[0].pixels.dtype == np.float16


def test_roundtrip_f32():
    image = _create_image(np.zeros((256, 256), dtype=np.float32))

    buffer = image.save()
    rt_image = load(buffer)
    assert rt_image.layers[0].channels[0].pixels.dtype == np.float32
    assert rt_image.layers[0].channels[0].pixels.shape == (256, 256)


def test_roundtrip_f16():
    image = _create_image(np.zeros((320, 240), dtype=np.float16))
    buffer = image.save()
    rt_image = load(buffer)
    assert rt_image.layers[0].channels[0].pixels.dtype == np.float16
    assert rt_image.layers[0].channels[0].pixels.shape == (320, 240)


def test_roundtrip_u32():
    image = _create_image(np.zeros((320, 240), dtype=np.uint32))
    buffer = image.save()
    rt_image = load(buffer)
    assert rt_image.layers[0].channels[0].pixels.dtype == np.uint32
    assert rt_image.layers[0].channels[0].pixels.shape == (320, 240)
