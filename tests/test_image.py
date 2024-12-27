import numpy as np

from exrio.image import ExrChannel, ExrImage, ExrLayer, load, load_from_path


def test_load_from_path():
    image = load_from_path("tests/fixtures/AllHalfValues.exr")
    assert image.layers[0].channels[0].pixels.shape == (256, 256)


def test_roundtrip():
    channel = ExrChannel(
        name="testc", width=256, height=256, pixels=np.zeros((256, 256))
    )
    layer = ExrLayer(
        name="testl", width=256, height=256, channels=[channel], attributes={}
    )
    image = ExrImage(layers=[layer], attributes={})

    buffer = image.save()
    rt_image = load(buffer)
    assert rt_image.layers[0].channels[0].pixels.shape == (256, 256)
    assert rt_image.layers[0].channels[0].name == "testc"
    assert rt_image.layers[0].name == "testl"
