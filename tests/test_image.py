from exrio.image import load_from_path


def test_load_from_path():
    image = load_from_path("tests/fixtures/AllHalfValues.exr")
    assert image.layers[0].channels[0].pixels.shape == (256, 256)
