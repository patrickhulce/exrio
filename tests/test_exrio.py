import tempfile

import numpy as np

from exrio._rust import ExrImage, ExrLayer


def test_basic_exr_roundtrip():
    test_data = np.array(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]],
        dtype=np.float32,
    )

    r_channel = test_data[:, :, 0]
    g_channel = test_data[:, :, 1]
    b_channel = test_data[:, :, 2]

    layer = ExrLayer("test_layer")
    layer.with_width(2)
    layer.with_height(2)
    layer.with_channel_f32("R", r_channel.reshape(-1).copy())
    layer.with_channel_f32("G", g_channel.reshape(-1).copy())
    layer.with_channel_f32("B", b_channel.reshape(-1).copy())

    image = ExrImage()
    image.with_layer(layer)
    test_metadata = {"test_attr": "test_value", "number": "42"}
    image.with_attributes(test_metadata)

    with tempfile.NamedTemporaryFile(suffix=".exr") as f:
        test_file = f.name
        image.save_to_path(test_file)

        read_image = ExrImage.load_from_path(test_file)
        read_layer = read_image.layers()[0]

    layer_attributes = read_layer.attributes()
    assert layer_attributes["test_attr"] == "test_value"
    assert layer_attributes["number"] == "42"

    assert read_layer.name() == "test_layer"
    assert read_layer.width() == 2
    assert read_layer.height() == 2

    read_red_channel = read_layer.pixels_f32()[2]  # Saved as BGR, not RGB
    np.testing.assert_array_almost_equal(
        read_red_channel, test_data[:, :, 0].reshape(-1)
    )
