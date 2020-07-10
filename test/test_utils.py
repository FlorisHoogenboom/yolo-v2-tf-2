import numpy as np

from keras_yolo import utils
from keras_yolo import layers


def test_flatten_anchor_boxes():
    grid_height = 3
    grid_width = 4
    batch_size = 2
    n_classes = 5
    anchors = [[1., 2.], [2., 1.]]

    anchor_layer = layers.AnchorLayer(
        grid_height,
        grid_width,
        anchors=anchors,
        n_classes=n_classes
    )

    X = np.random.randn(batch_size, grid_height, grid_width, 15)
    grid_boxes = anchor_layer(X)

    # First run tests with confidence and classes
    flattend_boxes = utils.flatten_anchor_boxes(
        grid_boxes,
        anchor_layer,
        include_conf=True,
        include_classes=True
    )

    assert flattend_boxes.shape == (batch_size,
                                    grid_height * grid_width * len(anchors),
                                    4 + 1 + n_classes)

    np.testing.assert_array_equal(flattend_boxes[0, 0], grid_boxes[0, 0, 0, 0])

    # Next, run the same tests but not including confidency and classes
    flattend_boxes = utils.flatten_anchor_boxes(
        grid_boxes[..., 0:4],
        anchor_layer,
        include_conf=False,
        include_classes=False
    )

    assert flattend_boxes.shape == (batch_size,
                                    grid_height * grid_width * len(anchors),
                                    4)

    np.testing.assert_array_equal(
        flattend_boxes[0, 0],
        grid_boxes[0, 0, 0, 0, 0:4]
    )


def test_boxes_to_coords():
    boxes = np.array([
        [1.0, 2.0, 2.0, 1.0]
    ])

    coords = utils.boxes_to_coords(boxes)
    expected_coords = np.array([
        [0.0, 1.5, 2.0, 2.5]
    ])

    np.testing.assert_array_equal(coords, expected_coords)
