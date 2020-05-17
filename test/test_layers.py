import numpy as np

from keras_yolo import layers


def test_anchor_layer_base_anchor_boxes():
    anchors = [
        [1, 1],
        [1, 1.5],
        [1.5, 1]
    ]
    grid_height = 2
    grid_width = 3
    layer = layers.AnchorLayer(
        grid_height=grid_height,
        grid_width=grid_width,
        anchors=anchors,
        n_classes=7
    )

    # Check that the shape matches the anchor boxes and grid height/width
    assert (
        layer.base_anchor_boxes.shape ==
        (grid_height, grid_width, len(anchors), 4)
    )

    # The xy coordinates should be centroids (hence end in 0.5)
    assert np.all(layer.base_anchor_boxes[..., 0:2] % 1 == 0.5)

    # Elements on the diagonals should be increasing stepwise
    diag_diffs = (
        layer.base_anchor_boxes[1:, 1:, :, 0:2] -
        layer.base_anchor_boxes[:-1, :-1, :, 0:2]
    )
    assert np.all(diag_diffs == 1)

    # The WH part of every coordinate section should simply match the anchors
    assert np.all(
        layer.base_anchor_boxes[..., 2:] == anchors
    )


def test_anchor_layer_predicted_boxes():
    anchors = [
        [1, 1],
        [1, 1.5],
        [1.5, 1]
    ]
    grid_height = 2
    grid_width = 3
    layer = layers.AnchorLayer(
        grid_height=grid_height,
        grid_width=grid_width,
        anchors=anchors,
        n_classes=7
    )

    input = np.random.randn(5, grid_height, grid_width, 1024)
    output = layer.compute_predicted_boxes(input)

    assert output.shape == (5, grid_height, grid_width, len(anchors), 4)

    # Since we are working with coordinates on the grid all the predictions
    # should be positive
    assert np.all(output >= 0)

    # The differences between adjacent cells should be max 2 since that is the
    # maximum grid size
    diag_diffs = (
        output[:, 1:, 1:, :, 0:2] - output[:, :-1, :-1, :, 0:2]
    )
    assert np.all((0 <= diag_diffs) & (diag_diffs <= 2))

    input = np.zeros(((5, grid_height, grid_width, 1024)))
    output = layer.compute_predicted_boxes(input)

    # With no stimulus, all predicted coordinates should be equal to the base
    # anchor boxes
    assert np.all((output - layer.base_anchor_boxes) == 0)


def test_anchor_layer_dimensions():
    pass


def test_anchor_layer_fed_with_zeros():
    pass
