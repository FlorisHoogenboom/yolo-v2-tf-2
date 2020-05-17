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

    # The yx coordinates should be centroids (hence end in 0.5)
    assert np.all(layer.base_anchor_boxes[..., 0:2] % 1 == 0.5)

    # Elements on the diagonals should be increasing stepwise
    diag_diffs = (
        layer.base_anchor_boxes[1:, 1:, :, 0:2] -
        layer.base_anchor_boxes[:-1, :-1, :, 0:2]
    )
    assert np.all(diag_diffs == 1)

    # Check that when moving in the y direction the y coordinate varies
    # this also checks that the order of the y and x coordinates is proper, i.e.
    # first the y axis and then the x axis
    y_diffs = (
        layer.base_anchor_boxes[1:, ..., 0] -
        layer.base_anchor_boxes[:-1, ..., 0]
    )
    assert np.all(y_diffs == 1)

    x_diffs = (
        layer.base_anchor_boxes[:, 1:, ..., 1] -
        layer.base_anchor_boxes[:, :-1, ..., 1]
    )
    assert np.all(x_diffs == 1)

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
    output = layer.compute_boxes(input)

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
    output = layer.compute_boxes(input)

    # With no stimulus, all predicted coordinates should be equal to the base
    # anchor boxes
    assert np.all((output - layer.base_anchor_boxes) == 0)


def test_anchor_layer_confidences():
    anchors = [
        [1, 1],
        [1, 1.5],
        [1.5, 1]
    ]
    batch_size = 5
    grid_height = 2
    grid_width = 3
    layer = layers.AnchorLayer(
        grid_height=grid_height,
        grid_width=grid_width,
        anchors=anchors,
        n_classes=7
    )

    input = np.random.randn(batch_size, grid_height, grid_width, 1024)
    output = layer.compute_confidences(input)

    assert (
        output.shape == (batch_size, grid_height, grid_width, len(anchors), 1)
    )

    # Confidences are probabilities
    assert np.all((0 <= output) & (output <= 1))

    # Each cell should output it's own probability
    assert (
        np.unique(output).shape ==
        (batch_size * grid_height * grid_width * len(anchors),)
    )

    # When fed with no stimulus all probabilities should resort to 0.5
    input = np.zeros((batch_size, grid_height, grid_width, 1024))
    output = layer.compute_confidences(input)
    assert np.all(output == 0.5)


def test_anchor_layer_classes():
    anchors = [
        [1, 1],
        [1, 1.5],
        [1.5, 1]
    ]
    batch_size = 5
    grid_height = 2
    grid_width = 3
    n_classes = 7
    layer = layers.AnchorLayer(
        grid_height=grid_height,
        grid_width=grid_width,
        anchors=anchors,
        n_classes=n_classes
    )

    input = np.random.randn(batch_size, grid_height, grid_width, 1024)
    output = layer.compute_classes(input)

    assert output.shape == (
        (batch_size, grid_height, grid_width, len(anchors), n_classes)
    )

    # The classes should be independent of the exact anchor box
    assert np.all(
        (output[..., 0, :] == output[..., 1, :]) &
        (output[..., 1, :] == output[..., 2, :])
    )

    # It should be probabilities over the last axis.
    np.testing.assert_almost_equal(output.numpy().sum(axis=-1), 1.0, decimal=5)


def test_anchor_layer_dimensions():
    pass


def test_anchor_layer_fed_with_zeros():
    pass
