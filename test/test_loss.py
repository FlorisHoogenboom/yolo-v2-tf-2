import numpy as np

from keras_yolo import loss

def test_compute_iou():
    predicted_boxes = np.array([[
        [1., 1., 3., 3.],
        [3., 3., 5., 5.],
        [7., 7., 8., 8.]
    ]])

    true_boxes = np.array([[
        [2., 2., 4., 4.],
        [1., 1., 3., 3.]
    ]])

    iou = loss.compute_iou(predicted_boxes, true_boxes)

    expected_iou = [[
        [1 / 7, 1],
        [1 / 7, 0],
        [0, 0]
    ]]

    np.testing.assert_array_almost_equal(iou, expected_iou, decimal=3)
