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

    expected_iou = np.array([[
        [1 / 7, 1],
        [1 / 7, 0],
        [0, 0]
    ]])

    np.testing.assert_array_almost_equal(iou, expected_iou, decimal=3)


def test_compute_best_iou():
    iou = np.array([[
        [1 / 7, 1],
        [1, 0],
        [0, 0]
    ]])

    expected_mask = np.array([[
        [False, True],
        [True, False],
        [False, False]
    ]])

    best_iou_mask = loss.compute_best_iou_mask(iou)

    np.testing.assert_array_equal(best_iou_mask, expected_mask)
