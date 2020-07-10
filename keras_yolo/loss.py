import tensorflow as tf


def compute_iou(pred_coords, true_coords):
    """
    Calculates the IOU between all boxes in two collections.

    Args:
        pred_coords (tf.Tensor): Collection of predicted boxes. Should be
            encoded as corner coordinates and have shape
            ``(batch_size, n_predicted_boxes, 4)``
        true_coords (tf.Tensor): Collection of true boxes. Should be encoded as
            coordinates and have shape ``(batch_size, n_true_boxes, 4)``. This
            tensor might be zero padded.

    Returns:
        tf.Tensor: A tensor of shape
            ``(batch_size, n_predicted_boxes, n_true_boxes)``
    """
    inner_tl = tf.maximum(pred_coords[:, :, None, :2],
                          true_coords[:, None, :, :2])
    inner_br = tf.minimum(pred_coords[:, :, None, 2:],
                          true_coords[:, None, :, 2:])

    inner_hw = tf.maximum(inner_br - inner_tl, 0)
    intersection_area = inner_hw[..., 0] * inner_hw[..., 1]

    pred_hw = pred_coords[..., :2] - pred_coords[..., 2:]
    pred_area = pred_hw[..., 0] * pred_hw[..., 1]

    true_hw = true_coords[..., :2] - true_coords[..., 2:]
    true_area = true_hw[..., 0] * true_hw[..., 1]

    union_area = (
        pred_area[:, :, None] + true_area[:, None, :] - intersection_area
    )

    # We add a small epsilon to the denominator to prevent divisions by zero
    div_result = tf.truediv(intersection_area, union_area + 0.0001)

    return div_result


def compute_best_iou_mask(iou):
    """
    Computes a mask that assigns each true box to a best matching predicted box.

    This method uses a random pertubabition to resolve ties. This means that
    eacht truth box will get assigned only one predicted box.

    Args:
        iou (tf.Tensor): A tensor as returned by ``compute_iou``.

    Returns:
        tf.Tensor: A boolean tensor of the same format as the input tensor.
    """
    epsilon = 0.00001
    tie_broken_iou = iou + tf.random.normal(iou.shape, stddev=epsilon)

    largest_iou = tf.reduce_max(
        tie_broken_iou,
        axis=1,
        keepdims=True
    )

    return (tie_broken_iou == largest_iou) & (iou > 0)
