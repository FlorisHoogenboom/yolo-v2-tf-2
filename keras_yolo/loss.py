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
