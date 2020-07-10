import tensorflow as tf

from keras_yolo import utils


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


def get_base_and_predicted_boxes(network_output, network):
    """

    Args:
        network_output (list of tf.Tensor): A list of output tensors of the
            anchor heads in the Yolo network.
        network (keras_yolo.model.Yolo): An instance of the Yolo network

    Returns:
        tf.Tensor: A tensor with shape
            ``(batch_size, n_total_boxes, 4 + 1 + n_classes)``
    """
    if type(network_output) is not list:
        network_output = [network_output]

    predicted_boxes = []
    base_boxes = []
    for anchor_output, anchor_head in zip(network_output, network.anchor_heads):
        predicted_boxes.append(
            utils.flatten_anchor_boxes(
                anchor_output,
                anchor_head,
                include_conf=True,
                include_classes=True
            )
        )

        base_boxes.append(
            utils.flatten_anchor_boxes(
                anchor_head.base_anchor_boxes,
                anchor_head,
                include_conf=False,
                include_classes=False
            )
        )

    predicted_boxes = tf.concat(predicted_boxes, axis=1)
    base_boxes = tf.concat(base_boxes, axis=1)

    return predicted_boxes, base_boxes
