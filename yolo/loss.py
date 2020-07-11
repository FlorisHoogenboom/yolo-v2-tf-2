import tensorflow as tf

from yolo import utils


IOU_THRESHOLD = 0.6
COORD_LOSS_WEIGHT = 5
NEG_CONF_LOSS_WEIGHT = 0.5


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


def warmup_loss(y_pred, network):
    """
    Loss to warmup the network for training. This loss function pushes the
    bounding boxes to their positions and dimensions based on the anchors.

    Args:
        y_pred (tf.Tensor or list of tf.Tensor): The output(s) of the Yolo
            network.
        network (keras_yolo.model.Yolo): Instance of the network that produced
            ``y_pred``.

    Returns:
        tf.Tensor: Scalar tensor representing the total loss.
    """
    predicted_boxes, base_boxes = (
        utils.get_base_and_predicted_boxes(y_pred, network)
    )

    xy_warmup_loss = tf.reduce_sum(
        tf.square(base_boxes[..., 0:2] - predicted_boxes[..., 0:2])
    )

    wh_warmup_loss = tf.reduce_sum(
        tf.square(tf.sqrt(base_boxes[..., 0:2]) -
                  tf.sqrt(predicted_boxes[..., 0:2]))
    )

    warmup_loss = xy_warmup_loss + wh_warmup_loss

    return warmup_loss


def loss(y_pred, y_true, network):
    """

    Args:
        y_pred (tf.Tensor or list of tf.Tensor): The output(s) of the Yolo
            network.
        y_true (tf.Tensor): A tensor of shape
            ``(batch_size, max_train_boxes, 4)``
        network (keras_yolo.model.Yolo): Instance of the network that produced
            ``y_pred``.

    Returns:
        tf.Tensor: Scalar tensor representing the total loss.
    """
    predicted_boxes, base_boxes = (
        utils.get_base_and_predicted_boxes(y_pred, network)
    )

    # Convert all boxes to actual coordinates
    pred_coords = utils.boxes_to_coords(predicted_boxes)
    base_coords = utils.boxes_to_coords(base_boxes)
    true_coords = utils.boxes_to_coords(y_true)

    # Compute the IOU masks
    base_boxes_iou = compute_iou(base_coords, true_coords)
    pred_boxes_iou = compute_iou(pred_coords, true_coords)

    # Derive the loss masks from the IOU masks
    base_boxes_mask = compute_best_iou_mask(base_boxes_iou)
    conf_mask = base_boxes_mask | (pred_boxes_iou > IOU_THRESHOLD)

    # Compute the actual loss components
    xy_loss = tf.reduce_mean(
        tf.boolean_mask(
            tf.square(
                predicted_boxes[:, :, None, 0:2] - y_true[:, None, :, 0:2]
            ),
            base_boxes_mask
        )
    )

    wh_loss = tf.reduce_mean(
        tf.boolean_mask(
            tf.square(
                tf.sqrt(predicted_boxes[:, :, None, 2:4]) -
                tf.sqrt(y_true[:, None, :, 2:4])
            ),
            base_boxes_mask
        )
    )

    positive_conf_loss = tf.reduce_mean(
        tf.boolean_mask(
            tf.square(
                predicted_boxes[..., 4, None] - pred_boxes_iou
            ),
            conf_mask
        )
    )

    negative_conf_loss = tf.reduce_mean(
        tf.boolean_mask(
            tf.square(
                predicted_boxes[..., 4, None]
            ),
            ~(tf.reduce_any(conf_mask, axis=-1, keepdims=True))
        )
    )

    return (
            COORD_LOSS_WEIGHT * (xy_loss + wh_loss) +
            positive_conf_loss +
            NEG_CONF_LOSS_WEIGHT * negative_conf_loss
            # TODO: Class loss should still be implemented.
    )
