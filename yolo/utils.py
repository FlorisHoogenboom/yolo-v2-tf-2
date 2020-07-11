import numpy as np
import tensorflow as tf


def flatten_anchor_boxes(
        anchor_output,
        anchor_layer,
        include_conf=True,
        include_classes=True
):
    """
    Transforms a tensor as returned by ``keras_yolo.layers.AnchorLayer`` into a
    tensor of the format ``(batch_size, n_boxes, n_box_params)``.

    Args:
        anchor_output (tf.Tensor): Tensor as returned by
            ``keras_yolo.layer.AnchorLayer``
        anchor_layer (keras_yolo.layer.AnchorLayer): The layer that returned
            the tensor passed as first argument.
        include_conf (bool): Whether the confidence parameter is included in
            the ``anchor_output`` parameter.
        include_classes (bool): Whether the classes parameters are included in
            the ``anchor_output`` parameter.

    Returns:
        tf.Tensor: A tensor of the shape ``(batch_size, n_boxes, n_box_params)``
    """
    n_grid_cells = anchor_layer.grid_height * anchor_layer.grid_width
    n_anchors = anchor_layer.n_anchors
    n_classes = anchor_layer.n_classes * include_classes
    n_conf = 1 * include_conf
    desired_shape = (-1, n_grid_cells * n_anchors, 4 + n_conf + n_classes)

    return tf.reshape(
        anchor_output,
        shape=desired_shape
    )


def boxes_to_coords(boxes):
    """
    Converts a tensor of boxes encoded by centerpoint and width/height to a
    tensor of boxes encoded in coordinates.

    Args:
        boxes (tf.Tensor): Tensor with the first four entries of the last
            dimension representing the centerpoint and height/width of the box.

    Returns:
        tf.Tensor: Tensor of the same shape as the input, but with the entries
            of the last dimention representing box coordinates.
    """
    top_left_yx = boxes[..., 0:2] - 0.5 * boxes[..., 2:4]
    bottom_right_yx = boxes[..., 0:2] + 0.5 * boxes[..., 2:4]

    return tf.concat([top_left_yx, bottom_right_yx], axis=-1)


class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size:self.offset]

    def reset(self):
        self.offset = 4


def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == 'i':
        boxes = boxes.astype('float')

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        )

    # return only the bounding boxes that were picked
    return boxes[pick]


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
            flatten_anchor_boxes(
                anchor_output,
                anchor_head,
                include_conf=True,
                include_classes=True
            )
        )

        base_boxes.append(
            flatten_anchor_boxes(
                anchor_head.base_anchor_boxes,
                anchor_head,
                include_conf=False,
                include_classes=False
            )
        )

    predicted_boxes = tf.concat(predicted_boxes, axis=1)
    base_boxes = tf.concat(base_boxes, axis=1)

    return predicted_boxes, base_boxes
