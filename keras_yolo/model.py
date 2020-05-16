import numpy as np
import tensorflow as tf
from keras.layers import (
    BatchNormalization, Conv2D, Input, Lambda,
    MaxPooling2D, Reshape, Softmax
)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras.models import Model


def tile_outputs(output, max_boxes):
    expanded = tf.expand_dims(output, 4)
    return tf.tile(expanded, [1, 1, 1, 1, max_boxes, 1])


class Yolo(Model):
    INPUT_SIZE = (416, 416)
    GRID_SIZE = (13, 13)

    def __init__(self, n_labels, anchors=None, max_boxes=10):
        self.n_labels = n_labels
        self.anchors = anchors
        self.max_boxes = max_boxes
        self.warmup_rounds = 10

        input, output = self._get_graph()

        super().__init__(inputs=input, outputs=output)

    def base_anchor_boxes(self):
        grid_h, grid_w = Yolo.GRID_SIZE
        anchors_array = np.array(self.anchors).astype('float32')

        cell_x = tf.to_float(
            tf.reshape(
                tf.tile(tf.range(grid_w), [grid_h]), (grid_h, grid_w, 1, 1)
            )
        )
        cell_y = tf.transpose(cell_x, (1, 0, 2, 3))

        # The resulting grid + 0.5 are the center points of the actual cells
        xy_grid = (
            tf.tile(
                tf.concat([cell_x, cell_y], -1), [1, 1, len(self.anchors), 1]
            ) + 0.5
        )

        wh_grid = tf.tile(anchors_array[None, None, :], [grid_h, grid_w, 1, 1])

        return tf.concat([xy_grid, wh_grid], axis=-1)

    def _anchor_boxes(self, input_tensor):
        base_boxes = self.base_anchor_boxes()
        xy_base = base_boxes[..., 0:2]
        wh_base = base_boxes[..., 2:4]

        # Substract 1/2 from the result of applying sigmoid to get a
        # value in the range (-0.5, 0.5). Hence, they are fluctuating around
        # the grid centers
        xy = xy_base + (tf.sigmoid(input_tensor[..., 0:2]) - 0.5)

        # TODO: Solve this.... this returns inf...
        wh = tf.clip_by_value(wh_base * tf.exp(input_tensor[..., 2:4]), 0, 130)

        return tf.concat([xy, wh], axis=-1)

    def _tile_probas(self, x):
        return tf.tile(tf.expand_dims(x, 3), [1, 1, 1, len(self.anchors), 1])

    def _get_graph(self):
        image_h, image_w = Yolo.INPUT_SIZE
        grid_h, grid_w = Yolo.GRID_SIZE

        input_image = Input(shape=(image_h, image_w, 3))

        # Layer 1
        x = Conv2D(
            32, (3, 3), strides=(1, 1), padding='same',
            name='conv_1', use_bias=False
        )(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2
        x = Conv2D(
            64, (3, 3), strides=(1, 1), padding='same',
            name='conv_2', use_bias=False
        )(x)
        x = BatchNormalization(name='norm_2')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 3
        x = Conv2D(
            128, (3, 3), strides=(1, 1), padding='same',
            name='conv_3', use_bias=False
        )(x)
        x = BatchNormalization(name='norm_3')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 4
        x = Conv2D(
            64, (1, 1), strides=(1, 1), padding='same',
            name='conv_4', use_bias=False
        )(x)
        x = BatchNormalization(name='norm_4')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 5
        x = Conv2D(
            128, (3, 3), strides=(1, 1), padding='same',
            name='conv_5', use_bias=False
        )(x)
        x = BatchNormalization(name='norm_5')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(
            256, (3, 3), strides=(1, 1), padding='same',
            name='conv_6', use_bias=False
        )(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 7
        x = Conv2D(
            128, (1, 1), strides=(1, 1), padding='same',
            name='conv_7', use_bias=False
        )(x)
        x = BatchNormalization(name='norm_7')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 8
        x = Conv2D(
            256, (3, 3), strides=(1, 1), padding='same',
            name='conv_8', use_bias=False
        )(x)
        x = BatchNormalization(name='norm_8')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 9
        x = Conv2D(
            512, (3, 3), strides=(1, 1), padding='same',
            name='conv_9', use_bias=False
        )(x)
        x = BatchNormalization(name='norm_9')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 10
        x = Conv2D(
            256, (1, 1), strides=(1, 1), padding='same',
            name='conv_10', use_bias=False
        )(x)
        x = BatchNormalization(name='norm_10')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 11
        x = Conv2D(
            512, (3, 3), strides=(1, 1), padding='same',
            name='conv_11', use_bias=False
        )(x)
        x = BatchNormalization(name='norm_11')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 12
        x = Conv2D(
            256, (1, 1), strides=(1, 1), padding='same',
            name='conv_12', use_bias=False
        )(x)
        x = BatchNormalization(name='norm_12')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 13
        x = Conv2D(
            512, (3, 3), strides=(1, 1), padding='same',
            name='conv_13', use_bias=False
        )(x)
        x = BatchNormalization(name='norm_13')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Store x to use as a skip connection later in the graph
        skip_connection = x

        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 14
        x = Conv2D(
            1024, (3, 3), strides=(1, 1), padding='same',
            name='conv_14', use_bias=False
        )(x)
        x = BatchNormalization(name='norm_14')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 15
        x = Conv2D(
            512, (1, 1), strides=(1, 1), padding='same',
            name='conv_15', use_bias=False
        )(x)
        x = BatchNormalization(name='norm_15')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 16
        x = Conv2D(
            1024, (3, 3), strides=(1, 1), padding='same',
            name='conv_16', use_bias=False
        )(x)
        x = BatchNormalization(name='norm_16')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 17
        x = Conv2D(
            512, (1, 1), strides=(1, 1), padding='same',
            name='conv_17', use_bias=False
        )(x)
        x = BatchNormalization(name='norm_17')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 18
        x = Conv2D(
            1024, (3, 3), strides=(1, 1), padding='same',
            name='conv_18', use_bias=False
        )(x)
        x = BatchNormalization(name='norm_18')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 19
        x = Conv2D(
            1024, (3, 3), strides=(1, 1), padding='same',
            name='conv_19', use_bias=False
        )(x)
        x = BatchNormalization(name='norm_19')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 20
        x = Conv2D(
            1024, (3, 3), strides=(1, 1), padding='same',
            name='conv_20', use_bias=False
        )(x)
        x = BatchNormalization(name='norm_20')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 21
        skip_connection = Conv2D(
            64, (1, 1), strides=(1, 1), padding='same',
            name='conv_21', use_bias=False
        )(skip_connection)
        skip_connection = BatchNormalization(name='norm_21')(skip_connection)
        skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
        skip_connection = Lambda(lambda x: tf.space_to_depth(x, block_size=2))(
            skip_connection
        )

        x = concatenate([skip_connection, x])

        # Layer 22
        x = Conv2D(
            1024, (3, 3), strides=(1, 1), padding='same',
            name='conv_22', use_bias=False
        )(x)
        x = BatchNormalization(name='norm_22')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # First we predict 4 coordinates per box to which we will apply the
        # right transformations in the Yolo.anchor_boxes method
        boxes = Conv2D(
            len(self.anchors) * 4,
            (1, 1),
            strides=(1, 1),
            padding='same',
            name='boxes',
            activation='linear',
        )(x)
        boxes = Reshape((grid_h, grid_w, len(self.anchors), 4))(boxes)
        boxes = Lambda(self._anchor_boxes)(boxes)

        confidence = Conv2D(
            len(self.anchors) * 1,
            (1, 1),
            strides=(1, 1),
            padding='same',
            name='conf',
            activation='sigmoid',
        )(x)
        confidence = Reshape((grid_h, grid_w, len(self.anchors), 1))(confidence)

        classes = Conv2D(
            self.n_labels, (1, 1), strides=(1, 1), padding='same', name='probas'
        )(x)
        classes = Softmax()(classes)
        tiled_classes = Lambda(self._tile_probas)(classes)

        output = concatenate([boxes, confidence, tiled_classes])
        output = Lambda(lambda x: tile_outputs(x, self.max_boxes))(output)

        return input_image, output

    def _warmup_loss(self, y_true, y_pred):
        base_anchor_boxes = self.base_anchor_boxes()

        xy_true = base_anchor_boxes[..., 0:2]
        wh_true = base_anchor_boxes[..., 2:4]

        xy_pred = y_pred[..., 0, 0:2]
        wh_pred = y_pred[..., 0, 2:4] + 0.0000001

        wh_loss = tf.reduce_sum(
            tf.squared_difference(tf.sqrt(wh_pred), tf.sqrt(wh_true))
        )
        xy_loss = tf.reduce_sum(tf.squared_difference(xy_true, xy_pred))

        return wh_loss + xy_loss

    @staticmethod
    def compute_iou(y_true, y_pred):
        xy_true = y_true[..., 0:2]
        wh_true = y_true[..., 2:4]

        xy_pred = y_pred[..., 0:2]
        wh_pred = y_pred[..., 2:4]

        xy_tl_true = xy_true - 0.5 * wh_true
        xy_br_true = xy_true + 0.5 * wh_true

        xy_tl_pred = xy_pred - 0.5 * wh_pred
        xy_br_pred = xy_pred + 0.5 * wh_pred

        # IOU computation
        max_xy_tl = tf.maximum(xy_tl_true, xy_tl_pred)
        min_xy_br = tf.minimum(xy_br_true, xy_br_pred)

        intersect_wh = tf.maximum(min_xy_br - max_xy_tl, 0)
        intersection_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        union_area_pred = wh_pred[..., 0] * wh_pred[..., 1]
        union_area_truth = wh_true[..., 0] * wh_true[..., 1]
        union_area = union_area_pred + union_area_truth - intersection_area

        return tf.truediv(intersection_area, union_area)

    def pad_base_boxes_for_loss(self):
        base_boxes = tf.tile(
            tf.expand_dims(self.base_anchor_boxes(), axis=-2),
            [1, 1, 1, self.max_boxes, 1],
        )
        return base_boxes

    def build_iou_mask(self, iou, threshold=None):
        """
        Computes the IOU mask to use for training. If no threshold is provided
        the best matching anchor of all anchors is returned. If a threshold is
        given all values with an iou above that threshold are returned

        Args:
            iou: Tensor of shape (None, GRID_H, GRID_W,
                N_ANCHORS, MAX_TRAIN_BOXES)
            threshold: (None or Float) If given, the IOU threshold to use for
                the mask
        Returns:
            Tensor: A boolean tensor of shape (None, GRID_H, GRID_W, N_ANCHORS,
                MAX_TRAIN_BOXES)
        """
        grid_h, grid_w = Yolo.GRID_SIZE
        if threshold is None:
            max_iou = tf.tile(
                tf.reduce_max(iou, axis=[1, 2, 3], keepdims=True),
                [1, grid_h, grid_w, len(self.anchors), 1],
            )

            return (iou >= max_iou) & (max_iou > 0)
        else:
            return (iou > threshold) & (iou > 0)

    def _loss(self, y_true, y_pred):
        xy_true = y_true[..., 0:2]
        wh_true = y_true[..., 2:4]

        xy_pred = y_pred[..., 0:2]
        wh_pred = y_pred[..., 2:4]

        conf_pred = y_pred[..., 4]

        # Compute IOUs to match predicted boxes with actual boxes
        base_boxes = self.pad_base_boxes_for_loss()
        anchors_iou = Yolo.compute_iou(y_true, base_boxes)
        predictions_iou = Yolo.compute_iou(y_true, y_pred)
        # We compute a tie breaker that assigns the best box in case of ties
        tie_breaker = -0.0001 * tf.norm(xy_true - base_boxes[..., 0:2], axis=-1)

        # Compute masks we will use to determine which coordinates
        # to adapt and apply these masks to get matching records
        anchors_mask = self.build_iou_mask(anchors_iou + tie_breaker)
        predictions_mask = self.build_iou_mask(predictions_iou, threshold=0.6)

        xy_true_masked = tf.boolean_mask(xy_true, anchors_mask)
        wh_true_masked = tf.boolean_mask(wh_true, anchors_mask)

        xy_pred_masked = tf.boolean_mask(xy_pred, anchors_mask)
        wh_pred_masked = tf.boolean_mask(wh_pred, anchors_mask)

        # Compute the XY and WH loss
        xy_loss = tf.reduce_mean(
            tf.squared_difference(xy_true_masked, xy_pred_masked)
        )
        wh_loss = tf.reduce_mean(
            tf.squared_difference(
                tf.sqrt(wh_true_masked), tf.sqrt(wh_pred_masked)
            )
        )
        coord_loss = xy_loss + wh_loss
        coord_loss = 5 * coord_loss

        # Compute the object confidence loss
        conf_pred_masked = tf.boolean_mask(
            conf_pred, anchors_mask | predictions_mask
        )
        true_confs_masked = tf.boolean_mask(
            predictions_iou, anchors_mask | predictions_mask
        )
        obj_conf_loss = tf.reduce_mean(
            tf.squared_difference(true_confs_masked, conf_pred_masked)
        )

        # compute the no object confidence loss
        no_conf_pred_masked = tf.boolean_mask(
            conf_pred, ~(anchors_mask | predictions_mask)
        )
        no_obj_conf_loss = tf.reduce_mean(
            tf.squared_difference(0.0, no_conf_pred_masked)
        )

        return coord_loss + obj_conf_loss + 0.5 * no_obj_conf_loss

    def loss(self, y_true, y_pred):
        seen = tf.Variable(0.0)
        seen = tf.assign_add(seen, 1.0)

        loss = tf.cond(
            tf.less(seen, self.warmup_rounds),
            lambda: self._warmup_loss(y_true, y_pred),
            lambda: self._loss(y_true, y_pred),
        )

        return loss
