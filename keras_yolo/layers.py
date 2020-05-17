import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models


class YoloConvLayer(models.Model):
    def __init__(self, name, n_filters, filter_size=(3, 3), strides=(1, 1)):
        super().__init__(name=name)

        # Initialize the layers for this sublayer
        self.conv = layers.Conv2D(
            n_filters, filter_size, strides=strides,
            padding='same', use_bias=False
        )
        self.norm = layers.BatchNormalization()
        self.activation = layers.LeakyReLU(alpha=0.1)

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.norm(x, training=training)
        x = self.activation(x)

        return x


class AnchorLayer(models.Model):
    def __init__(self,
                 grid_height,
                 grid_width,
                 anchors,
                 n_classes,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.anchors = anchors
        self.n_classes = n_classes

        self.boxes_conv = layers.Conv2D(
            len(self.anchors) * 4,
            (1, 1),
            strides=(1, 1),
            padding='same',
            name='boxes',
            activation='linear',
        )

        self.confidence_conv = layers.Conv2D(
            len(self.anchors) * 1,
            (1, 1),
            strides=(1, 1),
            padding='same',
            name='conf',
            activation='sigmoid',
        )

        self.classes_conv = layers.Conv2D(
            self.n_classes,
            (1, 1),
            strides=(1, 1),
            padding='same',
            name='probas',
            activation='softmax'
        )

    def compute_predicted_boxes(self, input):
        unaligned_preds = self.boxes_conv(input)
        preds = tf.reshape(
            unaligned_preds,
            (-1, self.grid_height, self.grid_width, len(self.anchors), 4)
        )

        yx_base = self.base_anchor_boxes[..., 0:2]
        hw_base = self.base_anchor_boxes[..., 2:4]

        # Substract 1/2 from the result of applying sigmoid to get a
        # value in the range (-0.5, 0.5). Hence, they are fluctuating around
        # the grid centers
        yx = yx_base + (tf.sigmoid(preds[..., 0:2]) - 0.5)

        # Clip the WH values at an arbitrary boundary such that we prevent
        # them from exploding to infinity during training
        # TODO: Fix the arbitrary clipping bound
        hw = tf.clip_by_value(hw_base * tf.exp(preds[..., 2:4]), 0, 130)

        return tf.concat([yx, hw], axis=-1)

    def compute_confidences(self, input):
        unaligned_confs = self.confidence_conv(input)
        confs = tf.reshape(
            unaligned_confs,
            (-1, self.grid_height, self.grid_width, len(self.anchors), 1)
        )

        return confs

    def compute_classes(self, input):
        unaligned_classes = self.classes_conv(input)
        classes = tf.tile(
            unaligned_classes[:, :, :, None, :],
            [1, 1, 1, len(self.anchors), 1]
        )
        return classes

    @property
    def base_anchor_boxes(self):
        h, w = self.grid_height, self.grid_width

        y_centroids = tf.tile(
            tf.range(h, dtype='float')[:, None],
            [1, w]
        ) + 0.5
        x_centroids = tf.tile(
            tf.range(w, dtype='float')[None, :],
            [h, 1]
        ) + 0.5

        yx_centroids = tf.concat(
            [y_centroids[..., None, None], x_centroids[..., None, None]],
            axis=-1
        )
        yx_centroids = tf.tile(yx_centroids, [1, 1, len(self.anchors), 1])

        hw = tf.tile(
            tf.constant(self.anchors)[None, None, ...],
            [h, w, 1, 1]
        )

        return tf.concat([yx_centroids, hw], axis=-1)
