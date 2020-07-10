import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization, concatenate, Conv2D, Input, Lambda,
    LeakyReLU, MaxPooling2D
)
from tensorflow.keras.models import Model

from keras_yolo import layers


class Yolo(object):
    INPUT_SIZE = (416, 416)
    GRID_SIZE = (13, 13)

    def __init__(self, n_labels, anchors):
        self.n_labels = n_labels
        self.anchors = anchors
        self.warmup_rounds = 10

        self.inputs, self.outputs, self.anchor_heads = self._get_graph()
        self.model = Model(inputs=self.inputs, outputs=self.outputs)

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
        skip_connection = Lambda(lambda x: tf.nn.space_to_depth(x, block_size=2))(
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

        # Initialize the definitive anchor head.
        anchor_head = layers.AnchorLayer(
            grid_height=grid_h,
            grid_width=grid_w,
            anchors=self.anchors,
            n_classes=self.n_labels
        )

        anchor_output = anchor_head(x)

        return input_image, [anchor_output], [anchor_head]
