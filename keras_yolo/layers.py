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
    pass
