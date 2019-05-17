from keras.models import Model
from keras.layers import (
    Reshape, Activation, Conv2D, Input, MaxPooling2D,
    BatchNormalization, Flatten, Dense, Lambda,
    Softmax, LeakyReLU)


class YoloConvLayer(Model):
    def __init__(self, name, n_filters, filter_size=(3,3), strides=(1,1)):
        super().__init__(name=name)

        # Initialize the layers for this sublayer
        self.conv = Conv2D(n_filters, filter_size, strides=strides,
                           padding='same', use_bias=False)
        self.norm = BatchNormalization()
        self.activation = LeakyReLU(alpha=0.1)

    def call(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)

        return x



