from typing import Dict
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (Activation, Add, Concatenate, Conv1D, Conv2D, Dense,
                          GlobalAveragePooling2D, GlobalMaxPooling2D, Lambda, BatchNormalization,
                          Reshape, multiply,AveragePooling2D)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.enable_eager_execution(config=config)
layers = tf.keras.layers

"""
The architecture of generator is a modified U-Net.
There are skip connections between the encoder and decoder (as in U-Net).
"""

class Conv(layers.Layer):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()

        self.conv = tf.keras.Sequential([
            layers.Conv2D(out_channels, kernel_size=3, strides=4, padding='valid', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

    def call(self, inputs):
        return self.conv(inputs)

class DoubleConv(layers.Layer):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = tf.keras.Sequential([
            layers.Conv2D(mid_channels, kernel_size=3, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(out_channels, kernel_size=3, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

    def call(self, inputs):
        return self.double_conv(inputs)

class L2Pooling2D(tf.keras.layers.Layer):
    def __init__(self, pool_size, strides, padding='same', **kwargs):
        super( L2Pooling2D, self).__init__(**kwargs)
        self.pool = AveragePooling2D(pool_size=pool_size, padding='same', strides=strides)
        self.n = pool_size[0] * pool_size[1]

    def call(self, inputs):
        # return tf.sqrt(self.pool(tf.square(inputs)) * self.n)
        return tf.sqrt(self.pool(tf.square(self.pool(inputs))))
        # return tf.sqrt(self.pool(tf.square(inputs)))

def se_block(input_feature, ratio=16, name=""):
        channel = K.int_shape(input_feature)[-1]

        se_feature = GlobalAveragePooling2D()(input_feature)
        se_feature = Reshape((1, 1, channel))(se_feature)

        se_feature = Dense(channel // ratio,
                           activation='relu',
                           kernel_initializer='he_normal',
                           use_bias=False,
                           name="se_block_one_" + str(name))(se_feature)

        se_feature = Dense(channel,
                           kernel_initializer='he_normal',
                           use_bias=False,
                           name="se_block_two_" + str(name))(se_feature)
        se_feature = Activation('sigmoid')(se_feature)

        se_feature = multiply([input_feature, se_feature])

        dw_feature = GlobalMaxPooling2D()(input_feature)
        dw_feature = Reshape((1, 1, channel))(dw_feature)
        dw_feature = tf.keras.layers.DepthwiseConv2D(kernel_size=2,
                                                     strides=1,
                                                     padding='same',
                                                     depth_multiplier=1,
                                                     use_bias=False,
                                                     activation='relu')(dw_feature)
        dw_feature = layers.Conv2D(filters=channel, kernel_size=1, strides=1)(dw_feature)
        dw_feature = Activation('sigmoid')(dw_feature)
        dw_feature = multiply([input_feature, dw_feature])

        return se_feature + dw_feature

class Down(layers.Layer):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpooling = layers.MaxPooling2D(pool_size=2, strides=2)
        self.averagepooling = layers.AveragePooling2D(pool_size=2, strides=2)
        self.l2Pooling2D = L2Pooling2D(pool_size=(2, 2), strides=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def call(self, inputs):
        x1 = self.maxpooling(inputs)
        x2 = self.averagepooling(inputs)
        x3 = self.maxpooling(inputs)
        x5 = x1 + x2 + x3
        x6 = self.conv(x5)

        return x6


class Up(layers.Layer):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = layers.Conv2DTranspose(in_channels // 2, kernel_size=2, strides=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def call(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.shape[1] - x1.shape[1]
        diff_x = x2.shape[2] - x1.shape[2]
        x1 = tf.pad(x1, [[0, 0], [diff_y // 2, diff_y - diff_y // 2],
                         [diff_x // 2, diff_x - diff_x // 2], [0, 0]])
        x = tf.concat([x2, x1], axis=-1)
        x = se_block(x)
        return self.conv(x)


class OutConv(layers.Layer):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__()
        self.out_conv = layers.Conv2D(num_classes, kernel_size=1)

    def call(self, inputs):
        return self.out_conv(inputs)

class EncoderLayer(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides_s = 2, apply_batchnorm=True, add=False, padding_s='same'):
        super(EncoderLayer, self).__init__()
        initializer = tf.random_normal_initializer(mean=0., stddev=0.02)
        conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides_s,
                             padding=padding_s, kernel_initializer=initializer, use_bias=False)
        ac = layers.LeakyReLU()
        self.encoder_layer = None
        if add:
            self.encoder_layer = tf.keras.Sequential([conv])
        elif apply_batchnorm:
            bn = layers.BatchNormalization()
            self.encoder_layer = tf.keras.Sequential([conv, bn, ac])
        else:
            self.encoder_layer = tf.keras.Sequential([conv, ac])

    def call(self, x):
        return self.encoder_layer(x)

class DecoderLayer(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides_s=2, apply_dropout=False, add=False):
            super(DecoderLayer, self).__init__()
            initializer = tf.random_normal_initializer(mean=0., stddev=0.02)
            dconv = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides_s,
                                           padding='same', kernel_initializer=initializer, use_bias=False)
            bn = layers.BatchNormalization()
            ac = layers.ReLU()
            self.decoder_layer = None

            if add:
                self.decoder_layer = tf.keras.Sequential([dconv])
            elif apply_dropout:
                drop = layers.Dropout(rate=0.5)
                self.decoder_layer = tf.keras.Sequential([dconv, bn, drop, ac])
            else:
                self.decoder_layer = tf.keras.Sequential([dconv, bn, ac])

    def call(self, x):
        return self.decoder_layer(x)

class Resize(tf.keras.Model):
    def __init__(self):
        super(Resize, self).__init__()

        # Resize Input
        p_layer_1 = DecoderLayer(filters=2, kernel_size=4, strides_s=2, apply_dropout=False, add=True)
        p_layer_2 = DecoderLayer(filters=2, kernel_size=4, strides_s=2, apply_dropout=False, add=True)
        p_layer_3 = EncoderLayer(filters=2, kernel_size=(6, 1), strides_s=(4, 1), apply_batchnorm=False, add=True)  #(1, 64, 32, 2)

        self.p_layers = [p_layer_1, p_layer_2, p_layer_3]


    def call(self, x):
        # pass the encoder and record xs
        for p_layer in self.p_layers:
            x = p_layer(x)

        return x

class Generator(tf.keras.Model):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.resize = Resize()
        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def call(self, inputs):
        x0 = self.resize(inputs)
        x1 = self.in_conv(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return logits
