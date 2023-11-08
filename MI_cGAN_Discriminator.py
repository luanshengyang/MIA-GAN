import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from cGANGenerator import EncoderLayer
import os


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.enable_eager_execution(config=config)
layers = tf.keras.layers

"""
The Discriminator is a PatchGAN.
"""
class DilatedModule(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides_s, d, padding='same'):
        super(DilatedModule, self).__init__()
        self.dilated_layer = tf.keras.layers.Conv2D(padding=padding,
                                                    filters=filters,
                                                    kernel_size=kernel_size,
                                                    strides=1,
                                                    dilation_rate=d)
        #self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1)

    def call(self, x):
        x1 = self.dilated_layer(x)
        #x2 = self.conv1(x)
        x = x1 + x
        return x

class FeatureFusion(tf.keras.Model):
    #def __init__(self, filters, padding_s='same'):
    def __init__(self, filters, apply_batchnorm, padding='same'):
        super(FeatureFusion, self).__init__()
        self.FeatureFusion1 = DilatedModule(filters=filters, padding=padding, kernel_size=1, strides_s=1, d=1)
        self.FeatureFusion2 = DilatedModule(filters=filters, padding=padding, kernel_size=1, strides_s=1, d=2)
        self.FeatureFusion3 = DilatedModule(filters=filters, padding=padding, kernel_size=1, strides_s=1, d=3)
        self.FeatureFusion4 = DilatedModule(filters=filters, padding=padding, kernel_size=1, strides_s=1, d=4)
        self.FeatureFusion5 = DilatedModule(filters=filters, padding=padding, kernel_size=1, strides_s=1, d=5)
        #self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1)

    def call(self, x):
        x1 = self.FeatureFusion1(x)
        x2 = self.FeatureFusion2(x)
        x3 = self.FeatureFusion3(x)
        x4 = self.FeatureFusion4(x)
        x5 = self.FeatureFusion5(x)
        #x6 = self.conv1(x)
        x = x1 + x2 + x3 + x4 + x5
        return x


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        # downsample
        self.feature_fusion_1 = FeatureFusion(filters=2, apply_batchnorm=False)
        self.feature_fusion_2 = FeatureFusion(filters=2, apply_batchnorm=True)
        self.feature_fusion_3 = FeatureFusion(filters=2, apply_batchnorm=True)
        self.feature_fusion_4 = FeatureFusion(filters=2, apply_batchnorm=True)

        # conv block1
        self.zero_pad1 = layers.ZeroPadding2D()                                
        self.conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)
        self.bn1 = layers.BatchNormalization()                                 
        self.ac = layers.LeakyReLU()

        # block2
        self.zero_pad2 = tf.keras.layers.ZeroPadding2D()                       
        self.last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer) 

    def call(self, y):
        """inputs can be generated image. """
        target = y
        x = target     
        x = self.feature_fusion_1(x)
        x = self.feature_fusion_2(x)
        x = self.feature_fusion_3(x)
        x = self.feature_fusion_4(x)

        x = self.zero_pad1(x)
        x = self.conv(x)
        x = self.bn1(x)
        x = self.ac(x)

        x = self.zero_pad2(x)
        x = self.last(x)
        return x













