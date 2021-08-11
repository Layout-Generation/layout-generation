import math
from glob import glob
import tensorflow as tf
from tensorflow.keras import initializers
import numpy as np
import random
from utils import *


class RelationModule(tf.keras.Model):
    def __init__(self, channels=128, output_dim=128, key_dim=128, **kwargs):
        super(RelationModule, self).__init__(**kwargs)
        self.key_dim = channels
        self.output_dim = channels
        self.channels = channels
        self.key = tf.keras.layers.Conv2D(output_dim, (1, 1), strides=(1, 1), padding='same', kernel_initializer=initializers.TruncatedNormal(
            stddev=0.02, mean=0.0), bias_initializer=initializers.constant(0.0))
        self.query = tf.keras.layers.Conv2D(key_dim, (1, 1), strides=(1, 1), padding='same', kernel_initializer=initializers.TruncatedNormal(
            stddev=0.02, mean=0.0), bias_initializer=initializers.constant(0.0))
        self.value = tf.keras.layers.Conv2D(key_dim, (1, 1), strides=(1, 1), padding='same', kernel_initializer=initializers.TruncatedNormal(
            stddev=0.02, mean=0.0), bias_initializer=initializers.constant(0.0))
        self.projection = tf.keras.layers.Conv2D(channels, (1, 1), strides=(1, 1), padding='same', kernel_initializer=initializers.TruncatedNormal(
            stddev=0.02, mean=0.0), bias_initializer=initializers.constant(0.0))

    def call(self, inputs):
        f_k = tf.reshape(self.key(inputs), [
                         inputs.shape[0], inputs.shape[1]*inputs.shape[2], self.key_dim])
        f_q = tf.reshape(self.query(inputs), [
                         inputs.shape[0], inputs.shape[1]*inputs.shape[2], self.key_dim])
        f_q = tf.transpose(f_q, perm=[0, 2, 1])
        f_v = tf.reshape(self.value(inputs), [
                         inputs.shape[0], inputs.shape[1]*inputs.shape[2], self.output_dim])

        attention_weight = tf.matmul(
            f_k, f_q)/(inputs.shape[1]*inputs.shape[2])
        out = tf.matmul(tf.transpose(attention_weight, perm=[0, 2, 1]), f_v)
        out = tf.reshape(
            out, [inputs.shape[0], inputs.shape[1], inputs.shape[2], self.output_dim])
        out = self.projection(out)
        return out


class Discriminator(tf.keras.Model):
    def __init__(self, n_filters=32, n_hidden=128, layout_dim=(28, 28), render=layout_bbox, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.layout_dim = layout_dim
        self.render = render
        self.act = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.conv1 = tf.keras.layers.Conv2D(32, (5, 5), input_shape=layout_dim, strides=(
            2, 2), padding='valid', kernel_initializer=initializers.TruncatedNormal(stddev=0.02, mean=0.0), bias_initializer=initializers.constant(0.0))
        self.bn1 = tf.keras.layers.BatchNormalization(
            epsilon=1e-5, momentum=0.9)

        self.conv2 = tf.keras.layers.Conv2D(32*2, (5, 5), strides=(2, 2), padding='valid', kernel_initializer=initializers.TruncatedNormal(
            stddev=0.02, mean=0.0), bias_initializer=initializers.constant(0.0))
        self.bn2 = tf.keras.layers.BatchNormalization(
            epsilon=1e-5, momentum=0.9)

        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(512, kernel_initializer=initializers.RandomNormal(
            stddev=0.02, mean=0.0), bias_initializer=initializers.constant(0.0))
        self.bn3 = tf.keras.layers.BatchNormalization(
            epsilon=1e-5, momentum=0.9)
        self.fc2 = tf.keras.layers.Dense(1, kernel_initializer=initializers.RandomNormal(
            stddev=0.02, mean=0.0), bias_initializer=initializers.constant(0.0))

    def call(self, inputs):
        x = self.render(inputs, self.layout_dim[0], self.layout_dim[1])
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = self.act(self.bn3(self.fc1(x)))
        out = self.fc2(x)
        return out


class Generator(tf.keras.Model):
    def __init__(self, n_filters=128, output_dim=2, n_component=128, n_class=1, include_probability=False, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.n_filters = n_filters
        self.output_dim = output_dim
        self.n_component = n_component
        self.n_class = n_class
        self.include_probability = include_probability

        self.act = tf.keras.layers.ReLU()
        self.conv1_1 = tf.keras.layers.Conv2D(n_filters//4, (1, 1), input_shape=(self.n_component, 1, self.n_class+self.output_dim), strides=(
            1, 1), padding='same', kernel_initializer=initializers.TruncatedNormal(stddev=0.02, mean=0.0), bias_initializer=initializers.constant(0.0))
        self.bn1_1 = tf.keras.layers.BatchNormalization(
            epsilon=1e-5, momentum=0.9)
        self.conv1_2 = tf.keras.layers.Conv2D(n_filters//16, (1, 1), strides=(1, 1), padding='same',
                                              kernel_initializer=initializers.TruncatedNormal(stddev=0.02, mean=0.0), bias_initializer=initializers.constant(0.0))
        self.bn1_2 = tf.keras.layers.BatchNormalization(
            epsilon=1e-5, momentum=0.9)
        self.conv1_3 = tf.keras.layers.Conv2D(n_filters//16, (1, 1), strides=(1, 1), padding='same',
                                              kernel_initializer=initializers.TruncatedNormal(stddev=0.02, mean=0.0), bias_initializer=initializers.constant(0.0))
        self.bn1_3 = tf.keras.layers.BatchNormalization(
            epsilon=1e-5, momentum=0.9)
        self.conv1_4 = tf.keras.layers.Conv2D(n_filters//4, (1, 1), strides=(1, 1), padding='same',
                                              kernel_initializer=initializers.TruncatedNormal(stddev=0.02, mean=0.0), bias_initializer=initializers.constant(0.0))
        self.bn1_4 = tf.keras.layers.BatchNormalization(
            epsilon=1e-5, momentum=0.9)

        self.relation1 = RelationModule(
            channels=n_filters//4, output_dim=n_filters//4, key_dim=n_filters//4)
        self.relation2 = RelationModule(
            channels=n_filters//4, output_dim=n_filters//4, key_dim=n_filters//4)
        self.relation3 = RelationModule(
            channels=n_filters, output_dim=n_filters, key_dim=n_filters)
        self.relation4 = RelationModule(
            channels=n_filters, output_dim=n_filters, key_dim=n_filters)

        self.bn_x1 = tf.keras.layers.BatchNormalization(
            epsilon=1e-5, momentum=0.9)
        self.bn_x2 = tf.keras.layers.BatchNormalization(
            epsilon=1e-5, momentum=0.9)
        self.bn_x3 = tf.keras.layers.BatchNormalization(
            epsilon=1e-5, momentum=0.9)
        self.bn_x4 = tf.keras.layers.BatchNormalization(
            epsilon=1e-5, momentum=0.9)

        self.bn_x5 = tf.keras.layers.BatchNormalization(
            epsilon=1e-5, momentum=0.9)
        self.bn_x6 = tf.keras.layers.BatchNormalization(
            epsilon=1e-5, momentum=0.9)
        self.bn_x7 = tf.keras.layers.BatchNormalization(
            epsilon=1e-5, momentum=0.9)
        self.bn_x8 = tf.keras.layers.BatchNormalization(
            epsilon=1e-5, momentum=0.9)

        self.conv2_1 = tf.keras.layers.Conv2D(n_filters, (1, 1), strides=(1, 1), padding='same', kernel_initializer=initializers.TruncatedNormal(
            stddev=0.02, mean=0.0), bias_initializer=initializers.constant(0.0))
        self.bn2_1 = tf.keras.layers.BatchNormalization(
            epsilon=1e-5, momentum=0.9)
        self.conv2_2 = tf.keras.layers.Conv2D(n_filters//4, (1, 1), strides=(1, 1), padding='same',
                                              kernel_initializer=initializers.TruncatedNormal(stddev=0.02, mean=0.0), bias_initializer=initializers.constant(0.0))
        self.bn2_2 = tf.keras.layers.BatchNormalization(
            epsilon=1e-5, momentum=0.9)
        self.conv2_3 = tf.keras.layers.Conv2D(n_filters//4, (1, 1), strides=(1, 1), padding='same',
                                              kernel_initializer=initializers.TruncatedNormal(stddev=0.02, mean=0.0), bias_initializer=initializers.constant(0.0))
        self.bn2_3 = tf.keras.layers.BatchNormalization(
            epsilon=1e-5, momentum=0.9)
        self.conv2_4 = tf.keras.layers.Conv2D(n_filters, (1, 1), strides=(1, 1), padding='same', kernel_initializer=initializers.TruncatedNormal(
            stddev=0.02, mean=0.0), bias_initializer=initializers.constant(0.0))
        self.bn2_4 = tf.keras.layers.BatchNormalization(
            epsilon=1e-5, momentum=0.9)
        self.geometric_param = tf.keras.layers.Conv2D(output_dim, (1, 1), strides=(
            1, 1), padding='same', kernel_initializer=initializers.TruncatedNormal(stddev=0.001, mean=0.0), bias_initializer=initializers.constant(0.0))
        self.class_score = tf.keras.layers.Conv2D(n_class, (1, 1), strides=(1, 1), padding='same', kernel_initializer=initializers.TruncatedNormal(
            stddev=0.02, mean=0.0), bias_initializer=initializers.constant(0.0))

    def call(self, x):
        x = tf.reshape(x, [x.shape[0], self.n_component,
                       1, self.n_class+self.output_dim])
        h1_0 = self.bn1_1(self.conv1_1(x))
        h1_1 = self.act(self.bn1_2(self.conv1_2(x)))
        h1_2 = self.act(self.bn1_3(self.conv1_3(h1_1)))
        h1_3 = self.bn1_4(self.conv1_4(h1_2))
        embedding = self.act(tf.add(h1_0, h1_3))
        embedding = tf.reshape(
            embedding, [x.shape[0], self.n_component, 1, 256])

        context = self.act(self.bn_x2(
            tf.add(embedding, self.bn_x1(self.relation1(embedding)))))
        context = self.act(self.bn_x4(
            tf.add(context, self.bn_x3(self.relation2(context)))))

        h2_0 = self.bn2_1(self.conv2_1(context))
        h2_1 = self.act(self.bn2_2(self.conv2_2(h2_0)))
        h2_2 = self.act(self.bn2_3(self.conv2_3(h2_1)))
        h2_3 = self.bn2_4(self.conv2_4(h2_2))
        decoded = self.act(tf.add(h2_0, h2_3))

        decoded = self.act(self.bn_x6(
            tf.add(decoded, self.bn_x5(self.relation3(decoded)))))
        decoded = self.act(self.bn_x8(
            tf.add(decoded, self.bn_x7(self.relation4(decoded)))))

        out = self.geometric_param(decoded)
        out = tf.sigmoid(tf.reshape(
            out, [-1, self.n_component, self.output_dim]))

        cls_score = self.class_score(decoded)
        cls_prob = tf.sigmoid(tf.reshape(
            cls_score, [-1, self.n_component, self.n_class]))
        final_pred = tf.concat([out, cls_prob], axis=-1)
        return final_pred
