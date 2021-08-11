import numpy as np
import tensorflow as tf
import math


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]  # size = 8 X 8 for 64 batch size
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img


def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w


def layout_point(final_pred, output_height, output_width):
    bbox_pred = tf.reshape(final_pred, [64, 128, 2])

    x_r = tf.reshape(tf.range(output_width, dtype=tf.float32),
                     [1, output_width, 1, 1])
    x_r = tf.reshape(tf.tile(x_r, [1, 1, output_width, 1]), [
                     1, output_width*output_width, 1, 1])
    x_r = tf.tile(x_r, [64, 1, 128, 1])

    y_r = tf.reshape(tf.range(output_height, dtype=tf.float32), [
                     1, 1, output_height, 1])
    y_r = tf.reshape(tf.tile(y_r, [1, output_height, 1, 1]), [
                     1, output_height*output_height, 1, 1])
    y_r = tf.tile(y_r, [64, 1, 128, 1])

    x_pred = tf.reshape(
        tf.slice(bbox_pred, [0, 0, 0], [-1, -1, 1]), [64, 1, 128, 1])
    x_pred = tf.tile(x_pred, [1, output_width*output_width, 1, 1])
    x_pred = (output_width-1.0) * x_pred

    y_pred = tf.reshape(
        tf.slice(bbox_pred, [0, 0, 1], [-1, -1, 1]), [64, 1, 128, 1])
    y_pred = tf.tile(y_pred, [1, output_height*output_height, 1, 1])
    y_pred = (output_height-1.0) * y_pred

    x_diff = tf.maximum(0.0, 1.0-tf.abs(x_r - x_pred))
    y_diff = tf.maximum(0.0, 1.0-tf.abs(y_r - y_pred))
    xy_diff = x_diff * y_diff

    xy_max = tf.nn.max_pool(xy_diff, ksize=[1, 1, 128, 1], strides=[
                            1, 1, 1, 1], padding='VALID')
    xy_max = tf.reshape(xy_max, [64, output_height, output_width, 1])

    return xy_max
