import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
plt.style.use('dark_background')


def convert_to_cxywh(data):
    bboxes = data[..., 0:4]
    labels = data[..., 4:]
    mask = np.zeros_like(data[..., 3:4])
    labels = np.concatenate((mask, labels), axis=2)
    labels = np.argmax(labels, axis=2)
    class_info = np.expand_dims(labels, axis=2)
    cxywh = np.concatenate((class_info, bboxes), axis=2)
    cxywh[..., 1] = cxywh[..., 1] - cxywh[..., 3]/2
    cxywh[..., 2] = cxywh[..., 2] - cxywh[..., 4]/2
    return cxywh


def generate_colors(class_names=None, n_class=50):
    cmap = ["", "#dc143c", "#ffff00", "#00ff00", "#ff00ff", "#1e90ff", "#fff5ee",
            "#00ffff", "#8b008b", "#ff4500", "#8b4513", "#808000", "#483d8b",
            "#008000", "#000080", "#9acd32", "#ffa500", "#ba55d3", "#00fa9a",
            "#dc143c", "#0000ff", "#f08080", "#f0e68c", "#dda0dd", "#ff1493"]
    colors = dict()
    if class_names == None:
        class_names = []
        for i in range(n_class):
            class_names.append('class'+str(i+1))
    for i in range(n_class):
        colors[class_names[i]] = cmap[i]
    return colors


class_names = ['None', 'Text', 'Title', 'List', 'Table', 'Figure']
colors = generate_colors(n_class=6, class_names=class_names)


def plot_layouts(pred, colors, class_names, path=""):
    height = 15
    width = 9
    fig = plt.figure(figsize=(width, height), dpi=50, facecolor=(0, 0, 0))
    height_ratio = [0.25, 1, 1, 1, 1, 1]
    grid = plt.GridSpec(6, 4,
                        hspace=0.05, wspace=0.05,
                        height_ratios=height_ratio,
                        left=0.02, right=0.98, top=0.98, bottom=0.02)
    index = 0
    legend = []
    ax = fig.add_subplot(grid[index: index+4])
    index += 4
    for i in range(1, 6):
        legend.append(Patch(facecolor=colors[class_names[i]]+"40",
                            edgecolor=colors[class_names[i]],
                            label=class_names[i]))

    ax.legend(handles=legend, ncol=3, loc=8, fontsize=25, facecolor=(0, 0, 0))
    ax.axis('off')

    for i in range(16):
        ax = fig.add_subplot(grid[index])
        index += 1

        data = pred[i]
        rect1 = patches.Rectangle((0, 0), 180, 240)
        rect1.set_color((0, 0, 0, 1))
        ax.add_patch(rect1)
        for box in data:

            c, x, y, w, h = box
            if c == 0:
                continue
            x = x*180
            y = y*240
            w = w*180
            h = h*240
            rect = patches.Rectangle((x, y), w, h, linewidth=2)
            rect.set_color(colors[class_names[int(c)]]+"00")
            rect.set_linestyle('-')
            rect.set_edgecolor(colors[class_names[int(c)]])
            ax.add_patch(rect)
        ax.plot()
        ax.set_facecolor((0, 0, 0))
        for spine in ax.spines.values():
            spine.set_edgecolor('green')
            spine.set_linewidth(2)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(path, facecolor=(0, 0, 0))


def layout_bbox(final_pred, output_height, output_width):
    final_pred = tf.reshape(final_pred, [64, 9, 9])
    bbox_reg = tf.slice(final_pred, [0, 0, 0], [-1, -1, 4])
    cls_prob = tf.slice(final_pred, [0, 0, 4], [-1, -1, 5])

    bbox_reg = tf.reshape(bbox_reg, [64, 9, 4])

    x_c = tf.slice(bbox_reg, [0, 0, 0], [-1, -1, 1]) * output_width
    y_c = tf.slice(bbox_reg, [0, 0, 1], [-1, -1, 1]) * output_height
    w = tf.slice(bbox_reg, [0, 0, 2], [-1, -1, 1]) * output_width
    h = tf.slice(bbox_reg, [0, 0, 3], [-1, -1, 1]) * output_height

    x1 = x_c - 0.5*w
    x2 = x_c + 0.5*w
    y1 = y_c - 0.5*h
    y2 = y_c + 0.5*h

    xt = tf.reshape(tf.range(output_width, dtype=tf.float32), [1, 1, 1, -1])
    xt = tf.reshape(tf.tile(xt, [64, 9, output_height, 1]), [64, 9, -1])

    yt = tf.reshape(tf.range(output_height, dtype=tf.float32), [1, 1, -1, 1])
    yt = tf.reshape(tf.tile(yt, [64, 9, 1, output_width]), [64, 9, -1])

    x1_diff = tf.reshape(xt-x1, [64, 9, output_height, output_width, 1])
    y1_diff = tf.reshape(yt-y1, [64, 9, output_height, output_width, 1])
    x2_diff = tf.reshape(x2-xt, [64, 9, output_height, output_width, 1])
    y2_diff = tf.reshape(y2-yt, [64, 9, output_height, output_width, 1])

    x1_line = tf.nn.relu(1.0 - tf.abs(x1_diff)) * tf.minimum(
        tf.nn.relu(y1_diff), 1.0) * tf.minimum(tf.nn.relu(y2_diff), 1.0)
    x2_line = tf.nn.relu(1.0 - tf.abs(x2_diff)) * tf.minimum(
        tf.nn.relu(y1_diff), 1.0) * tf.minimum(tf.nn.relu(y2_diff), 1.0)
    y1_line = tf.nn.relu(1.0 - tf.abs(y1_diff)) * tf.minimum(
        tf.nn.relu(x1_diff), 1.0) * tf.minimum(tf.nn.relu(x2_diff), 1.0)
    y2_line = tf.nn.relu(1.0 - tf.abs(y2_diff)) * tf.minimum(
        tf.nn.relu(x1_diff), 1.0) * tf.minimum(tf.nn.relu(x2_diff), 1.0)

    xy_max = tf.reduce_max(tf.concat(
        [x1_line, x2_line, y1_line, y2_line], axis=-1), axis=-1, keepdims=True)

    spatial_prob = tf.multiply(
        tf.tile(xy_max, [1, 1, 1, 1, 5]), tf.reshape(cls_prob, [64, 9, 1, 1, 5]))
    spatial_prob_max = tf.reduce_max(spatial_prob, axis=1, keepdims=False)

    return spatial_prob_max
