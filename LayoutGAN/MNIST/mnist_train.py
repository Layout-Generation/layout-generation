import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
import random
from mnist_utils import *
from mnist_modules import *
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from PIL import ImageFont, ImageDraw
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class LayoutGAN(object):
    def __init__(self, geometric_dim=2, n_class=1, batch_size=64, n_component=128, layout_dim=(28, 28), d_lr=1e-5, g_lr=1.01e-5, update_ratio=2, clip_value=0.08568, dataset_name='default', dataset_path='./data/pre_data_cls.npy', checkpoint_dir=None, sample_dir=None):
        self.batch_size = batch_size
        self.n_component = n_component
        self.n_class = n_class
        self.geometric_dim = geometric_dim
        self.layout_dim = layout_dim
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.data = np.load(dataset_path)
        self.build_model(d_lr, g_lr)
        self.sample_dir = sample_dir
        self.update_ratio = update_ratio
        self.clip_value = clip_value
        epoch_step = len(self.data) // self.batch_size
        dlr = tf.keras.optimizers.schedules.ExponentialDecay(
            1e-5, epoch_step*20, 0.1, staircase=True, name=None)

    def build_model(self, dlr, g_lr):
        self.G = self.build_generator()
        self.D = self.build_discriminator()
        self.d_opt = tf.keras.optimizers.Adam(dlr)
        self.g_opt = tf.keras.optimizers.Adam(g_lr)

    def step(self, real_data, noise, training=True):
        with tf.GradientTape() as disc_tape:
            disc_loss = self.discriminator_loss(real_data, noise)

        if(training):
            gradients_of_discriminator = disc_tape.gradient(
                disc_loss, self.D.trainable_variables)
            self.d_opt.apply_gradients(
                zip(gradients_of_discriminator, self.D.trainable_variables))

        for i in range(2):
            with tf.GradientTape() as gen_tape:
                gen_loss = self.generator_loss(noise)

            if(training):
                gradients_of_generator = gen_tape.gradient(
                    gen_loss, self.G.trainable_variables)
                self.g_opt.apply_gradients(
                    zip(gradients_of_generator, self.G.trainable_variables))

        return gen_loss, disc_loss

    def train(self):
        epoch_step = len(self.data) // self.batch_size
        sample = self.data[0:self.batch_size]
        sample_inputs = np.array(sample).astype(np.float32)
        sample_inputs = sample_inputs * 28.0 / 27.0
        sample_z = np.random.normal(
            0.5, 0.13, (self.batch_size, self.n_component, self.n_class, self.geometric_dim))
        counter = 1
        start_time = time.time()

        for epoch in range(150):
            np.random.shuffle(self.data)
            batch_idxs = len(self.data) // self.batch_size

            for idx in range(0, batch_idxs):
                batch = self.data[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_images = np.array(batch).astype(np.float32)

                batch_images = batch_images * 28.0 / 27.0
                batch_z = np.random.normal(
                    0.5, 0.13, (self.batch_size, self.n_component, self.n_class, self.geometric_dim))
                g_loss, d_loss = self.step(batch_images, batch_z)
                counter += 1
                if np.mod(counter, 10) == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.4f, g_loss: %.4f"
                          % (epoch, idx, batch_idxs, time.time()-start_time, d_loss, g_loss))

                if np.mod(counter, 1) == 0:
                    samples = self.G(sample_z)
                    g_loss, d_loss = self.step(
                        sample_inputs, sample_z, training=False)
                    samples = np.reshape(samples, (64, 128, 2))
                    samples = 27.0 * samples

                    img_all = np.zeros(
                        (64, self.layout_dim[0], self.layout_dim[1], 3), dtype=np.uint8)
                    rendered_layout = self.D.render(
                        samples, self.layout_dim[0], self.layout_dim[1])
                    img_list = []
                    for img_ind in range(64):
                        pointset = np.rint(
                            samples[img_ind, :, :]).astype(np.int)
                        pointset = pointset[~(pointset == 0).all(1)]

                        img = np.zeros((28, 28), dtype=np.float32)
                        img[pointset[:, 0], pointset[:, 1]] = 255
                        img_list.append(img/255)
                        img = Image.fromarray(img.astype('uint8'), 'L')
                        img_all[img_ind, :, :, :] = np.array(
                            img.convert('RGB'))
                    img_all = np.squeeze(
                        merge(img_all, image_manifold_size(samples.shape[0])))
                    plt.imsave('{}/train_{:02d}_{:04d}.jpg'.format(self.sample_dir,
                               epoch, idx), np.array(img_all, dtype=np.uint8))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" %
                          (d_loss, g_loss))

    def render(self):
        pass

    def build_discriminator(self):
        return Discriminator(layout_dim=self.layout_dim, render=layout_point)

    def build_generator(self):
        return Generator(n_filters=512, output_dim=self.geometric_dim, n_component=self.n_component, n_class=self.n_class)

    def gradient_penalty(self, real, fake):
        alpha = tf.random.uniform(
            shape=[real.shape[0], 1, 1], minval=0.0, maxval=1.)
        interpolated = alpha * real + (1 - alpha) * fake
        with tf.GradientTape() as tape_p:
            tape_p.watch(interpolated)
            logit = self.D(interpolated)

        grad = tape_p.gradient(logit, interpolated)
        grad_norm = tf.norm(tf.reshape(grad, (real.shape[0], -1)), axis=1)

        return 10 * tf.reduce_mean(tf.square(grad_norm - 1.))

    def generator_loss(self, z):
        x = self.G(z, training=True)
        fake_score = self.D(x, training=True)
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_score, labels=tf.ones_like(tf.sigmoid(fake_score))))
        return g_loss

    def discriminator_loss(self, x, z):
        x_fake = self.G(z, training=True)
        true_score = self.D(x, training=True)
        fake_score = self.D(x_fake, training=True)
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=true_score, labels=tf.ones_like(tf.sigmoid(true_score))))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_score, labels=tf.zeros_like(tf.sigmoid(fake_score))))
        d_loss = d_loss_real + d_loss_fake
        return d_loss


batch_size = 64
n_component = 128
n_class = 1
geometric_dim = 2
# give approriate path
sample_dir = "../samples/MNIST_results"
gan = LayoutGAN(batch_size=batch_size, n_component=n_component, n_class=n_class, geometric_dim=geometric_dim,
                sample_dir=sample_dir, dataset_path="../data/mnist.npy")
gan.train()
