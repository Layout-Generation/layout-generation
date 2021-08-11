import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
import random
from utils import *
from modules import *
import matplotlib.pyplot as plt
from tensorflow.keras import initializers


os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class LayoutGAN(object):
    def __init__(self, geometric_dim=2, n_class=1, batch_size=64, n_component=128, layout_dim=(28, 28), d_lr=1e-5, g_lr=1e-5, update_ratio=2, clip_value=0.1, dataset_name='default', dataset_path='./data/pre_data_cls.npy', checkpoint_dir=None, sample_dir=None):
        self.batch_size = batch_size
        self.n_component = n_component
        self.n_class = n_class
        self.geometric_dim = geometric_dim
        self.layout_dim = layout_dim
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.data = np.load(dataset_path)
        self.data = self.data[:70000]
        self.build_model(d_lr, g_lr)
        self.sample_dir = sample_dir
        self.update_ratio = update_ratio
        self.clip_value = clip_value
        self.epochs = 50

    def build_model(self, d_lr, g_lr):
        self.G = self.build_generator()
        self.D = self.build_discriminator()
        epoch_step = len(self.data) // self.batch_size
        dlr = tf.keras.optimizers.schedules.ExponentialDecay(
            d_lr, decay_steps=20*epoch_step, decay_rate=0.1, staircase=True)
        self.d_opt = tf.keras.optimizers.Adam(dlr)
        self.g_opt = tf.keras.optimizers.Adam(dlr)

    def step(self, real_data, noise, training=True, step=0):
        with tf.GradientTape() as disc_tape:
            disc_loss = self.discriminator_loss(real_data, noise)
            if(training):
                gradients_of_discriminator = disc_tape.gradient(
                    disc_loss, self.D.trainable_variables)
                self.d_opt.apply_gradients(
                    zip(gradients_of_discriminator, self.D.trainable_variables))

        for i in range(self.update_ratio):
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
        sample_z_bbox = np.random.normal(0.5, 0.15, (self.batch_size, 9, 4))
        sample_z_cls = np.identity(
            5)[np.random.randint(5, size=(self.batch_size, 9))]
        sample_z = np.concatenate([sample_z_bbox, sample_z_cls], axis=-1)
        counter = 1
        start_time = time.time()

        for epoch in range(self.epochs):
            np.random.shuffle(self.data)
            batch_idxs = len(self.data) // self.batch_size

            for idx in range(0, batch_idxs):
                batch = self.data[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_images = np.array(batch).astype(np.float32)

                batch_z_bbox = np.random.normal(
                    0.5, 0.15, (self.batch_size, 9, 4))
                batch_z_cls = np.identity(
                    5)[np.random.randint(5, size=(self.batch_size, 9))]
                batch_z = np.concatenate([batch_z_bbox, batch_z_cls], axis=-1)

                g_loss, d_loss = self.step(batch_images, batch_z, step=idx)
                counter += 1
                if np.mod(counter, 50) == 0:

                    current_decayed_lr = self.d_opt._decayed_lr(
                        tf.float32).numpy()
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, lr:%.3E, d_loss: %.4f, g_loss: %.4f"
                          % (epoch, idx, batch_idxs, time.time()-start_time, current_decayed_lr, d_loss, g_loss))

                if np.mod(counter, 500) == 0:
                    G_samples = self.G(sample_z, training=False)
                    path = '{}/train_{:02d}_{:04d}_{:2.4f}_{:2.4f}.jpg'.format(
                        self.sample_dir, epoch, idx, d_loss, g_loss)
                    change = convert_to_cxywh(np.array(G_samples))
                    plot_layouts(change, colors=colors,
                                 class_names=class_names, path=path)
                    g_loss, d_loss = self.step(
                        sample_inputs, sample_z, training=False)
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" %
                          (d_loss, g_loss))

    def render(self):
        pass

    def build_discriminator(self):
        return Discriminator(layout_dim=self.layout_dim, render=layout_bbox)

    def build_generator(self):
        return Generator(n_filters=1024, output_dim=self.geometric_dim, n_component=self.n_component, n_class=self.n_class)

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


if __name__ == '__main__':
    batch_size = 64
    n_component = 9
    n_class = 5
    geometric_dim = 4
    gan = LayoutGAN(batch_size=batch_size, n_component=n_component,
                    n_class=n_class, layout_dim=(60, 40),
                    geometric_dim=geometric_dim,
                    sample_dir="dataset/AAAAA/Reproduced_LayoutGAN/samples",
                    dataset_path="dataset/AAAAA/Reproduced_LayoutGAN/data/sorted_c1publay.npy")

    gan.train()
