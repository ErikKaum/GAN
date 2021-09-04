import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

LATENT_DIM = 128
EPOCHS = 1
IMG_WIDTH = 64
IMG_HEIGHT = 64
CHANNELS = 3

def make_disciminator(number_of_blocks = 2):
    inputs = keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS), name='input_layer')
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same', name=f'conv_2d_block_1')(inputs)
    x = layers.LeakyReLU(alpha=0.2, name='leakyReLu_block_1')(x)

    for block in range(number_of_blocks):
        x = layers.Conv2D(128, kernel_size=4, strides=2, padding='same', name=f'conv_2d_block_{block+2}')(x)
        x = layers.LeakyReLU(alpha=0.2, name=f'leakyReLu_block_{block+2}')(x)

    x = layers.Flatten(name='flatten')(x)
    x = layers.Dropout(0.2, name='dropout')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output_layer')(x)

    model = keras.Model(inputs, outputs, name='discriminator')
    return model

def make_generator(number_of_blocks = 3):
    inputs = keras.Input(shape=((LATENT_DIM, )), name='input_layer')
    x = layers.Dense((8 * 8 * LATENT_DIM), name='dense')(inputs)
    x = layers.Reshape((8, 8, LATENT_DIM), name='reshape_layer')(x)
    
    for block in range(number_of_blocks):
        x = layers.Conv2DTranspose((LATENT_DIM*(block+1)), kernel_size=4, strides=2, padding="same", name=f'conv_2d_transpose_block_{block+1}')(x)
        x = layers.LeakyReLU(alpha=0.2, name =f'leakyReLu_block_{block+1}')(x)
    
    outputs = layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name='generator')
    return model

class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }

class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = keras.preprocessing.image.array_to_img(generated_images[i])
            img.save("generated_img_%03d_%d.png" % (epoch, i))


def main():
    discriminator = make_disciminator()
    generator = make_generator()

    discriminator.summary()
    generator.summary()



    # boilerplate edit later
    # gan = GAN(discriminator=discriminator, generator=generator, latent_dim=LATENT_DIM)
    # gan.compile(
    #     d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    #     g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    #     loss_fn=keras.losses.BinaryCrossentropy(),
    # )

    # gan.fit(
    #     dataset, epochs=EPOCHS, callbacks=[GANMonitor(num_img=10, latent_dim=LATENT_DIM)]
    # )

if __name__ == '__main__':
    main()


