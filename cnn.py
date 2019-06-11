#!/usr/bin/env python3
from collections import namedtuple

import numpy as np
import tensorflow as tf


# The neural network model
class Network:
    def __init__(self, input_shape = None, epochs = 5, batch_size = 10, logdir = "logs/"):
        self.epochs = epochs
        self.batch_size = batch_size
        self.logdir = logdir
        self.input_shape = input_shape
        self.build()


    def build(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape)

        hidden = inputs
        hidden = self.getCBLayer(filters=128, kernel_size=3, strides=1, padding='same', inputs=hidden)
        hidden = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(hidden)
        hidden = tf.keras.layers.Flatten()(hidden)
        hidden = tf.keras.layers.Dense(30)(hidden)
        hidden = tf.keras.layers.Dropout(0.2)(hidden)
        outputs = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(hidden)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()]
        )

        self.tb_callback = tf.keras.callbacks.TensorBoard(self.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None

    def getCBLayer(self, filters, kernel_size, strides, padding, inputs):
        new_layer = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding,
                                           use_bias=False)(inputs)
        new_layer = tf.keras.layers.BatchNormalization()(new_layer)
        new_layer = tf.keras.layers.ReLU()(new_layer)
        return new_layer

    def fit(self, train_data, train_target, val_data, val_target):
        self.model.fit(
             train_data, train_target,
             batch_size=self.batch_size, epochs=self.epochs,
             validation_data=(val_data, val_target),
        )

    def predict(self, data):
        predictions = self.model.predict(data)
        # TODO: round predictions
        return predictions

if __name__ == "__main__":
    # Fix random seeds
    np.random.seed(42)
    tf.random.set_seed(42)

    # Set arguments
    args = namedtuple("args", "batch_size, epochs, logdir")
    args.batch_size = 1000
    args.epochs = 16
    args.logdir = "logs"


    # Load data
    # TODO

    # Create the network and train
    network = Network(args)
    network.train(None, args) # TODO
    network.predict(None, args) # TODO