
import os
import glob
import merlintf
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def get_loss(loss):
    return eval(loss)


def get_metrics():
    return [loss_rmse, loss_abs_mse, loss_abs_mae]


@tf.function
def loss_rmse(y_true, y_pred):
    diff = (tf.cast(merlintf.complex_abs(y_true), tf.float64) - tf.cast(merlintf.complex_abs(y_pred), tf.float64))
    nominator = tf.cast(K.sum(tf.math.real(tf.math.conj(diff) * diff), axis=(1, 2, 3, 4)), tf.float64)
    denominator = tf.cast(K.sum(tf.math.real(tf.math.conj(y_true) * y_true), axis=(1, 2, 3, 4)), tf.float64)
    return K.mean(K.sqrt(tf.math.real(nominator / denominator)))


@tf.function
def loss_abs_mse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.complex64)
    y_ped = tf.cast(y_pred, tf.complex64)
    diff = (merlintf.complex_abs(y_true) - merlintf.complex_abs(y_pred))
    return K.mean(K.sum(tf.math.real(tf.math.conj(diff) * diff), axis=(1, 2, 3)), axis=(0, -1))


@tf.function
def loss_abs_mae(y_true, y_pred):
    y_true = tf.cast(y_true, tf.complex64)
    y_pred = tf.cast(y_pred, tf.complex64)
    diff = (merlintf.complex_abs(y_true) - merlintf.complex_abs(y_pred))
    return K.mean(K.sum(tf.sqrt(tf.math.real(tf.math.conj(diff) * diff) + 1e-9), axis=(1, 2, 3)), axis=(0, -1))
  
