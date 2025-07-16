
import tensorflow as tf
from tensorflow.signal import fft2d, ifft2d


def complex_scale(x, scale):
    return tf.complex(tf.math.real(x) * scale, tf.math.imag(x) * scale)
  

class IFFT2c(tf.keras.layers.Layer):
    def call(self, kspace, *args):
        if len(kspace.shape) == 4:
            kspace = tf.expand_dims(kspace, axis=0)
        kspace = tf.transpose(kspace, perm=[0, 1, 4, 2, 3])
        axes = [tf.rank(kspace) - 2, tf.rank(kspace) - 1]  # 3,4
        dtype = tf.math.real(kspace).dtype
        scale = tf.math.sqrt(tf.cast(tf.math.reduce_prod(tf.shape(kspace)[-2:]), dtype))
        image = complex_scale(tf.signal.fftshift(ifft2d(tf.signal.ifftshift(kspace, axes=axes)), axes=axes), scale)
        return tf.transpose(image, perm=[0, 1, 3, 4, 2])


class FFT2c(tf.keras.layers.Layer):
    def call(self, image, *args):
        if len(image.shape) == 4:
            image = tf.expand_dims(image, axis=0)
        if len(image.shape) == 6:
            image = tf.squeeze(image, axis=0)
        image = tf.transpose(image, perm=[0, 1, 4, 2, 3])
        dtype = tf.math.real(image).dtype
        axes = [tf.rank(image) - 2, tf.rank(image) - 1]  # axes have to be positive...
        scale = tf.math.sqrt(tf.cast(tf.math.reduce_prod(tf.shape(image)[-2:]), dtype))
        kspace = complex_scale(tf.signal.fftshift(fft2d(tf.signal.ifftshift(image, axes=axes)), axes=axes), 1/scale)
        return tf.transpose(kspace, perm=[0, 1, 3, 4, 2])


class IFFT2(tf.keras.layers.Layer):
    def call(self, kspace, *args):
        kspace = tf.transpose(kspace, perm=[0, 1, 4, 2, 3])
        dtype = tf.math.real(kspace).dtype
        scale = tf.math.sqrt(tf.cast(tf.math.reduce_prod(tf.shape(kspace)[-2:]), dtype))
        image = complex_scale(ifft2d(kspace), scale)
        return tf.transpose(image, perm=[0, 1, 3, 4, 2])


class FFT2(tf.keras.layers.Layer):
    def call(self, image, *args):
        image = tf.transpose(image, perm=[0, 1, 4, 2, 3])
        dtype = tf.math.real(image).dtype
        scale = tf.math.sqrt(tf.cast(tf.math.reduce_prod(tf.shape(image)[-2:]), dtype))
        kspace = complex_scale(fft2d(image), 1/scale)
        return tf.transpose(kspace, perm=[0, 1, 3, 4, 2])
