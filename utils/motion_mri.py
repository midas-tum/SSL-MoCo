
import tensorflow as tf

from tensorflow.signal import fft2d, ifft2d
from my_warp import WarpForward, WarpAdjoint
from utils.fft import FFT2, IFFT2, complex_scale


class FFT2c(tf.keras.layers.Layer):
    def call(self, image, *args):  
        if len(image.shape) == 5:
            image = tf.expand_dims(image, axis=0)
        image = tf.transpose(image, perm=[0, 5, 1, 2, 3, 4])  
        dtype = tf.math.real(image).dtype
        axes = [tf.rank(image) - 2, tf.rank(image) - 1]  
        scale = tf.math.sqrt(tf.cast(tf.math.reduce_prod(tf.shape(image)[-2:]), dtype))
        kspace = complex_scale(tf.signal.fftshift(fft2d(tf.signal.ifftshift(image, axes=axes)), axes=axes), 1/scale)
        return tf.transpose(kspace, perm=[0, 2, 3, 4, 5, 1])  


class IFFT2c(tf.keras.layers.Layer):
    def call(self, kspace, *args):  
        if len(kspace.shape) == 5:
            kspace = tf.expand_dims(kspace, axis=0)
        kspace = tf.transpose(kspace, perm=[0, 5, 1, 2, 3, 4])
        axes = [tf.rank(kspace) - 2, tf.rank(kspace) - 1]
        dtype = tf.math.real(kspace).dtype
        scale = tf.math.sqrt(tf.cast(tf.math.reduce_prod(tf.shape(kspace)[-2:]), dtype))
        image = complex_scale(tf.signal.fftshift(ifft2d(tf.signal.ifftshift(kspace, axes=axes)), axes=axes), scale)
        return tf.transpose(image, perm=[0, 2, 3, 4, 5, 1])


class Smaps(tf.keras.layers.Layer):
    def __init__(self, coil_axis=-1):
        super().__init__()
        self.coil_axis = coil_axis

    def call(self, img, smaps):
        img = tf.cast(img, dtype=tf.complex64)
        smaps = tf.cast(smaps, dtype=tf.complex64)
        return tf.expand_dims(img, self.coil_axis) * smaps  


class SmapsAdj(tf.keras.layers.Layer):
    def __init__(self, coil_axis=-1):
        super().__init__()
        self.coil_axis = coil_axis

    def call(self, coilimg, smaps):
        coilimg = tf.cast(coilimg, dtype=tf.complex64)
        smaps = tf.cast(smaps, dtype=tf.complex64)
        return tf.reduce_sum(coilimg * tf.math.conj(smaps), self.coil_axis, keepdims=False)


class MaskKspace(tf.keras.layers.Layer):
    def call(self, kspace, mask):
        return complex_scale(kspace, mask)


class MulticoilForwardOp(tf.keras.layers.Layer):
    def __init__(self, center=True, coil_axis=-1, channel_dim_defined=True):
        super().__init__()
        if center:
            self.fft2 = FFT2c()
        else:
            self.fft2 = FFT2()
        self.mask = MaskKspace()
        self.smaps = Smaps(coil_axis=coil_axis)
        self.channel_dim_defined = channel_dim_defined

    def call(self, image, mask, smaps):
        if image.shape[0] == 1:
            image = tf.squeeze(image, axis=0)
        if self.channel_dim_defined:
            coilimg = self.smaps(img=image[..., 0], smaps=smaps)
        else:
            coilimg = self.smaps(img=image, smaps=smaps)  
        kspace = self.fft2(image=coilimg)  
        masked_kspace = self.mask(kspace=kspace, mask=mask)  # (1, t1, t_all, M, N, C)
        return masked_kspace


class MulticoilAdjointOp(tf.keras.layers.Layer):
    def __init__(self, center=True, coil_axis=-1, channel_dim_defined=True):
        super().__init__()
        self.mask = MaskKspace()
        if center:
            self.ifft2 = IFFT2c()
        else:
            self.ifft2 = IFFT2()
        self.adj_smaps = SmapsAdj(coil_axis=coil_axis)
        self.channel_dim_defined = channel_dim_defined

    def call(self, kspace, mask, smaps):
        masked_kspace = self.mask(kspace=kspace, mask=mask)  
        coilimg = self.ifft2(kspace=masked_kspace)  
        img = self.adj_smaps(coilimg=coilimg, smaps=smaps) 
        if self.channel_dim_defined:
            return tf.expand_dims(img, axis=-1)
        else:
            return img  


class MulticoilMotionForwardOp(tf.keras.layers.Layer):
    def __init__(self, center=True, coil_axis=-1, channel_dim_defined=True):
        super().__init__()
        self.W = WarpForward()
        self.A = MulticoilForwardOp(center=center, coil_axis=coil_axis, channel_dim_defined=False)
        self.channel_dim_defined = channel_dim_defined

    def call(self, x, mask, smaps, u):
        if self.channel_dim_defined:
            x = self.W(x=x[..., 0], u=u)
        else:
            x = self.W(x=x, u=u)

        y = self.A(image=x, mask=mask, smaps=smaps)  
        return y


class MulticoilMotionAdjointOp(tf.keras.layers.Layer):
    def __init__(self, center=False, coil_axis=-1, channel_dim_defined=True):
        super().__init__()
        self.AH = MulticoilAdjointOp(center=center, coil_axis=coil_axis, channel_dim_defined=False)
        self.WH = WarpAdjoint()
        self.channel_dim_defined = channel_dim_defined

    def call(self, y, mask, smaps, u):
        x = self.AH(kspace=y, mask=mask, smaps=smaps)
        x = self.WH(x=x, u=u)
      
        if self.channel_dim_defined:
            return tf.expand_dims(x, -1)
        else:
            return x


class MulticoilForwardOp_NoMask(tf.keras.layers.Layer):
    def __init__(self, center=True, coil_axis=-1, channel_dim_defined=True):
        super().__init__()
        if center:
            self.fft2 = FFT2c()
        else:
            self.fft2 = FFT2()
        self.smaps = Smaps(coil_axis=coil_axis)
        self.channel_dim_defined = channel_dim_defined

    def call(self, image, mask, smaps):
        if self.channel_dim_defined:
            coilimg = self.smaps(img=image[..., 0], smaps=smaps)
        else:
            coilimg = self.smaps(img=image, smaps=smaps)
        kspace = self.fft2(image=coilimg)
        return kspace


class MulticoilMotionForwardOp_NoMask(tf.keras.layers.Layer):
    def __init__(self, center=True, coil_axis=-1, channel_dim_defined=True):
        super().__init__()
        self.W = WarpForward()
        self.A = MulticoilForwardOp_NoMask(center=center, coil_axis=coil_axis, channel_dim_defined=False)
        self.channel_dim_defined = channel_dim_defined

    def call(self, x, mask, smaps, u):
        if self.channel_dim_defined:
            x = self.W(x=x[..., 0], u=u)
        else:
            x = self.W(x=x, u=u)

        y = self.A(image=x, mask=mask, smaps=smaps)
        return y


class MulticoilAdjointOp_NoMask(tf.keras.layers.Layer):
    def __init__(self, center=True, coil_axis=-1, channel_dim_defined=True):
        super().__init__()
        if center:
            self.ifft2 = IFFT2c()
        else:
            self.ifft2 = IFFT2()
        self.adj_smaps = SmapsAdj(coil_axis=coil_axis)
        self.channel_dim_defined = channel_dim_defined

    def call(self, kspace, mask, smaps):
        coilimg = self.ifft2(kspace=kspace)
        img = self.adj_smaps(coilimg=coilimg, smaps=smaps)
        if self.channel_dim_defined:
            return tf.expand_dims(img, axis=-1)
        else:
            return img


class MulticoilMotionAdjointOp_NoMask(tf.keras.layers.Layer):
    def __init__(self, center=False, coil_axis=-1, channel_dim_defined=True):
        super().__init__()
        self.AH = MulticoilAdjointOp(center=center, coil_axis=coil_axis, channel_dim_defined=False)
        self.WH = WarpAdjoint()
        self.channel_dim_defined = channel_dim_defined

    def call(self, y, mask, smaps, u):
        x = self.AH(kspace=y, mask=mask, smaps=smaps)
        x = self.WH(x=x, u=u)

        if self.channel_dim_defined:
            return tf.expand_dims(x, -1)
        else:
            return x
