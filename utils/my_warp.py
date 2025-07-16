
import optotf.warp
import tensorflow as tf


class Warp(tf.keras.layers.Layer):
    def __init__(self, channel_last=True, mode='zeros'):
        super().__init__()
        self.channel_last = channel_last
        self.op = optotf.warp.warp_2d
        self.mode = mode

    def call(self, x, u):

        if self.channel_last:
            x = tf.transpose(x, [0, 3, 1, 2])
        if x.dtype == tf.complex64 or x.dtype == tf.complex128:
            out = tf.complex(self.op(tf.math.real(x), u, tf.math.real(x), mode=self.mode)[0],
                             self.op(tf.math.imag(x), u, tf.math.real(x), mode=self.mode)[0])
        else:
            out, _ = self.op(x, u, x, mode=self.mode)

        if self.channel_last:
            out = tf.transpose(out, [0, 2, 3, 1])
        out = tf.stop_gradient(out)
        return out


class WarpTranspose(tf.keras.layers.Layer):
    def __init__(self, channel_last=True, mode='zeros'):
        super().__init__()
        self.channel_last = channel_last
        self.op = optotf.warp.warp_2d_transpose
        self.mode = mode

    def call(self, grad_out, u, x):
        if self.channel_last:
            grad_out = tf.transpose(grad_out, [0, 3, 1, 2])
            x = tf.transpose(x, [0, 3, 1, 2])

        '''
        # visulization of flow fields
        flo = flow_vis.flow_to_color(u[0])
        plt.imshow(flo)
        plt.axis("off")
        plt.show()
        '''

        if grad_out.dtype == tf.complex64 or grad_out.dtype == tf.complex128:
            grad_x = tf.complex(self.op(tf.math.real(grad_out), u, tf.math.real(x), mode=self.mode)[0],
                                self.op(tf.math.imag(grad_out), u, tf.math.real(x), mode=self.mode)[0])
        else:
            grad_x, _ = self.op(grad_out, u, x, mode=self.mode)

        if self.channel_last:
            grad_x = tf.transpose(grad_x, [0, 2, 3, 1])
        grad_x = tf.stop_gradient(grad_x)
        return grad_x


class WarpForward(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.W = Warp(channel_last=False)

    def call(self, x, u):
        out_shape = tf.shape(u)[:-1]
        M, N = tf.shape(u)[-3:-1]
        x = tf.repeat(tf.expand_dims(x, -3), repeats=tf.shape(u)[-4], axis=-3)
        x = tf.reshape(x, (-1, 1, M, N))  # [batch, frames * frames_all, 1, M, N]
        u = tf.reshape(u, (-1, M, N, 2))  # [batch, frames * frames_all, M, N, 2]
        Wx = self.W(x, u)
        return tf.reshape(Wx, out_shape)


class WarpAdjoint(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.WH = WarpTranspose(channel_last=False) 

    def call(self, x, u):
        out_shape = tf.shape(u)[:-1]
        M, N = tf.shape(u)[-3:-1]
        x = tf.reshape(x, (-1, 1, M, N))  # [batch * frames * frames_all, 1, M, N]
        u = tf.reshape(u, (-1, M, N, 2))  # [batch * frames * frames_all, M, N, 2]
        x_warpT = self.WH(x, u, x)
        x_warpT = tf.reshape(x_warpT, out_shape)
        x_warpT = tf.math.reduce_sum(x_warpT, -3)
        return x_warpT
