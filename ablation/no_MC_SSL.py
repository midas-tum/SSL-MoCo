
import os
import time
import pickle
import merlintf
import datetime
import tensorflow as tf
import utils.my_utils as utils
import tensorflow.keras.backend as K
import model.one_dim_callbacks as callbacks

from utils import mri, DCPM_mri
from model.complex_unet import ComplexUNet_2Dt
from data_loader.no_MC_SSL_DataLoader import CINE2DDataset


class Scalar(tf.keras.layers.Layer):
    def __init__(self, init=1.0, train_scale=1.0, name=None):
        super().__init__(name=name)
        self.init = init
        self.train_scale = train_scale

    def build(self, input_shape):
        self._weight = self.add_weight(name='scalar',
                                       shape=(1,),
                                       constraint=tf.keras.constraints.NonNeg(),
                                       initializer=tf.keras.initializers.Constant(self.init))

    @property
    def weight(self):
        return self._weight * self.train_scale

    def call(self, inputs):
        return merlintf.complex_scale(inputs, self.weight)        


def get_CNN():
    return ComplexUNet_2Dt(dim='2Dt', filters=12, kernel_size_2d=(1, 5, 5), kernel_size_t=(3, 1, 1), downsampling='mp',
                           num_level=2, num_layer_per_level=1, activation_last=None)


def get_dc_layer():
    A = DCPM_mri.MulticoilForwardOp(center=True, coil_axis=-3, channel_dim_defined=True)
    AH = DCPM_mri.MulticoilAdjointOp(center=True, coil_axis=-3, channel_dim_defined=True)
    return merlintf.keras.layers.DCPM(A, AH, max_iter=5)


class UNET2dt_unrolled(tf.keras.Model):
    def __init__(self, num_iter, mode, name='UNET2dt_unrolled'):
        super().__init__(name=name)
        self.S_end = num_iter
        self.mode = mode
        self.tau = []
        self.ImgNet = []
        self.ImgDC = []

        for i in range(self.S_end):
            self.ImgNet.append(get_CNN())
            self.ImgDC.append(get_dc_layer())
            self.tau.append(Scalar(init=0.1))

    def img_dcpm(self, i, x, constans):
        # (1, t, x, y, c) -> (1, t, c, x, y)
        x = tf.transpose(x, (0, 1, 4, 2, 3))
        sub_y = tf.transpose(constans[0], (0, 1, 4, 2, 3))
        mask = tf.transpose(constans[1], (0, 1, 4, 2, 3))
        smap = tf.transpose(constans[2], (0, 1, 4, 2, 3))
        img_dc_layer = self.ImgDC[i]
        x = img_dc_layer([x] + [sub_y, mask, smap])
        x = tf.transpose(x, (0, 1, 3, 4, 2))
        return x

    def img_ksp_loss(self, x1, x2, y1, y2, mask_1, mask_2, smaps):
        # image loss between x1, x2
        x1 = tf.cast(x1, tf.complex64)
        x2 = tf.cast(x2, tf.complex64)
        diff = (x1 - x2)
        img_loss = K.mean(K.sum(tf.math.real(tf.math.conj(diff) * diff), axis=(1, 2, 3)), axis=(0, -1))

        # ksp calibration loss
        x1_ksp_mask2 = mri.MulticoilForwardOp(center=True)(x1, mask_2, smaps)
        ksp_loss_1 = K.mean(K.sum(tf.sqrt(tf.math.real(tf.math.conj((x1_ksp_mask2 - y2)) * (x1_ksp_mask2 - y2)) + 1e-9)))

        x2_ksp_mask1 = mri.MulticoilForwardOp(center=True)(x2, mask_1, smaps)
        ksp_loss_2 = K.mean(K.sum(tf.sqrt(tf.math.real(tf.math.conj((x2_ksp_mask1 - y1)) * (x2_ksp_mask1 - y1)) + 1e-9)))

        ksp_loss = ksp_loss_1 + ksp_loss_2
        return img_loss, ksp_loss

    def update_x(self, x, i, num_iter, constants):
        # image network
        img_net = self.ImgNet[i]
        den = img_net(x)
        x = x - merlintf.complex_scale(self.tau[i](den), 1 / num_iter)

        # dc operation
        x = self.img_dcpm(i, x, constants)
        return x

    def call(self, inputs):
        if self.mode == 'train':
            x1, y1, mask_1, x2, y2, mask_2, smaps = inputs
            with tf.device("/gpu:0"):
                x1 = self.update_x(x1, i=0, num_iter=self.S_end, constants=[y1, mask_1, smaps])
                x2 = self.update_x(x2, i=0, num_iter=self.S_end, constants=[y2, mask_2, smaps])
            with tf.device("/gpu:1"):
                x1 = self.update_x(x1, i=1, num_iter=self.S_end, constants=[y1, mask_1, smaps])
                x2 = self.update_x(x2, i=1, num_iter=self.S_end, constants=[y2, mask_2, smaps])

            img_loss, ksp_loss = self.img_ksp_loss(x1, x2, y1, y2, mask_1, mask_2, smaps)
            output = tf.stack((img_loss, ksp_loss), axis=0)
            return output

        if self.mode == 'pred':
            x = inputs[0]
            constants = inputs[1:]
            x = self.update_x(x, i=0, num_iter=self.S_end, constants=constants)
            x = self.update_x(x, i=1, num_iter=self.S_end, constants=constants)
            return x


def main(start_R, train_min_R, train_max_R, val_min_R, val_max_R):
    # dataset
    ds_train = CINE2DDataset(start_R, train_min_R, train_max_R, mode='train_no_174', shuffle=True)
    ds_val = CINE2DDataset(start_R, val_min_R, val_max_R, mode='val', shuffle=False)

    model = UNET2dt_unrolled(num_iter=2, mode='train')

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer, loss=utils.get_loss('img_ksp_loss'), run_eagerly=True)

    # initialize model to print model summary
    inputs, targets = ds_train.__getitem__(0)

    start = time.time()
    outputs = model.predict(inputs)
    end = time.time()
    print(end - start)
    print(model.summary())

    fold = 26
    exp_dir = 'experiments/fold__%d__/R__%d__%d' % (fold, train_min_R, train_max_R)
    log_dir = os.path.join(exp_dir, model.name + '-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    history = model.fit(ds_train, epochs=300, validation_data=ds_val, max_queue_size=4,
                        callbacks=callbacks.get_callbacks(ds_val, model, log_dir))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    main(start_R=2, train_min_R=2, train_max_R=24, val_min_R=12, val_max_R=12)
