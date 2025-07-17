
# self-supervised motion-compensated reconstruction network

import os
import time
import pickle
import datetime
import merlintf
import tensorflow as tf
import utils.my_utils as utils
import tensorflow.keras.backend as K
import model.one_dim_callbacks as callbacks

from utils import mri, motion_mri
from model.complex_unet import ComplexUNet_2Dt
from data_loader.MC_SSL_DataLoader import CINE2DDataset


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


def get_motion_dc_layer():
    A = motion_mri.MulticoilMotionForwardOp(center=True, channel_dim_defined=True, coil_axis=-1)
    AH = motion_mri.MulticoilMotionAdjointOp(center=True, channel_dim_defined=True, coil_axis=-1)
    return merlintf.keras.layers.DCPM(A, AH, max_iter=5)


class Hybrid_Net(tf.keras.Model):
    def __init__(self, mode, num_iter=1, neighbour_frame=1, name='HybridNet'):
        super().__init__(name=name)
        self.S_end = num_iter
        self.mode = mode
        self.neighbour_frame = neighbour_frame

        # Image  block
        self.ImgNet = []
        self.tau = []
        # dc layer for image
        self.ImgDC = []

        for i in range(self.S_end):
            self.tau.append(Scalar(init=0.1))
            self.ImgNet.append(get_CNN())
            self.ImgDC.append(get_motion_dc_layer())

    def update_x(self, x, i, num_iter, constants_motion):
        # image network
        img_net = self.ImgNet[i]
        den = img_net(x)
        x = x - merlintf.complex_scale(self.tau[i](den), 1 / num_iter)

        # dc operation
        img_dc_layer = self.ImgDC[i]
        x = img_dc_layer([x]+constants_motion)  # (1, 5, 176, 132, 1)
        return x

    def img_ksp_loss(self, x1, x2, y1, y2, mask_1, mask_2, smaps):
        if x1.dtype != tf.complex64:
            x1 = tf.cast(x1, tf.complex64)
        if x2.dtype != tf.complex64:
            x2 = tf.cast(x2, tf.complex64)

        img_loss = K.mean(K.sum(tf.math.real(tf.math.conj(x1 - x2) * (x1 - x2)), axis=(1, 2, 3)), axis=(0, -1))

        # ksp calibration loss
        x1_ksp_mask2 = mri.MulticoilForwardOp(center=True)(x1, mask_2, smaps)
        ksp_loss_1 = K.mean(K.sum(tf.sqrt(tf.math.real(tf.math.conj((x1_ksp_mask2 - y2)) * (x1_ksp_mask2 - y2)) + 1e-9)))

        x2_ksp_mask1 = mri.MulticoilForwardOp(center=True)(x2, mask_1, smaps)
        ksp_loss_2 = K.mean(K.sum(tf.sqrt(tf.math.real(tf.math.conj((x2_ksp_mask1 - y1)) * (x2_ksp_mask1 - y1)) + 1e-9)))

        ksp_loss = ksp_loss_1 + ksp_loss_2
        return img_loss, ksp_loss

    def call(self, inputs):
        if self.mode == 'train':
            x1, y1_motion, M1_motion, smap_motion, u, x2, y2_motion, M2_motion, y1, y2, M1, M2, smap = inputs
            constants_motion_1 = [y1_motion, M1_motion, smap_motion, u]
            constants_motion_2 = [y2_motion, M2_motion, smap_motion, u]
            with tf.device("/gpu:0"):
                x1 = self.update_x(x1, i=0, num_iter=self.S_end, constants_motion=constants_motion_1)
            with tf.device("/gpu:1"):
                x2 = self.update_x(x2, i=0, num_iter=self.S_end, constants_motion=constants_motion_2)
            with tf.device("/gpu:2"):
                x1 = self.update_x(x1, i=1, num_iter=self.S_end, constants_motion=constants_motion_1)
            with tf.device("/gpu:3"):
                x2 = self.update_x(x2, i=1, num_iter=self.S_end, constants_motion=constants_motion_2)

            img_loss, ksp_loss = self.img_ksp_loss(x1, x2, y1, y2, M1, M2, smap)
            output = tf.stack((img_loss, ksp_loss), axis=0)
            return output

        if self.mode == 'pred':
            x, y, mask, smaps = inputs[:4]
            constants_motion = inputs[4:]
            x = self.update_x(x, i=0, num_iter=self.S_end, constants_motion=constants_motion)
            x = self.update_x(x, i=1, num_iter=self.S_end, constants_motion=constants_motion)
            return x


def main(train_min_R, train_max_R, val_min_R, val_max_R):
    # dataset
    ds_train = CINE2DDataset(start_R=2, min_R=train_min_R, max_R=train_max_R, neighbour_frame=1, mode='train', shuffle=True)
    ds_val = CINE2DDataset(start_R=2, min_R=val_min_R, max_R=val_max_R,  neighbour_frame=1, mode='val', shuffle=False)

    model = Hybrid_Net(num_iter=2, neighbour_frame=1, mode='train')

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer, loss=utils.get_loss('img_ksp_loss'), metrics=utils.get_metrics(),
                  run_eagerly=True)

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

    history = model.fit(ds_train, epochs=500, validation_data=ds_val, max_queue_size=1,
                        callbacks=callbacks.get_callbacks(ds_val, model, log_dir))

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
    main(train_min_R=2, train_max_R=24, val_min_R=12, val_max_R=12)
