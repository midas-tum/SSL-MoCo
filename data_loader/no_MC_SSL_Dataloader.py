
import random
import merlintf
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.signal import fft2d, ifft2d


def FFT2c(image): 
    image = np.transpose(image, axes=[0, 1, 4, 2, 3])  
    axes = [3, 4]
    scale = np.sqrt(np.prod(image.shape[-2:]).astype(np.float64))
    kspace = merlintf.complex_scale(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image, axes=axes),
                                                                axes=axes), axes=axes), 1/scale)
    return np.transpose(kspace, axes=[0, 1, 3, 4, 2])


def IFFT2c(kspace):
    kspace = np.transpose(kspace, axes=[0, 1, 4, 2, 3])  # (12, 25, 30, 176, 132)
    axes = [3, 4]
    scale = np.sqrt(np.prod(kspace.shape[-2:]).astype(np.float64))
    image = merlintf.complex_scale(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace, axes=axes),
                                                                axes=axes), axes=axes), scale)
    return np.transpose(image, axes=[0, 1, 3, 4, 2])


class CINE2DDataset(tf.keras.utils.Sequence):
    """CINE data training set."""

    def __init__(self, start_R, min_R, max_R, mode='train', transform=None, shuffle=True):
        self.transform = transform
        self.mode = mode
        self.batch_size = 1
        self.min_R = min_R
        self.max_R = max_R
        self.start_R = start_R

        if self.mode == 'train':
            data_set = pd.read_csv('/home/studxusiy1/mr_recon/03_cine2dt/subj_dataset/CINE2D_h5_train.csv')
        elif self.mode == 'val' or self.mode == 'test':
            data_set = pd.read_csv('/home/studxusiy1/mr_recon/03_cine2dt/subj_dataset/CINE2D_h5_val.csv')

        self.data_set = []

        self.shuffle = shuffle

        for i in range(len(data_set)):
            subj = data_set.iloc[i]
            fname = subj.filename
            nPE = subj.nPE
            num_slices = subj.SLICE_DIM

            # specify the slices
            minsl = 0
            maxsl = num_slices - 1
            assert minsl <= maxsl

            attrs = {'nPE': nPE, 'metadata': subj.to_dict()}
            self.data_set += [(fname, minsl, maxsl, attrs)]

    def __len__(self):
        return len(self.data_set)

    def on_epoch_end(self):
        """Updates indeces after each epoch"""
        self.indeces = np.arange(len(self.data_set))
        if self.shuffle == True:
            np.random.shuffle(self.indeces)

    def __getitem__(self, idx):
        fname, minsl, maxsl, attrs = self.data_set[idx]
        fname = fname.split('.')[0]

        # according to batchsize, random choose slices
        print('selecting slices as one batch...')
        slice_range = np.arange(minsl, maxsl + 1)
        slice_prob = np.ones_like(slice_range, dtype=float)
        slice_prob /= slice_prob.sum()
        slidx = list(np.sort(np.random.choice(
            slice_range,
            min(self.batch_size, maxsl + 1 - minsl),
            p=slice_prob,
            replace=False,
        )))

        # load normalized image: norm_img: (z, t, x, y, 1)
        norm_imgc = np.load('norm_img_%s.txt.npy' % fname)
        batch_imgc = norm_imgc[slidx]  # (1, t, x, y, 1)

        # load coil-compressed averaged smap, (z, 1, x, y, c)
        cc_smap = np.load('cc_smap_15_%s.txt.npy' % fname)
        batch_smaps = cc_smap[slidx]  # (1, 1, x, y, c)

        # mask R=2:
        p = batch_imgc.shape[3]
        R = self.start_R
        # sd = 10
        mask = np.loadtxt("mask_VISTA_%dx%d_acc%d_%d.txt" % (p, 25, R, sd), dtype=int, delimiter=",")  # (y, 25)

        # mask 1:
        p = batch_imgc.shape[3]
        R_1 = random.randint(self.min_R, self.max_R)
        sd_1 = random.randint(1, 20)
        mask_1 = np.loadtxt("mask_VISTA_%dx%d_acc%d_%d.txt" % (p, 25, R_1, sd_1), dtype=int, delimiter=",")  
        mask_1 = mask_1 * mask
        mask_1 = np.expand_dims(np.transpose(mask_1), (0, 2, 4)) 

        # mask_2
        R_2 = random.randint(self.min_R, self.max_R)
        sd_2 = random.randint(1, 20)
        # ensure mask_2 differ from mask_1
        while R_1 == R_2 and sd_1 == sd_2:
            R_2 = random.randint(self.min_R, self.max_R)
            sd_2 = random.randint(1, 20)
        mask_2 = np.loadtxt("mask_VISTA_%dx%d_acc%d_%d.txt" % (p, 25, R_2, sd_2), dtype=int, delimiter=",")
        mask_2 = mask_2 * mask
        mask_2 = np.expand_dims(np.transpose(mask_2), (0, 2, 4))

        # create batch k-space
        imgccoil = batch_imgc * batch_smaps  # (1, t, x, y, c)
        coilkspace = FFT2c(imgccoil) 

        # apply mask
        masked_kspace_1 = mask_1 * coilkspace  
        masked_kspace_2 = mask_2 * coilkspace
        masked_coilimg_1 = IFFT2c(masked_kspace_1)  
        masked_coilimg_2 = IFFT2c(masked_kspace_2)
        masked_img_1 = np.expand_dims(np.sum(masked_coilimg_1 * np.conj(batch_smaps), -1), axis=-1)
        masked_img_2 = np.expand_dims(np.sum(masked_coilimg_2 * np.conj(batch_smaps), -1), axis=-1)

        masked_img_1 = tf.cast(masked_img_1, tf.complex64)
        masked_img_2 = tf.cast(masked_img_2, tf.complex64)
        masked_kspace_1 = tf.cast(masked_kspace_1, tf.complex64)
        masked_kspace_2 = tf.cast(masked_kspace_2, tf.complex64)
        mask_1 = mask_1.astype(np.float64)
        mask_2 = mask_2.astype(np.float64)
        input_smaps = tf.cast(batch_smaps, tf.complex64)

        label = tf.zeros(shape=masked_img_1.shape, dtype=tf.complex64)

        return [masked_img_1, masked_kspace_1, mask_1, masked_img_2, masked_kspace_2, mask_2, input_smaps], label
      
