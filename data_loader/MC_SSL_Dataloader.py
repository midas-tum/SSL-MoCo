
import random
import merlintf
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.signal import fft2d, ifft2d


def complex_scale(x, scale):
    return np.real(x) * scale + 1j * np.imag(x) * scale


def FFT2c(image):  # image: (12, 25, 176, 132, 15)
    axes = [2, 3]
    scale = np.sqrt(np.prod(image.shape[2:4]).astype(np.float64))
    kspace = complex_scale(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image, axes=axes), axes=axes), axes=axes), 1/scale)
    return kspace


def IFFT2c(kspace):
    axes = [2, 3]
    scale = np.sqrt(np.prod(kspace.shape[2:4]).astype(np.float64))
    image = complex_scale(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace, axes=axes), axes=axes), axes=axes), scale)
    return image


def get_neighboring_frames(n_frames, t, num_f):
    start_frame, end_frame = t - num_f, t + num_f + 1
    range_f = np.arange(start_frame, end_frame)
    range_f[range_f < 0] += n_frames
    range_f[range_f > (n_frames-1)] -= n_frames
    return range_f


class CINE2DDataset(tf.keras.utils.Sequence):
    """CINE data training set."""

    def __init__(self, start_R, min_R, max_R, neighbour_frame, mode='train', transform=None, shuffle=True):
        self.transform = transform
        self.mode = mode
        self.batch_size = 1
        self.min_R = min_R
        self.max_R = max_R
        self.start_R = start_R
        self.neighbour_frame = neighbour_frame

        if self.mode == 'train':
            data_set = pd.read_csv('CINE2D_h5_train.csv')
        elif self.mode == 'val' or self.mode == 'test':
            data_set = pd.read_csv('CINE2D_h5_val.csv')

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


    def get_neighbour_idx(self, neighbour_frame):
        # Construct an index matrix to represent the index of each frame and its adjacent frames
        # Create an index matrix of (T, 2*num_f + 1)
        indices = np.arange(25)[:, None] + np.arange(-neighbour_frame, neighbour_frame + 1)
        # Handle out-of-range indexes and ensure frames are circular
        indices = np.mod(indices, 25)  # Make sure all indices are in the range [0, T-1]
        return indices


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
        batch_imgc = np.load('norm_img_%s.txt.npy' % fname, mmap_mode='r')[slidx]

        # load coil-compressed averaged smap, (z, 1, x, y, c)
        batch_smaps = np.load('cc_smap_15_%s.txt.npy' % fname, mmap_mode='r')[slidx]

        # initially undersampling mask mask R=2 (simulate the situation where we only have the undersampled data):
        p = batch_imgc.shape[3]
        R = self.start_R
        sd = 10
        mask = np.loadtxt("mask_VISTA_%dx%d_acc%d_%d.txt" % (p, 25, R, sd), dtype=int,
            delimiter=",")  # (132, 25)

        # mask 1:
        p = batch_imgc.shape[3]
        R_1 = random.randint(self.min_R, self.max_R)
        sd_1 = random.randint(1, 20)
        mask_1 = np.loadtxt("mask_VISTA_%dx%d_acc%d_%d.txt" % (p, 25, R_1, sd_1), dtype=int,
            delimiter=",")  # (132, 25)
        mask_1 = mask_1 * mask
        mask_1 = np.expand_dims(np.transpose(mask_1), (0, 2, 4)).astype(np.float64)  # (1, t, 1, x, 1)

        # mask_2
        R_2 = random.randint(self.min_R, self.max_R)
        sd_2 = random.randint(1, 20)
        # ensure mask_2 differ from mask_1
        while R_1 == R_2 and sd_1 == sd_2:
            R_2 = random.randint(self.min_R, self.max_R)
            sd_2 = random.randint(1, 20)
        mask_2 = np.loadtxt("mask_VISTA_%dx%d_acc%d_%d.txt" % (p, 25, R_2, sd_2), dtype=int,
            delimiter=",")
        mask_2 = mask_2 * mask
        mask_2 = np.expand_dims(np.transpose(mask_2), (0, 2, 4)).astype(np.float64)

        # create batch k-space
        imgccoil = batch_imgc * batch_smaps  
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

        neighbour_index = self.get_neighbour_idx(neighbour_frame=self.neighbour_frame)  # shape: (25, 3)
        masked_kspace_1_motion = masked_kspace_1[:, neighbour_index, :, :, :]
        masked_kspace_1_motion = tf.cast(masked_kspace_1_motion, tf.complex64)
        masked_kspace_1 = tf.cast(masked_kspace_1, tf.complex64)

        masked_kspace_2_motion = masked_kspace_2[:, neighbour_index, :, :, :]
        masked_kspace_2_motion = tf.cast(masked_kspace_2_motion, tf.complex64)
        masked_kspace_2 = tf.cast(masked_kspace_2, tf.complex64)

        mask_1_motion = mask_1[:, neighbour_index, :, :, :]
        mask_2_motion = mask_2[:, neighbour_index, :, :, :]

        input_smaps = tf.cast(batch_smaps, tf.complex64)
        smaps_motion = tf.expand_dims(batch_smaps, axis=1)

        # motion field, (t, t, x, y, 2)
        total_frame = 2 * self.neighbour_frame + 1
        u = np.load(f'neighbour_{total_frame}_flow_%s_%d.txt.npy' % (fname, slidx[0]), allow_pickle=True)
        u = np.expand_dims(u, axis=0)  # (1, 25, 5, 176, 132, 2)

        label = tf.zeros(shape=masked_img_1.shape, dtype=tf.complex64)
        print('inputs shape:', masked_img_1.shape, masked_img_2.shape, masked_kspace_1.shape, masked_kspace_2.shape, mask_1.shape, mask_2.shape, input_smaps.shape, u.shape)

        return [masked_img_1, masked_kspace_1_motion, mask_1_motion, smaps_motion, u,
                masked_img_2, masked_kspace_2_motion, mask_2_motion,
                masked_kspace_1, masked_kspace_2, mask_1, mask_2, input_smaps], label

