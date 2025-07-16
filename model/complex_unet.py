
import merlintf
import numpy as np
import tensorflow as tf


class UNet_2Dt(tf.keras.Model):
    def __init__(self, dim='2Dt', filters=64, kernel_size_2d=(1, 3, 3), kernel_size_t=(3, 1, 1), pool_size=2,
                 num_layer_per_level=2, num_level=4,
                 activation='relu', activation_last='relu', kernel_size_last=1, use_bias=True,
                 normalization='none', downsampling='mp', upsampling='tc',
                 name='UNet_2Dt', padding='none', **kwargs):

        super().__init__(name=name)

        # validate input dimensions
        self.kernel_size_2d = merlintf.keras.utils.validate_input_dimension(dim, kernel_size_2d)
        self.kernel_size_t = merlintf.keras.utils.validate_input_dimension(dim, kernel_size_t)
        self.pool_size = merlintf.keras.utils.validate_input_dimension(dim, pool_size)

        self.dim = dim
        self.num_level = num_level
        self.num_layer_per_level = num_layer_per_level
        self.filters = filters
        self.use_bias = use_bias
        self.activation = activation
        self.activation_last = activation_last
        self.kernel_size_last = merlintf.keras.utils.validate_input_dimension(dim, kernel_size_last)
        self.normalization = normalization
        self.downsampling = downsampling
        self.upsampling = upsampling
        self.padding = padding
        self.norm_layer = merlintf.keras.layers.complex_norm.ComplexInstanceNormalization
        self.activation_layer = merlintf.keras.layers.complex_act.ModReLU
        if 'in_shape' in kwargs:
            self.use_padding = self.is_padding_needed(kwargs.get('in_shape'))
        else:
            self.use_padding = self.is_padding_needed()  # in_shape at build time not known

    def create_layers(self, **kwargs):
        # ------------- #
        # create layers #
        # ------------- #
        self.ops = []
        # encoder
        stage = []
        for ilevel in range(self.num_level):
            level = []
            for ilayer in range(self.num_layer_per_level):
                level.append(self.conv_layer(self.filters * (2 ** ilevel), self.kernel_size_2d,
                                             strides=[1, 1, 1], use_bias=self.use_bias,
                                             activation=None, padding='same', **kwargs))
                level.append(self.norm_layer())  # todo:'ModReLU'
                level.append(self.activation_layer())

                level.append(self.conv_layer(self.filters * (2 ** ilevel), self.kernel_size_t,
                                             strides=[1, 1, 1], use_bias=self.use_bias,
                                             activation=None, padding='same', **kwargs))
                level.append(self.norm_layer())  # todo: 'ModReLU'
                level.append(self.activation_layer())

            if self.downsampling == 'mp':
                level.append(callCheck(self.down_layer, pool_size=self.pool_size, **kwargs))
            else:
                # level.append(callCheck(self.down_layer, **kwargs))
                level.append(merlintf.keras.layers.ComplexConv3D(self.filters * (2 ** ilevel), (1, 1, 1),
                                                                 strides=[2, 2, 2],
                                                                 use_bias=None,
                                                                 activation=None,
                                                                 padding='same', **kwargs))
            stage.append(level)
        self.ops.append(stage)

        # bottleneck
        stage = []
        for ilayer in range(self.num_layer_per_level):
            stage.append(self.conv_layer(self.filters * (2 ** (self.num_level)), self.kernel_size_2d,
                                         strides=[1, 1, 1], use_bias=self.use_bias, activation=None,
                                         padding='same', **kwargs))
            stage.append(self.norm_layer())  # todo: 'ModReLU'
            stage.append(self.activation_layer())

            stage.append(self.conv_layer(self.filters * (2 ** (self.num_level)), self.kernel_size_t,
                                         strides=[1, 1, 1], use_bias=self.use_bias, activation=None,
                                         padding='same', **kwargs))
            stage.append(self.norm_layer())  # todo: 'ModReLU'
            stage.append(self.activation_layer())

        if self.upsampling == 'us':
            stage.append(self.up_layer(self.pool_size, **kwargs))
        elif self.upsampling == 'tc':
            stage.append(self.up_layer(self.filters * (2 ** (self.num_level - 1)), (1, 1, 1),
                                       strides=self.pool_size, use_bias=self.use_bias, activation=self.activation,
                                       padding='same', **kwargs))

        self.ops.append(stage)

        # decoder
        stage = []
        for ilevel in range(self.num_level - 1, -1, -1):
            level = []
            for ilayer in range(self.num_layer_per_level):
                level.append(self.conv_layer(self.filters * (2 ** ilevel), self.kernel_size_t,
                                             strides=1, use_bias=self.use_bias, activation=None,
                                             padding='same', **kwargs))
                level.append(self.norm_layer())  # todo: 'ModReLU'
                level.append(self.activation_layer())

                level.append(self.conv_layer(self.filters * (2 ** ilevel), self.kernel_size_2d,
                                             strides=1, use_bias=self.use_bias, activation=None,
                                             padding='same', **kwargs))
                level.append(self.norm_layer())  # todo: 'ModReLU'
                level.append(self.activation_layer())

            if ilevel > 0:
                if self.upsampling == 'us':
                    level.append(self.up_layer(self.pool_size, **kwargs))
                elif self.upsampling == 'tc':
                    level.append(self.up_layer(self.filters * (2 ** (ilevel - 1)), (1, 1, 1),
                                               strides=self.pool_size, use_bias=self.use_bias,
                                               activation=self.activation,
                                               padding='same', **kwargs))
            stage.append(level)
        self.ops.append(stage)

        # output convolution
        self.ops.append(self.conv_layer(self.out_cha, self.kernel_size_last, strides=1,
                                        use_bias=self.use_bias,
                                        activation=self.activation_last,
                                        padding='same', **kwargs))

    def is_padding_needed(self, in_shape=None):
        # in_shape (excluding batch and channel dimension!)
        if not self.padding.lower() == 'none' and in_shape is None:
            print(
                'merlintf.keras.models.unet: Check if input padding/output cropping is needed. No input shape specified, potentially switching to eager mode execution. Please provide input_shape by calling: model.is_padding_needed(input_shape)')
        if in_shape is None:  # input shape not specified or dynamically varying
            self.use_padding = True
            self.pad = None
            self.optotf_pad = None
        else:  # input shape specified
            self.pad, self.optotf_pad = self.calculate_padding(in_shape)
            if np.all(np.asarray(self.pad) == 0):
                self.use_padding = False
            else:
                self.use_padding = True
        if self.padding.lower() == 'force_none':
            self.use_padding = False
            self.pad = None
            self.optotf_pad = None
        if self.use_padding:
            if self.padding.lower() == 'none':
                self.padding = 'zero'  # default padding
            print('Safety measure: Enabling input padding and output cropping!')
            print('!!! Compile model with model.compile(run_eagerly=True) !!!')
        return self.use_padding

    def calculate_padding(self, in_shape):
        in_shape = np.asarray(in_shape)
        n_dim = merlintf.keras.utils.get_ndim(self.dim)
        if len(in_shape) > n_dim:
            in_shape = in_shape[:n_dim]
        factor = np.power(self.pool_size, self.num_level)
        paddings = np.ceil(in_shape / factor) * factor - in_shape
        pad = []
        optotf_pad = []
        for idx in range(n_dim):
            pad_top = paddings[idx].astype(np.int) // 2
            pad_bottom = paddings[idx].astype(np.int) - pad_top
            optotf_pad.extend([pad_top, pad_bottom])
            pad.append((pad_top, pad_bottom))
        return tuple(pad), optotf_pad[::-1]

    def calculate_padding_tensor(self, tensor):
        # calculate pad size
        # ATTENTION: input shape calculation with tf.keras.fit() ONLY possible in eager mode because of NoneType defined shapes! -> Force eager mode execution
        imshape = tensor.get_shape().as_list()
        if tf.keras.backend.image_data_format() == 'channels_last':  # default
            imshapenp = np.array(imshape[1:len(self.pool_size) + 1]).astype(float)
        else:  # channels_first
            imshapenp = np.array(imshape[2:len(self.pool_size) + 2]).astype(float)

        return self.calculate_padding(imshapenp)

    def call(self, inputs):
        if self.use_padding:
            if self.pad is None:  # input shape cannot be determined or fixed before compile
                pad, optotf_pad = self.calculate_padding_tensor(inputs)
            else:
                pad = self.pad  # local variable to avoid permanent storage of fixed pad value in case of dynamic input shapes
                optotf_pad = self.optotf_pad
            if self.padding.lower() == 'zero':
                x = self.pad_layer(pad)(inputs)
            else:
                x = self.pad_layer(inputs, optotf_pad, self.padding)
            # x = merlintf.keras.layers.pad(len(self.pool_size), inputs, optotf_pad, 'symmetric')  # symmetric padding via optox
        else:
            x = inputs
        xforward = []
        # encoder
        for ilevel in range(self.num_level):
            for iop, op in enumerate(self.ops[0][ilevel]):
                if iop == len(self.ops[0][ilevel]) - 1:
                    xforward.append(x)
                if op is not None:
                    x = op(x)

        # bottleneck
        for op in self.ops[1]:
            if op is not None:
                x = op(x)

        # decoder
        for ilevel in range(self.num_level - 1, -1, -1):
            x = tf.keras.layers.concatenate([x, xforward[ilevel]])
            for op in self.ops[2][self.num_level - 1 - ilevel]:
                if op is not None:
                    x = op(x)

        # output convolution
        x = self.ops[3](x)
        if self.use_padding:
            x = self.crop_layer(pad)(x)
        return x


class ComplexUNet_2Dt(UNet_2Dt):
    def __init__(self, dim='2Dt', filters=64, kernel_size_2d=(1, 3, 3), kernel_size_t=(3, 1, 1), pool_size=2,
                 num_layer_per_level=2, num_level=4,
                 activation='ModReLU', activation_last='ModReLU', kernel_size_last=1, use_bias=True,
                 normalization='none', downsampling='st', upsampling='tc',
                 name='ComplexUNet_2Dt', padding='none', **kwargs):
        """
        Builds the complex-valued 2D/2D+t/3D/3D+t/4D UNet model
        """
        super().__init__(dim, filters, kernel_size_2d, kernel_size_t, pool_size, num_layer_per_level, num_level,
                         activation, activation_last, kernel_size_last, use_bias, normalization, downsampling,
                         upsampling, name, padding, **kwargs)

        self.dim = '3D'
        # get correct conv operator
        self.conv_layer = merlintf.keras.layers.ComplexConvolution('3D')
        if self.padding.lower() == 'zero':
            self.pad_layer = merlintf.keras.layers.ZeroPadding('3D')
        else:
            if self.dim == '2D':
                self.pad_layer = merlintf.keras.layers.Pad2D
            elif self.dim == '3D':
                self.pad_layer = merlintf.keras.layers.Pad3D
            else:
                raise RuntimeError(f"Padding for {dim} and {self.padding} not implemented!")

        self.crop_layer = merlintf.keras.layers.Cropping('3D')

        # output convolution
        # self.activation_layer_last = merlintf.keras.layers.Activation(self.activation_last)
        self.out_cha = 1

        # get downsampling operator
        n_dim = merlintf.keras.utils.get_ndim(self.dim)
        if downsampling == 'mp':
            self.down_layer = merlintf.keras.layers.MagnitudeMaxPooling('3D')
            self.strides = [1] * num_layer_per_level
        elif downsampling == 'st':
            # self.down_layer = None
            self.down_layer = merlintf.keras.layers.ComplexConvolution('3D')
            # self.strides = [[1] * n_dim] * (num_layer_per_level - 1)
            # self.strides.append(list(self.pool_size))
            self.strides = [2, 2, 2]
        else:
            raise RuntimeError(f"Downsampling operation {downsampling} not implemented!")

        # get upsampling operator
        if upsampling == 'us':
            self.up_layer = merlintf.keras.layers.UpSampling(dim)  # TODO check if working for complex
        elif upsampling == 'tc':
            self.up_layer = merlintf.keras.layers.ComplexConvolutionTranspose('3D')
        else:
            raise RuntimeError(f"Upsampling operation {upsampling} not implemented!")

        super().create_layers(**kwargs)


def callCheck(fhandle, **kwargs):
    if fhandle is not None:
        return fhandle(**kwargs)
    else:
        return fhandle

