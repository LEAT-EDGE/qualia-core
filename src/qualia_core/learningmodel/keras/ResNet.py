
class ResBlock: # Not an actual Keras model because it is harder to serialize and get it to work on third-party tools
    expansion = 1

    def __init__(self,
                layers_t,
                in_filters,
                filters,
                kernel_size,
                stride,
                padding,
                batch_norm: bool=False,
                bn_momentum: float=0.99,
                force_projection_with_stride: bool=True):
        super().__init__()
        from tensorflow.keras.layers import BatchNormalization, Activation, Add
        from tensorflow.python.keras.backend import unique_object_name

        self.in_filters = in_filters
        self.filters = filters
        self.stride = stride
        self.batch_norm = batch_norm
        self.force_projection_with_stride = force_projection_with_stride
        
        self.padding_1 = layers_t.ZeroPadding(padding=padding)
        self.conv_1 = layers_t.Conv(filters=filters, kernel_size=kernel_size, strides=1, use_bias=not batch_norm)
        if batch_norm:
            self.batchnorm_1 = BatchNormalization(momentum=bn_momentum)
        if self.stride != 1:
            self.maxpool_1 = layers_t.MaxPooling(stride)
        self.relu_1 = Activation('relu')

        self.padding_2 = layers_t.ZeroPadding(padding=padding)
        self.conv_2 = layers_t.Conv(name=unique_object_name('conv_ref', zero_based=True), filters=filters, kernel_size=kernel_size, strides=1, use_bias=not batch_norm)
        if batch_norm:
            self.batchnorm_2 = BatchNormalization(momentum=bn_momentum)

        if self.stride != 1:
            self.maxpool_shortcut = layers_t.MaxPooling(stride)

        if self.in_filters != self.expansion*self.filters or force_projection_with_stride and self.stride != 1:
            self.conv_shortcut = layers_t.Conv(name=unique_object_name('conv_shortcut', zero_based=True), filters=self.expansion * filters, kernel_size=1, strides=1, use_bias=not batch_norm)
            if batch_norm:
                self.batchnorm_shortcut = BatchNormalization(momentum=bn_momentum)

        self.add = Add()
        self.relu_2 = Activation('relu')


    def __call__(self, inputs):
        x = self.padding_1(inputs)
        x = self.conv_1(x)
        if self.batch_norm:
            x = self.batchnorm_1(x)
        if self.stride != 1:
            x = self.maxpool_1(x)
        x = self.relu_1(x)
        
        x = self.padding_2(x)
        x = self.conv_2(x)
        if self.batch_norm:
            x = self.batchnorm_2(x)

        # shortcut
        tmp = inputs
        if self.in_filters != self.expansion*self.filters or self.force_projection_with_stride and self.stride != 1:
            tmp = self.conv_shortcut(tmp)
            if self.batch_norm:
                tmp = self.batchnorm_shortcut(tmp)
        if self.stride != 1:
            tmp = self.maxpool_shortcut(tmp)

        x = self.add([x, tmp])
        x = self.relu_2(x)

        return x

    @property
    def output_shape(self):
        return self.relu_2.output_shape

def ResNet(input_shape,
        output_shape,
        filters: int=(15, 15, 30, 60, 120), # + (120,)
        kernel_sizes: tuple=(3, 3, 3, 3, 3),

        num_blocks: tuple=(2, 2, 2, 2), # + (2,)
        strides: tuple=(1, 1, 2, 2, 2), # + (2,)
        paddings: int=(1, 1, 1, 1, 1),

        prepool: int=1,
        batch_norm: bool=False,
        bn_momentum: float=0.99,
        force_projection_with_stride: bool=True,

        dims: int=1):

    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Flatten, Dense, BatchNormalization, Activation
    import tensorflow.keras.backend as backend

    if dims == 1:
        import qualia_core.learningmodel.keras.layers1d as layers_t
    elif dims == 2:
        import qualia_core.learningmodel.keras.layers2d as layers_t
    else:
        raise ValueError('Only dims=1 or dims=2 supported')

    # Do not track moving statistics (avg/var) for inference, use batch statistics only like PyTorch's track_running_stats
    track_running_stats ={} # {'training': True}

    # Actual model definition
    # Conv → BatchNorm → ResBlock → ResBlock → ResBlock → AveragePooling → Flatten → FullyConnected(out)

    inputs = Input(shape=input_shape)

    if prepool > 1:
        x = layers_t.AveragePooling(pool_size=prepool)(inputs)
    else:
        x = inputs

    # First non-residual convolution
    x = layers_t.ZeroPadding(padding=paddings[0])(x)
    x = layers_t.Conv(filters=filters[0], kernel_size=kernel_sizes[0], strides=strides[0], use_bias=not batch_norm)(x)
    if batch_norm:
        x = BatchNormalization(momentum=bn_momentum)(x)
    x = Activation('relu')(x)

    # Residual layers
    in_filters = filters[0]
    for f, kernel_size, layer_stride, padding, num_blocks in zip(filters[1:], kernel_sizes[1:], strides[1:], paddings[1:], num_blocks):
        for block_stride in [layer_stride] + [1] * (num_blocks - 1):
            block = ResBlock(layers_t=layers_t,
                        in_filters=in_filters,
                        filters=f,
                        kernel_size=kernel_size,
                        stride=block_stride,
                        padding=padding,
                        batch_norm=batch_norm,
                        bn_momentum=bn_momentum) # 1st block of layer
            x = block(x)
            in_filters = f * block.expansion

    # Reduce last conv layer output feature maps
    x = layers_t.MaxPooling(x.shape[1])(x)
    x = Flatten()(x)
    x = Dense(output_shape[0], use_bias=True)(x)
    outputs = Activation('softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name='ResNet')

    return model
