from tensorflow.keras import Model, layers, activations


def _depth(v, divisor=8, min_value=None):
    """
    copied from tensorflow MobileNet code
    # This function is taken from the original tf repo.
    # It ensures that all layers have a channel number that is divisible by 8
    # It can be seen here:
    # https://github.com/tensorflow/models/blob/master/research/
    # slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def squeeze_and_excite(x, in_channels, out_channels, activation, reduction_ratio=4):
    x = layers.GlobalAvgPool2D()(x)
    x = layers.Dense(_depth(in_channels // reduction_ratio))(x)
    x = layers.Activation(activation)(x)
    x = layers.Dense(out_channels)(x)
    x = layers.Activation(activations.sigmoid)(x)
    return x


def fused_mbconv(x, in_channels, out_channels, kernel_size, activation, stride=1, expansion=6):
    shortcut = x
    expanded = _depth(in_channels * expansion)

    x = layers.Conv2D(expanded, kernel_size, stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(epsilon=1e-5)(x)
    x = layers.Activation(activation)(x)

    x = layers.Conv2D(out_channels, (1, 1), 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(epsilon=1e-5)(x)
    if stride == 1 and in_channels == out_channels:
        x = layers.Add()([x, shortcut])
    return layers.Activation(activation)(x)


def mbconv(x, in_channels, out_channels, kernel_size, activation, stride=1, reduction_ratio=4, expansion=6):
    shortcut = x
    expanded = _depth(in_channels * expansion)

    x = layers.Conv2D(expanded, (1, 1), 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(epsilon=1e-5)(x)
    x = layers.Activation(activation)(x)

    x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(epsilon=1e-5)(x)
    x = layers.Activation(activation)(x)

    se = squeeze_and_excite(x, in_channels, expanded, activation, reduction_ratio)
    x = layers.Add()([x, se])

    x = layers.Conv2D(out_channels, (1, 1), 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(epsilon=1e-5)(x)
    if stride == 1 and in_channels == out_channels:
        x = layers.Add()([x, shortcut])
    return layers.Activation(activation)(x)


def repeat(x, count, in_channels, out_channels, kernel_size, activation,
           stride=1, reduction_ratio=4, expansion=6, fused=False, dropout=.1):
    for i in range(count):
        if fused:
            x = fused_mbconv(x, in_channels, out_channels, kernel_size, activation, stride, expansion)
        else:
            x = mbconv(x, in_channels, out_channels, kernel_size, activation, stride,
                       reduction_ratio, expansion)
        if (dropout is not None) and (dropout != 0):
            x = layers.SpatialDropout2D(dropout)(x)
    return x


def stage(x, count, in_channels, out_channels, kernel_size, activation,
          stride=1, reduction_ratio=4, expansion=6, fused=False, dropout=.1):
    x = repeat(x, count=1, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
               activation=activation, stride=stride, reduction_ratio=reduction_ratio,
               expansion=expansion, fused=fused, dropout=dropout)
    x = repeat(x, count=count - 1, in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
               activation=activation, stride=1, reduction_ratio=reduction_ratio,
               expansion=expansion, fused=fused, dropout=dropout)
    return x


def small(in_shape, num_classes, alpha=1.0, spatial_dropout_rate=0., dropout_rate=0.):
    """
    EfficientNet-V2-s, re-implementation according to
    https://arxiv.org/abs/2104.00298
    EfficientNetV2: Smaller Models and Faster Training
    by Mingxing Tan, Quoc V. Le

    :param in_shape: input shape in form of (H, W, C)
    :param num_classes: number of classes to output
    :param alpha: width factor, default to 1.0
    :param spatial_dropout_rate: probability to drop after each MBConv/stage, 0 or None means no dropout will be applied
    :param dropout_rate: probability to drop after GlobalAveragePooling, 0 or None means no dropout will be applied
    :return: a tf.keras model
    """
    inp = layers.Input(in_shape)
    x = layers.Conv2D(24, kernel_size=(3, 3), strides=2, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization(epsilon=1e-5)(x)
    x = layers.Activation(activations.swish)(x)

    # stage 1
    x = stage(x, count=2, in_channels=_depth(24 * alpha), out_channels=_depth(24 * alpha), kernel_size=(3, 3),
              activation=activations.swish, stride=1, expansion=1, fused=True,
              dropout=spatial_dropout_rate)

    # stage 2
    x = stage(x, count=4, in_channels=_depth(24 * alpha), out_channels=_depth(48 * alpha), kernel_size=(3, 3),
              activation=activations.swish, stride=2, expansion=4, fused=True,
              dropout=spatial_dropout_rate)

    # stage 3
    x = stage(x, count=4, in_channels=_depth(48 * alpha), out_channels=_depth(64 * alpha), kernel_size=(3, 3),
              activation=activations.swish, stride=2, expansion=4, fused=True,
              dropout=spatial_dropout_rate)

    # stage 4
    x = stage(x, count=6, in_channels=_depth(64 * alpha), out_channels=_depth(128 * alpha), kernel_size=(3, 3),
              activation=activations.swish, stride=2, reduction_ratio=4, expansion=4, fused=False,
              dropout=spatial_dropout_rate)

    # stage 5
    x = stage(x, count=9, in_channels=_depth(128 * alpha), out_channels=_depth(160 * alpha), kernel_size=(3, 3),
              activation=activations.swish, stride=1, reduction_ratio=4, expansion=6, fused=False,
              dropout=spatial_dropout_rate)

    # stage 6
    x = stage(x, count=15, in_channels=_depth(160 * alpha), out_channels=_depth(272 * alpha), kernel_size=(3, 3),
              activation=activations.swish, stride=2, reduction_ratio=4, expansion=6, fused=False,
              dropout=spatial_dropout_rate)

    # stage 7
    x = layers.Conv2D(_depth(1792 * alpha), (1, 1), strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(epsilon=1e-5)(x)
    x = layers.Activation(activations.swish)(x)

    x = layers.GlobalAvgPool2D()(x)
    if (dropout_rate is not None) and (dropout_rate != 0):
        x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(num_classes)(x)
    x = layers.Activation(activations.softmax)(x)

    return Model(inp, x)


def main():
    model = small((224, 224, 3), 1000)
    model.summary()


if __name__ == '__main__':
    main()
