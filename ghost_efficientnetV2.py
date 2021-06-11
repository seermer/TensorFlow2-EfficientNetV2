from tensorflow.keras import Model, layers, activations
import tensorflow_addons as tfa
import math

"""
round_filters and round_repeats are borrowed from official repo
https://github.com/google/automl/tree/master/efficientnetv2
"""


def round_filters(filters, multiplier=1.):
    divisor = 8
    min_depth = 8
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    return int(new_filters)


def round_repeats(repeats, multiplier=1.):
    return int(math.ceil(multiplier * repeats))


def squeeze_and_excite(x, in_channels, out_channels, activation, reduction_ratio=4):
    x = layers.GlobalAvgPool2D()(x)
    x = layers.Dense(in_channels // reduction_ratio)(x)
    x = layers.Activation(activation)(x)
    x = layers.Dense(out_channels)(x)
    x = layers.Activation(activations.sigmoid)(x)
    return x

def ghost_conv(x, out_channels, kernel_size, stride, kernel_regularizer=None):
    x1 = layers.Conv2D(out_channels // 2, kernel_size=kernel_size, strides=stride, padding="same",
                       use_bias=False, kernel_regularizer=kernel_regularizer)(x)
    x2 = layers.BatchNormalization(epsilon=1e-5)(x1)
    x2 = layers.Activation(activations.elu)(x2)
    x2 = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding="same",
                                use_bias=False, kernel_regularizer=kernel_regularizer)(x2)
    return layers.Concatenate()([x1, x2])

def fused_mbconv(x, in_channels, out_channels, kernel_size, activation, stride=1, reduction_ratio=4,
                 expansion=6, dropout=None, drop_connect=.2):
    shortcut = x
    expanded = round_filters(in_channels * expansion)

    if stride == 2:
        shortcut = layers.AveragePooling2D()(shortcut)
    if in_channels != out_channels:
        shortcut = ghost_conv(shortcut, out_channels, (1, 1), 1)

    if expansion != 1:
        x = ghost_conv(x, expanded, kernel_size, stride)
        x = layers.BatchNormalization(epsilon=1e-5)(x)
        x = layers.Activation(activation)(x)

        if (dropout is not None) and (dropout != 0.):
            x = layers.Dropout(dropout)(x)

    if reduction_ratio is not None:
        se = squeeze_and_excite(x, in_channels, expanded, activation, reduction_ratio)
        x = layers.Multiply()([x, se])

    x = ghost_conv(x, out_channels, (1, 1) if expansion != 1 else kernel_size, 1)
    x = layers.BatchNormalization(epsilon=1e-5)(x)

    x = tfa.layers.StochasticDepth()([shortcut, x])
    return x


def mbconv(x, in_channels, out_channels, kernel_size, activation, stride=1,
           reduction_ratio=4, expansion=6, dropout=None, drop_connect=.2):
    shortcut = x
    expanded = round_filters(in_channels * expansion)

    if stride == 2:
        shortcut = layers.AveragePooling2D()(shortcut)
    if in_channels != out_channels:
        shortcut = ghost_conv(shortcut, out_channels, (1, 1), 1)

    if expansion != 1:
        x = ghost_conv(x, expanded, (1, 1), 1)
        x = layers.BatchNormalization(epsilon=1e-5)(x)
        x = layers.Activation(activation)(x)

    x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(epsilon=1e-5)(x)
    x = layers.Activation(activation)(x)

    if (expansion != 1) and (dropout is not None) and (dropout != 0.):
        x = layers.Dropout(dropout)(x)

    if reduction_ratio is not None:
        se = squeeze_and_excite(x, in_channels, expanded, activation, reduction_ratio)
        x = layers.Multiply()([x, se])

    x = ghost_conv(x, out_channels, (1, 1), 1)
    x = layers.BatchNormalization(epsilon=1e-5)(x)
    x = tfa.layers.StochasticDepth()([shortcut, x])
    return x


def repeat(x, count, in_channels, out_channels, kernel_size, activation,
           stride=1, reduction_ratio=None, expansion=6, fused=False, dropout=None, drop_connect=.2):
    for i in range(count):
        if fused:
            x = fused_mbconv(x, in_channels, out_channels, kernel_size,
                             activation, stride, reduction_ratio, expansion, dropout, drop_connect)
        else:
            x = mbconv(x, in_channels, out_channels, kernel_size, activation, stride,
                       reduction_ratio, expansion, dropout, drop_connect)
    return x


def stage(x, count, in_channels, out_channels, kernel_size, activation,
          stride=1, reduction_ratio=None, expansion=6, fused=False, dropout=None, drop_connect=.2):
    x = repeat(x, count=1, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
               activation=activation, stride=stride, reduction_ratio=reduction_ratio,
               expansion=expansion, fused=fused, dropout=dropout, drop_connect=drop_connect)
    x = repeat(x, count=count - 1, in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
               activation=activation, stride=1, reduction_ratio=reduction_ratio,
               expansion=expansion, fused=fused, dropout=dropout, drop_connect=drop_connect)
    return x


def base(cfg, num_classes=1000, input_tensor=None, activation=activations.swish,
         width_mult=1., depth_mult=1., conv_dropout_rate=None, dropout_rate=None, drop_connect=.2):
    """
    EfficientNet-V2-s, re-implementation according to
    https://arxiv.org/abs/2104.00298
    and official code
        https://github.com/google/automl/tree/master/efficientnetv2
    EfficientNetV2: Smaller Models and Faster Training
    by Mingxing Tan, Quoc V. Le

    :param cfg: configuration of stages
    :param num_classes: number of classes to output
    :param input_tensor: given a tensor as input, if provided, in_shape will be ignored
    :param activation: activation to use across hidden layers
    :param width_mult: width factor, default to 1.0
    :param depth_mult: depth multiplier, default to 1.0
    :param conv_dropout_rate: probability to drop after each MBConv/stage, 0 or None means no dropout will be applied
    :param dropout_rate: probability to drop after GlobalAveragePooling, 0 or None means no dropout will be applied
    :param drop_connect: probability to drop spatially in skip connections, 0 or None means no dropout will be applied
    :return: a tf.keras model
    """
    inp = input_tensor
    # stage 0
    x = layers.Conv2D(cfg[0][4], kernel_size=(3, 3), strides=2, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization(epsilon=1e-5)(x)
    x = layers.Activation(activation)(x)

    for stage_cfg in cfg:
        x = stage(x, count=round_repeats(stage_cfg[0], depth_mult),
                  in_channels=round_filters(stage_cfg[4], width_mult),
                  out_channels=round_filters(stage_cfg[5], width_mult),
                  kernel_size=stage_cfg[1], activation=activation, stride=stage_cfg[2],
                  reduction_ratio=stage_cfg[7], expansion=stage_cfg[3], fused=stage_cfg[6] == 1,
                  dropout=conv_dropout_rate, drop_connect=drop_connect)

    # final stage
    x = layers.Conv2D(round_filters(1280, width_mult), (1, 1), strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(epsilon=1e-5)(x)
    x = layers.Activation(activation)(x)

    x = layers.GlobalAvgPool2D()(x)
    if (dropout_rate is not None) and (dropout_rate != 0):
        x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(num_classes)(x)
    x = layers.Activation(activations.softmax)(x)

    return Model(inp, x)


def s(in_shape=(224, 224, 3), num_classes=1000, input_tensor=None, activation=activations.swish,
      width_mult=1., depth_mult=1., conv_dropout_rate=None, dropout_rate=None, drop_connect=.2):
    """
        EfficientNet-V2-s, re-implementation according to
        https://arxiv.org/abs/2104.00298
        and official code
        https://github.com/google/automl/tree/master/efficientnetv2
        EfficientNetV2: Smaller Models and Faster Training
        by Mingxing Tan, Quoc V. Le

        :param in_shape: input shape of the model, in form of (H, W, C)
        :param num_classes: number of classes to output
        :param input_tensor: given a tensor as input, if provided, in_shape will be ignored
        :param activation: activation to use across hidden layers
        :param width_mult: width factor, default to 1.0
        :param depth_mult: depth multiplier, default to 1.0
        :param conv_dropout_rate: probability to drop after each MBConv/stage, 0 or None means no dropout will be applied
        :param dropout_rate: probability to drop after GlobalAveragePooling, 0 or None means no dropout will be applied
        :param drop_connect: probability to drop spatially in skip connections, 0 or None means no dropout will be applied
        :return: a tf.keras model
    """

    # each row is a stage
    # count, kernel size, stride, expansion ratio, in channel, out channel, is fused(1 if true), reduction ratio(None if no se)
    cfg = [
        [2, 3, 1, 1, 24, 24, 1, None],
        [4, 3, 2, 4, 24, 48, 1, None],
        [4, 3, 2, 4, 48, 64, 1, None],
        [6, 3, 2, 4, 64, 128, 0, 4],
        [9, 3, 1, 6, 128, 160, 0, 4],
        [15, 3, 2, 6, 160, 256, 0, 4],
    ]
    input_tensor = layers.Input(in_shape) if input_tensor is None else input_tensor
    return base(cfg=cfg, num_classes=num_classes, input_tensor=input_tensor, activation=activation,
                width_mult=width_mult, depth_mult=depth_mult, conv_dropout_rate=conv_dropout_rate,
                dropout_rate=dropout_rate, drop_connect=drop_connect)


def m(in_shape=(224, 224, 3), num_classes=1000, input_tensor=None, activation=activations.swish,
      width_mult=1.0, depth_mult=1., conv_dropout_rate=None, dropout_rate=None, drop_connect=.2):
    """
        EfficientNet-V2-m, re-implementation according to
        https://arxiv.org/abs/2104.00298
        and official code
        https://github.com/google/automl/tree/master/efficientnetv2
        EfficientNetV2: Smaller Models and Faster Training
        by Mingxing Tan, Quoc V. Le

        :param in_shape: input shape of the model, in form of (H, W, C)
        :param num_classes: number of classes to output
        :param input_tensor: given a tensor as input, if provided, in_shape will be ignored
        :param activation: activation to use across hidden layers
        :param width_mult: width factor, default to 1.0
        :param depth_mult: depth multiplier, default to 1.0
        :param conv_dropout_rate: probability to drop after each MBConv/stage, 0 or None means no dropout will be applied
        :param dropout_rate: probability to drop after GlobalAveragePooling, 0 or None means no dropout will be applied
        :param drop_connect: probability to drop spatially in skip connections, 0 or None means no dropout will be applied
        :return: a tf.keras model
    """

    # each row is a stage
    # count, kernel size, stride, expansion ratio, in channel, out channel, is fused(1 if true), reduction ratio(None if no se)
    cfg = [
        [3, 3, 1, 1, 24, 24, 1, None],
        [5, 3, 2, 4, 24, 48, 1, None],
        [5, 3, 2, 4, 48, 80, 1, None],
        [7, 3, 2, 4, 80, 160, 0, 4],
        [14, 3, 1, 6, 160, 176, 0, 4],
        [18, 3, 2, 6, 176, 304, 0, 4],
        [5, 3, 1, 6, 304, 512, 0, 4],
    ]
    input_tensor = layers.Input(in_shape) if input_tensor is None else input_tensor
    return base(cfg=cfg, num_classes=num_classes, input_tensor=input_tensor, activation=activation,
                width_mult=width_mult, depth_mult=depth_mult, conv_dropout_rate=conv_dropout_rate,
                dropout_rate=dropout_rate, drop_connect=drop_connect)


def l(in_shape=(224, 224, 3), num_classes=1000, input_tensor=None, activation=activations.swish,
      width_mult=1.0, depth_mult=1., conv_dropout_rate=None, dropout_rate=None, drop_connect=.2):
    """
        EfficientNet-V2-l, re-implementation according to
        https://arxiv.org/abs/2104.00298
        and official code
        https://github.com/google/automl/tree/master/efficientnetv2
        EfficientNetV2: Smaller Models and Faster Training
        by Mingxing Tan, Quoc V. Le

        :param in_shape: input shape of the model, in form of (H, W, C)
        :param num_classes: number of classes to output
        :param input_tensor: given a tensor as input, if provided, in_shape will be ignored
        :param activation: activation to use across hidden layers
        :param width_mult: width factor, default to 1.0
        :param depth_mult: depth multiplier, default to 1.0
        :param conv_dropout_rate: probability to drop after each MBConv/stage, 0 or None means no dropout will be applied
        :param dropout_rate: probability to drop after GlobalAveragePooling, 0 or None means no dropout will be applied
        :param drop_connect: probability to drop spatially in skip connections, 0 or None means no dropout will be applied
        :return: a tf.keras model
    """

    # each row is a stage
    # count, kernel size, stride, expansion ratio, in channel, out channel, is fused(1 if true), reduction ratio(None if no se)

    cfg = [
        [4, 3, 1, 1, 32, 32, 1, None],
        [7, 3, 2, 4, 32, 64, 1, None],
        [7, 3, 2, 4, 64, 96, 1, None],
        [10, 3, 2, 4, 96, 192, 0, 4],
        [19, 3, 1, 6, 192, 224, 0, 4],
        [25, 3, 2, 6, 224, 384, 0, 4],
        [7, 3, 1, 6, 384, 640, 0, 4],
    ]
    input_tensor = layers.Input(in_shape) if input_tensor is None else input_tensor
    return base(cfg=cfg, num_classes=num_classes, input_tensor=input_tensor, activation=activation,
                width_mult=width_mult, depth_mult=depth_mult, conv_dropout_rate=conv_dropout_rate,
                dropout_rate=dropout_rate, drop_connect=drop_connect)


def xl(in_shape=(224, 224, 3), num_classes=1000, input_tensor=None, activation=activations.swish,
      width_mult=1.0, depth_mult=1., conv_dropout_rate=None, dropout_rate=None, drop_connect=.2):
    """
            EfficientNet-V2-xl, re-implementation according to
            https://arxiv.org/abs/2104.00298
            and official code
            https://github.com/google/automl/tree/master/efficientnetv2
            EfficientNetV2: Smaller Models and Faster Training
            by Mingxing Tan, Quoc V. Le

            :param in_shape: input shape of the model, in form of (H, W, C)
            :param num_classes: number of classes to output
            :param input_tensor: given a tensor as input, if provided, in_shape will be ignored
            :param activation: activation to use across hidden layers
            :param width_mult: width factor, default to 1.0
            :param depth_mult: depth multiplier, default to 1.0
            :param conv_dropout_rate: probability to drop after each MBConv/stage, 0 or None means no dropout will be applied
            :param dropout_rate: probability to drop after GlobalAveragePooling, 0 or None means no dropout will be applied
            :param drop_connect: probability to drop spatially in skip connections, 0 or None means no dropout will be applied
            :return: a tf.keras model
        """

    cfg = [
        [4, 3, 1, 1, 32, 32, 1, None],
        [8, 3, 2, 4, 32, 64, 1, None],
        [8, 3, 2, 4, 64, 96, 1, None],
        [16, 3, 2, 4, 96, 192, 0, 4],
        [24, 3, 1, 6, 192, 256, 0, 4],
        [32, 3, 2, 6, 256, 512, 0, 4],
        [8, 3, 1, 6, 512, 640, 0, 4],
    ]
    input_tensor = layers.Input(in_shape) if input_tensor is None else input_tensor
    return base(cfg=cfg, num_classes=num_classes, input_tensor=input_tensor, activation=activation,
                width_mult=width_mult, depth_mult=depth_mult, conv_dropout_rate=conv_dropout_rate,
                dropout_rate=dropout_rate, drop_connect=drop_connect)


def main():
    model = s((224, 224, 3), 1000)
    model.summary()


if __name__ == '__main__':
    main()
