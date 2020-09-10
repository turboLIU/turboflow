import tensorflow as tf


def calc_param(x):
    b, h, w, c = x.get_shape().as_list()
    if b is None:
        return h*w*c
    else:
        return b*h*w*c


def _variable_on_cpu(shape,
                     weight_decay=0.995,
                     l2_loss = True,
                     debug=False,
                     name="conv"):
    with tf.device("/cpu:0"):
        w = tf.compat.v1.get_variable(shape=shape, initializer=tf.random_normal_initializer(0.0, 0.01),
                                      dtype=tf.float32, name=name)
        if debug:
            tf.summary.histogram(name, w)
        # tf.add_to_collection(name="weights_l2_loss", value=self.calc_l2_loss(w, 1-weight_decay))
        if l2_loss:
            tf.compat.v1.add_to_collection(name="weights_l2_loss", value=(1 - weight_decay) * tf.nn.l2_loss(w))
    return w


def init_params(*args, **kwargs):
    kernel_size = kwargs["kernel_size"]
    in_channels = kwargs["in_channels"]
    out_channels = kwargs["out_channels"]
    use_bias = kwargs["use_bias"]
    name = kwargs["name"]
    # weight_decay = kwargs["weight_decay"]
    w_shape = [kernel_size, kernel_size, in_channels, out_channels]
    w = _variable_on_cpu(w_shape, name="%s_w"%name)
    b = 0
    if use_bias:
        b_shape = [out_channels]
        b = _variable_on_cpu(b_shape, l2_loss=False, name="%s_b"%name)
        # return w, b
    return w, b


def calc_loss(*args, **kwargs):
    loss_type = kwargs["loss_type"]
    x = kwargs["x"]
    y = kwargs["y"]
    if loss_type == "L1":
        return tf.reduce_sum(tf.abs(x - y), name="L1_loss")
    elif loss_type == "L2":
        return tf.nn.l2_loss((x - y), name="L2_loss")

def hard_sigmoid(x):
    return tf.nn.relu6((x + 3) / 6)


def hard_swish( x):
    return x * hard_sigmoid(x)

def activation(*args, **kwargs):
    act_type = kwargs["act_type"]
    act_type = act_type.lower()
    if act_type == "relu":
        return tf.nn.relu(args[0])
    elif act_type == "relu6":
        return tf.nn.relu6(args[0])
    elif act_type == "lrelu":
        slope = kwargs["slope"]
        return tf.nn.leaky_relu(args[0], alpha=slope)
    elif act_type == "prelu":
        slope = tf.compat.v1.get_variable(shape=[1], initializer=tf.constant_initializer(value=0.1),trainable=True, dtype=tf.float32)
        y = slope * args[0]
        return tf.maximum(args[0], y, name="prelu")
    elif act_type == "tanh":
        return tf.nn.tanh(args[0])
    elif act_type == "selu":
        '''
        权重必须服从（0，1）的正态分布  W ~~ N(0,1)
        '''
        with tf.name_scope('elu') as scope:
            alpha = 1.6732632423543772848170429916717
            scale = 1.0507009873554804934193349852946
            return scale * tf.where(args[0] >= 0.0, args[0], alpha * tf.nn.elu(args[0]))
    elif act_type == "hswish":
        return hard_swish(args[0])
    elif act_type == "hsigmoid":
        return hard_sigmoid(args[0])
    elif act_type == "sigmoid":
        return tf.nn.sigmoid(args[0])
    elif act_type == "linear":
        return args[0]
    else:
        return NotImplementedError


def calc_l2_loss(weight, weight_decay):
    _, _, _, outchannel = weight.get_shape().as_list()
    return (weight_decay) * tf.reduce_sum(tf.square(weight)) / outchannel


def calc_l1_loss(weight, weight_decay):
    _, _, _, outchannel = weight.get_shape().as_list()
    return (weight_decay) * tf.reduce_sum(tf.abs(weight)) / outchannel


def batch_norm(x, training, name):
    x = tf.compat.v1.layers.batch_normalization(x, training=training, name=name)
    return x


def batch_norm_(*args, **kwargs):
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    b, h, w, c = args[0].get_shape().as_list()
    name = kwargs["name"]
    decay = 0.99
    epsilon = 1e-3
    gama = tf.compat.v1.get_variable(shape=c, initializer=tf.ones_initializer, name="%s_gama" % name)
    beta = tf.compat.v1.get_variable(shape=c, initializer=tf.zeros_initializer, name="%s_beta" % name)
    pop_mean = tf.compat.v1.get_variable(shape=c, initializer=tf.zeros_initializer, trainable=True,
                                         name="%s_pop_mean" % name)
    pop_variance = tf.compat.v1.get_variable(shape=c, initializer=tf.ones_initializer, trainable=True,
                                             name="%s_pop_variance" % name)

    if kwargs["training"] == True:
        average_mean, average_varance = tf.compat.v1.nn.moments(args[0], axes=[0, 1, 2], keep_dims=False, name=name)

        train_mean = tf.compat.v1.assign(pop_mean, pop_mean * decay + average_mean * (1 - decay))
        train_variance = tf.compat.v1.assign(pop_variance, pop_variance * decay + average_varance * (1 - decay))
        # tf.train.ExponentialMovingAverage
        tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, [train_mean, train_variance])
        # with tf.control_dependencies([train_mean, train_variance]):
        return tf.nn.batch_normalization(args[0], average_mean, average_varance, beta, gama, epsilon)
    else:
        # print("test phase~")
        return tf.nn.batch_normalization(args[0], pop_mean, pop_variance, beta, gama, epsilon)


def instance_norm(*args, **kwargs):  # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    b, h, w, c = args[0].get_shape().as_list()
    name = kwargs["name"]
    decay = 0.99
    epsilon = 1e-3
    shape = [b, 1, 1, c]
    gama = tf.compat.v1.get_variable(shape=shape, initializer=tf.ones_initializer, name="%s_gama" % name)
    beta = tf.compat.v1.get_variable(shape=shape, initializer=tf.zeros_initializer, name="%s_beta" % name)
    pop_mean = tf.compat.v1.get_variable(shape=shape, initializer=tf.zeros_initializer, trainable=True, name="%s_pop_mean" % name)
    pop_variance = tf.compat.v1.get_variable(shape=shape, initializer=tf.ones_initializer, trainable=True,
                                   name="%s_pop_variance" % name)

    if kwargs["training"] == True:
        average_mean, average_varance = tf.nn.moments(args[0], axes=[1, 2], keep_dims=True, name=name)

        train_mean = tf.compat.v1.assign(pop_mean, pop_mean * decay + average_mean * (1 - decay))
        train_variance = tf.compat.v1.assign(pop_variance, pop_variance * decay + average_varance * (1 - decay))
        # tf.train.ExponentialMovingAverage
        tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, [train_mean, train_variance])
        # with tf.control_dependencies([train_mean, train_variance]):
        return tf.nn.batch_normalization(args[0], average_mean, average_varance, beta, gama, epsilon)
    else:
        return tf.nn.batch_normalization(args[0], pop_mean, pop_variance, beta, gama, epsilon)

    # return layers.instance_norm(args[0], kwargs["name"], trainable=kwargs["training"])


def global_average_pooling(x, name="GAP"):
    return tf.compat.v1.reduce_mean(x, axis=[1, 2], keep_dims=True, name="Global_Average_Pooling_%s" % name)


def DropOut(x, droprate, training):
    '''
    :param x:
    :param droprate: 被丢弃的概率
    :param training:
    :return:
    '''
    if training:
        x = tf.nn.dropout(x, keep_prob=droprate, name="dropout")
    else:
        x = tf.nn.dropout(x, keep_prob=1, name="dropout")
    return x


def ResBlock(x, outdim, name="ResBlock"):
    _, h, w, c = x.get_shape().as_list()
    ix = ConvBlock(x, in_channels=c, out_channels=outdim, kernel_size=3, stride=1,
                        BN=True, use_bias=False, padding=True, act_type="relu", name="%s_1" % name)
    ix = ConvBlock(ix, in_channels=outdim, out_channels=outdim, kernel_size=3, stride=1,
                        BN=True, use_bias=False, padding=True, act_type="relu", name="%s_2" % name)
    if c == outdim:
        ix = ix + x
    else:
        ix = ConvBlock(x, in_channels=c, out_channels=outdim, kernel_size=3, stride=1,
                            BN=True, use_bias=False, padding=True, act_type="relu", name="%s_uniform" % name)
        ix = ix + x
    return tf.nn.relu(ix)


def SqzResBlock( x, outdim, name="ExpResBlock"):
    _, h, w, c = x.get_shape().as_list()
    cs = max(c // 4, 16)
    ix = ConvBlock(x, in_channels=c, out_channels=cs, kernel_size=1, stride=1,
                        BN=True, use_bias=False, padding=False, act_type="relu", name="%s_1" % name)
    ix = ConvBlock(ix, in_channels=cs, out_channels=cs, kernel_size=3, stride=1,
                        BN=True, use_bias=False, padding=True, act_type="relu", name="%s_2" % name)
    ix = ConvBlock(ix, in_channels=cs, out_channels=outdim, kernel_size=1, stride=1,
                        BN=True, use_bias=False, padding=False, act_type="relu", name="%s_3" % name)
    if c == outdim:
        ix = ix + x
    else:
        x = ConvBlock(x, in_channels=c, out_channels=outdim, kernel_size=3, stride=1,
                           BN=True, use_bias=False, padding=True, act_type="relu", name="%s_uniform" % name)
        ix = ix + x
    return tf.nn.relu(ix)


def ConvBlock( x, in_channels, out_channels, kernel_size, stride=1, name="ConvBlock",
              BN=True, use_bias=True, padding=True, act_type="relu", BN_type="BN", istrain=True):
    if padding:
        pdsz = kernel_size // 2
        x = tf.pad(x, [[0, 0], [pdsz, pdsz], [pdsz, pdsz], [0, 0]], name="pad")
    # x = tf.cast(x, tf.float16)
    weight, bias = init_params(kernel_size=kernel_size, in_channels=in_channels,
                                    out_channels=out_channels, use_bias=use_bias, name=name)

    x = tf.compat.v1.nn.conv2d(x, filter=weight, strides=[1, stride, stride, 1], padding="VALID")
    if BN:
        if BN_type == "BN":
            x = batch_norm(x, training=istrain, name="%s_BN" % name)
        elif BN_type == "IN":
            x = instance_norm(x, name="%s_IN" % name, training=istrain)
        else:
            raise NotImplementedError('[ERROR] BN type [%s] is not implemented!' % BN_type)
    if use_bias:
        # 注意： BN前加不加bias对结果都没有任何影响，由BN公式中可以推导出来
        # 这里bias加在BN后面
        # bias = tf.cast(bias, tf.float16)
        x = tf.nn.bias_add(x, bias)
    x = activation(x, act_type=act_type)
    return x


def DeConvBlock( x, in_channels, out_channels, kernel_size, stride=1, name="DeConvBlock",
                BN=True, use_bias=True, padding="VALID", act_type="relu",
                 scale=2, BN_type="IN", istrain=True):

    b, h, w, c = x.get_shape().as_list()
    out_shape = [b, h * scale, w * scale, out_channels]
    if use_bias:
        weight, bias = init_params(kernel_size=kernel_size, in_channels=out_channels,
                                    out_channels=in_channels, use_bias=use_bias, name=name)
    else:
        weight = init_params(kernel_size=kernel_size, in_channels=out_channels,
                                    out_channels=in_channels, use_bias=use_bias, name=name)

    x = tf.nn.conv2d_transpose(x, filter=weight, output_shape=out_shape,
                               strides=[1, stride, stride, 1], padding=padding)
    if BN:
        if BN_type == "BN":
            x = batch_norm(x, training=istrain, name="%s_BN"%name)
        elif BN_type == "IN":
            x = instance_norm(x, name="%s_IN" % name, training=istrain)
        else:
            raise NotImplementedError('[ERROR] BN type [%s] is not implemented!' % BN_type)
    if use_bias:
        x = tf.nn.bias_add(x, bias)
    x = activation(x, act_type=act_type)
    return x


def Full_Connected_Block(x, outnum):
    return NotImplementedError


def DepthWise_Conv( x, in_channels, channel_multiplier, kernel_size, stride=1, name="DepthWise_Conv",
                   use_bias=True, BN=True, padding=True, act_type="relu",
                    BN_type="BN", istrain=True):
    if padding:
        pdsz = kernel_size // 2
        x = tf.pad(x, [[0, 0], [pdsz, pdsz], [pdsz, pdsz], [0, 0]], name="pad")

    weight, _ = init_params(kernel_size=kernel_size, in_channels=in_channels,
                                 out_channels=channel_multiplier, use_bias=False, name=name)
    x = tf.nn.depthwise_conv2d(x, filter=weight, strides=[1, stride, stride, 1], padding="VALID")
    if BN:
        if BN_type == "BN":
            x = batch_norm(x, training=istrain, name="%s_BN" % name)
        elif BN_type == "IN":
            x = instance_norm(x, name="%s_IN" % name, training=istrain)
        else:
            raise NotImplementedError('[ERROR] BN type [%s] is not implemented!' % BN_type)
    if use_bias:
        bias = _variable_on_cpu(shape=[in_channels * channel_multiplier], l2_loss=False, name="%s_b"%name)
        x = tf.nn.bias_add(x, bias)
    x = activation(x, act_type=act_type)
    return x

def BottleNeck( inputs, out_channels, kernel_size, stride, exp_size,
               padding=True, act_type="hswish", shortcut=True, name="bottleneck"):
    _, h, w, c = inputs.get_shape().as_list()
    x = ConvBlock(inputs, c, exp_size, kernel_size=1, stride=1, use_bias=False,
                       BN=True, padding=False, act_type=act_type, name="%s_Conv1x1" % name)  # 1*1*i*64*i*64

    x = DepthWise_Conv(x, in_channels=exp_size, channel_multiplier=1, kernel_size=kernel_size,
                            stride=stride, BN=True, use_bias=False, padding=padding, act_type=act_type,
                            name="DWC_%s" % name)

    x = ConvBlock(x, in_channels=exp_size, out_channels=out_channels, kernel_size=1, stride=1,
                       use_bias=False, BN=True, padding=False, act_type="linear", name="%s_last" % name)
    if shortcut:
        if c == out_channels and stride == 1:
            x += inputs
        elif c != out_channels and stride == 1:
            inputs = ConvBlock(inputs, c, out_channels, kernel_size=1, stride=1,
                                    BN=True, use_bias=False, padding=False, act_type="linear", name="%s_conv" % name)
            x += inputs
        else:
            Warning('stride = %d cannot shortcut, RESET shortcut False' % stride)
    return activation(x, act_type=act_type)