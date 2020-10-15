import tensorflow as tf
from tensorflow.python.framework import graph_io
import sys
import numpy as np
import json
import os

class basic_network(object):
    def __init__(self, cfg):
        if cfg is None:
            self.istrain = True
            self.BN_type = "BN"
            # self.scale = 2
            self.c_dim = 3
        else:
            self.cfg = cfg
            self.BN_type = self.cfg.BN_type
            self.istrain = self.cfg.istrain
            self.c_dim = self.cfg.c_dim
        self.params_count = 0
        self.gpu_count = 0
    
    def load_TurboLIU_Params(self, jsonParams):
        f = open(jsonParams, "r")
        Params = json.load(f)
        f.close()
        self.TurboParams = self.process_params(Params)

    def calc_param(self, x):
        _, h, w, c = x.get_shape().as_list()
        self.gpu_count += h*w*c

    def _variable_on_cpu(self, shape, weight_decay=0.995, l2_loss=True, name="var"):
        with tf.device("/cpu:0"):
            w = tf.compat.v1.get_variable(shape=shape, initializer=tf.random_normal_initializer(0.0, 0.01),
                                          dtype=tf.float32, name=name)
            # tf.summary.histogram("%s_w" % name, w)
            if l2_loss:
                tf.compat.v1.add_to_collection(name="weights_l2_loss", value=(1 - weight_decay) * tf.nn.l2_loss(w))
        return w

    def init_params(self, *args, **kwargs):
        # def _variable_on_cpu(w_shape, b_shape, weight_decay=0.995, use_bias=True, name="conv"):
        #     with tf.device("/cpu:0"):
        #         # with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        #             # w = tf.Variable(tf.random_normal(w_shape, 0.0, 1.0), trainable=True, name="%s_w" % name)
        #             w = tf.compat.v1.get_variable(shape=w_shape, initializer=tf.random_normal_initializer(0.0, 0.01),dtype=tf.float32, name="%s_w" % name)
        #             # tf.summary.histogram("%s_w" % name, w)
        #             # tf.add_to_collection(name="weights_l2_loss", value=self.calc_l2_loss(w, 1-weight_decay))
        #             tf.compat.v1.add_to_collection(name="weights_l2_loss", value=(1-weight_decay)*tf.nn.l2_loss(w))
        #             # b = tf.Variable(tf.zeros(b_shape), trainable=use_bias, name="%s_b" % name)
        #             b = tf.compat.v1.get_variable(shape=b_shape,initializer=tf.constant_initializer, trainable=use_bias,dtype=tf.float32, name="%s_b" % name)
        #     return w, b
        kernel_size = kwargs["kernel_size"]
        in_channels = kwargs["in_channels"]
        out_channels = kwargs["out_channels"]
        use_bias = kwargs["use_bias"]
        name = kwargs["name"]
        w_shape = [kernel_size, kernel_size, in_channels, out_channels]
        w = self._variable_on_cpu(w_shape, name="%s_w"%name)
        b = 0
        self.params_count += kernel_size * kernel_size * in_channels * out_channels
        if use_bias:
            b_shape = [out_channels]
            b = self._variable_on_cpu(b_shape, l2_loss=False, name="%s_b"%name)
            self.params_count += out_channels
        return w, b

    def calc_loss(self, *args, **kwargs):
        loss_type = kwargs["loss_type"]
        x = kwargs["x"]
        y = kwargs["y"]
        if loss_type == "L1":
            return tf.reduce_sum(tf.abs(x-y), name="L1_loss")
        elif loss_type == "L2":
            return tf.nn.l2_loss((x-y), name="L2_loss")

    def activation(self, *args, **kwargs):
        act_type = kwargs["act_type"]
        act_type = act_type.lower()
        if act_type == "relu":
            return tf.nn.relu(args[0])
        elif act_type == "relu6":
            return tf.nn.relu6(args[0])
        elif act_type == "lrelu":
            slope = kwargs["slope"]
            y = slope*args[0]
            return tf.maximum(args[0], y)
        elif act_type == "prelu":
            return tf.nn.leaky_relu(args[0], alpha=0.2)
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
            return self.hard_swish(args[0])
        elif act_type == "hsigmoid":
            return self.hard_sigmoid(args[0])
        elif act_type == "sigmoid":
            return tf.nn.sigmoid(args[0])
        elif act_type == "linear":
            return args[0]
        else:
            return NotImplementedError

    def calc_l2_loss(self, weight, weight_decay):
        _, _, _, outchannel = weight.get_shape().as_list()
        return (weight_decay) * tf.reduce_sum(tf.square(weight)) / outchannel

    def calc_l1_loss(self, weight, weight_decay):
        _, _, _, outchannel = weight.get_shape().as_list()
        return (weight_decay)*tf.reduce_sum(tf.abs(weight)) / outchannel

    def batch_norm_(self, x, training, name):
        x = tf.compat.v1.layers.batch_normalization(x, training=training, name=name)
        return x

    def batch_norm(self, *args, **kwargs):
        #///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        b, h, w, c = args[0].get_shape().as_list()
        name = kwargs["name"]
        decay = 0.99
        epsilon = 1e-3
        gama = tf.compat.v1.get_variable(shape=c, initializer=tf.ones_initializer, name="%s_gama" % name)
        beta = tf.compat.v1.get_variable(shape=c, initializer=tf.zeros_initializer, name="%s_beta" % name)
        pop_mean = tf.compat.v1.get_variable(shape=c, initializer=tf.zeros_initializer,trainable=True, name="%s_pop_mean" % name)
        pop_variance = tf.compat.v1.get_variable(shape=c, initializer=tf.ones_initializer,trainable=True, name="%s_pop_variance" % name)

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

    def instance_norm(self, *args, **kwargs):#///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        b, h, w, c = args[0].get_shape().as_list()
        name = kwargs["name"]
        decay = 0.99
        epsilon = 1e-3
        shape = [b, 1, 1, c]
        gama = tf.get_variable(shape=shape, initializer=tf.ones_initializer, name="%s_gama" % name)
        beta = tf.get_variable(shape=shape, initializer=tf.zeros_initializer, name="%s_beta" % name)
        pop_mean = tf.get_variable(shape=shape, initializer=tf.zeros_initializer,trainable=True, name="%s_pop_mean" % name)
        pop_variance = tf.get_variable(shape=shape, initializer=tf.ones_initializer,trainable=True, name="%s_pop_variance" % name)

        if kwargs["training"] == True:
            average_mean, average_varance = tf.nn.moments(args[0], axes=[1, 2], keep_dims=True, name=name)

            train_mean = tf.assign(pop_mean, pop_mean * decay + average_mean * (1 - decay))
            train_variance = tf.assign(pop_variance, pop_variance * decay + average_varance * (1 - decay))
            # tf.train.ExponentialMovingAverage
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, [train_mean, train_variance])
            # with tf.control_dependencies([train_mean, train_variance]):
            return tf.nn.batch_normalization(args[0], average_mean, average_varance, beta, gama, epsilon)
        else:
            return tf.nn.batch_normalization(args[0], pop_mean, pop_variance, beta, gama, epsilon)
        
        # return layers.instance_norm(args[0], kwargs["name"], trainable=kwargs["training"])

    def hard_sigmoid(self, x):
        return tf.nn.relu6((x+3)/6)

    def hard_swish(self, x):
        return x * self.hard_sigmoid(x)

    def global_average_pooling(self, x, name="GAP"):
        return tf.compat.v1.reduce_mean(x, axis=[1, 2], keep_dims=True, name="Global_Average_Pooling_%s" % name)

    def DropOut(self, x, droprate, training):
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

    def ResBlock(self, x, outdim, name="ResBlock"):
        _, h, w, c = x.get_shape().as_list()
        ix = self.ConvBlock(x, in_channels=c, out_channels=outdim, kernel_size=3, stride=1,
                            BN=True, use_bias=False, padding=True,act_type="relu", name="%s_1"%name)
        ix = self.ConvBlock(ix, in_channels=outdim, out_channels=outdim, kernel_size=3, stride=1,
                            BN=True, use_bias=False, padding=True,act_type="relu", name="%s_2"%name)
        if c == outdim:
            ix = ix + x
        else:
            ix = self.ConvBlock(x, in_channels=c, out_channels=outdim, kernel_size=3, stride=1,
                            BN=True, use_bias=False, padding=True,act_type="relu", name="%s_uniform"%name)
            ix = ix + x
        return tf.nn.relu(ix)

    def SqzResBlock(self, x, outdim, name="ExpResBlock"):
        _, h, w, c = x.get_shape().as_list()
        cs = max(c//4, 16)
        ix = self.ConvBlock(x, in_channels=c, out_channels=cs, kernel_size=1, stride=1,
                            BN=True, use_bias=False, padding=False, act_type="relu", name="%s_1"%name)
        ix = self.ConvBlock(ix, in_channels=cs, out_channels=cs, kernel_size=3, stride=1,
                            BN=True, use_bias=False, padding=True, act_type="relu", name="%s_2"%name)
        ix = self.ConvBlock(ix, in_channels=cs, out_channels=outdim, kernel_size=1, stride=1,
                            BN=True, use_bias=False, padding=False, act_type="relu", name="%s_3"%name)
        if c == outdim:
            ix = ix + x
        else:
            x = self.ConvBlock(x, in_channels=c, out_channels=outdim, kernel_size=3, stride=1,
                                BN=True, use_bias=False, padding=True, act_type="relu", name="%s_uniform" % name)
            ix = ix + x
        return tf.nn.relu(ix)

    def ConvBlock(self, x, in_channels, out_channels, kernel_size, stride=1, name="ConvBlock",
                  BN=True, use_bias=True, padding=True, act_type="relu",
                  FUSE_BN=False):

        if padding:
            pdsz = kernel_size // 2
            x = tf.pad(x, [[0, 0], [pdsz, pdsz], [pdsz, pdsz], [0, 0]], name="pad")
            
        if (not self.istrain) and FUSE_BN and BN:
            return self.ConvBlock_FuseBN_with_jsonParams(x, in_channels, out_channels, kernel_size, stride=stride, 
                                                         name=name, use_bias=use_bias, act_type=act_type)
        elif (not self.istrain) and FUSE_BN and (not BN):
            return self.ConvBlock_with_jsonParams(x, in_channels, out_channels, kernel_size, stride=stride,
                                                         name=name, use_bias=use_bias, act_type=act_type)
        # x = tf.cast(x, tf.float16)
        weight, bias = self.init_params(kernel_size=kernel_size, in_channels=in_channels,
                                        out_channels=out_channels, use_bias=use_bias, name=name)

        x = tf.compat.v1.nn.conv2d(x, filter=weight, strides=[1, stride, stride, 1], padding="VALID")
        if BN:
            if self.BN_type == "BN":
                x = self.batch_norm(x, training=self.istrain, name="%s_BN"%name)
            elif self.BN_type == "IN":
                x = self.instance_norm(x, training=self.istrain, name="%s_IN"%name)
            else:
                raise NotImplementedError('[ERROR] BN type [%s] is not implemented!' % self.BN_type)
        if use_bias:
            # 注意： BN前加不加bias对结果都没有任何影响，由BN公式中可以推导出来
            # 这里bias加在BN后面
            # bias = tf.cast(bias, tf.float16)
            x = tf.nn.bias_add(x, bias)
        x = self.activation(x, act_type=act_type)
        return x

    def DeConvBlock(self, x, in_channels, out_channels, kernel_size, stride=1, name="DeConvBlock",
                    BN=True, use_bias=True, padding="VALID", act_type="relu",
                    FUSE_BN=False):
        b, h, w, c = x.get_shape().as_list()
        out_shape = [b, h * self.scale, w * self.scale, out_channels]

        if (not self.istrain) and FUSE_BN and BN:
            return self.DeConvBlock_FuseBN_with_jsonParams(x, in_channels, out_shape, kernel_size, stride=stride,
                                                         name=name, use_bias=use_bias, act_type=act_type)

        weight, bias = self.init_params(kernel_size=kernel_size, in_channels=out_channels,
                                        out_channels=in_channels, use_bias=use_bias, name=name)
        x = tf.nn.conv2d_transpose(x, filter=weight, output_shape=out_shape,
                                   strides=[1, stride, stride, 1], padding=padding)
        if BN:
            if self.BN_type == "BN":
                x = self.batch_norm(x, training=self.istrain)
            elif self.BN_type == "IN":
                x = self.instance_norm(x, name="%s_IN" % name, training=self.istrain)
            else:
                raise NotImplementedError('[ERROR] BN type [%s] is not implemented!' % self.BN_type)
        if use_bias:
            x = tf.nn.bias_add(x, bias)
        x = self.activation(x, act_type=act_type)
        return x

    def Full_Connected_Block(self, x, outnum):
        return NotImplementedError


    def DepthWise_Conv(self, x, in_channels, channel_multiplier, kernel_size, stride=1, name="DepthWise_Conv",
                  use_bias=True, BN=True, padding=True, act_type="relu",
                       FUSE_BN=False):
        if padding:
            pdsz = kernel_size // 2
            x = tf.pad(x, [[0, 0], [pdsz, pdsz], [pdsz, pdsz], [0, 0]], name="pad")
            
        if (not self.istrain) and FUSE_BN and BN:
            return self.DepthWise_FuseBN_with_jsonParams(x, in_channels, channel_multiplier, kernel_size, stride=stride, 
                                                         name=name, use_bias=use_bias, act_type=act_type)
        # if (not self.istrain) and FUSE_BN and BN:
        #     return self.DepthWise_with_jsonParams(x, in_channels, channel_multiplier, kernel_size, stride=stride,
        #                                                  name=name, use_bias=use_bias, act_type=act_type)
        weight, bias = self.init_params(kernel_size=kernel_size, in_channels=in_channels,
                                     out_channels=channel_multiplier, use_bias=False, name=name)
        # weight = tf.where(tf.less(tf.abs(weight), self.cfg.warp_threshold), tf.zeros_like(weight), weight)
        # weight = tf.cast(weight, tf.float16)
        # print("before depth", x)
        x = tf.nn.depthwise_conv2d(x, filter=weight, strides=[1, stride, stride, 1], padding="VALID")
        # print("after depth", x)
        if BN:
            if self.BN_type == "BN":
                x = self.batch_norm(x, training=self.istrain, name="%s_BN" % name)
            elif self.BN_type == "IN":
                x = self.instance_norm(x, name="%s_IN" % name, training=self.istrain)
            else:
                raise NotImplementedError('[ERROR] BN type [%s] is not implemented!' % self.BN_type)
        if use_bias:
            # bias = tf.compat.v1.get_variable(shape=[in_channels * channel_multiplier], trainable=True, name="%s_b" % name)
            bias = self._variable_on_cpu([in_channels*channel_multiplier], l2_loss=False, name="%s_b"%name)
            self.params_count += (in_channels * channel_multiplier - channel_multiplier)
            x = tf.nn.bias_add(x, bias)
        x = self.activation(x, act_type=act_type)
        return x

    def BottleNeck(self, inputs, out_channels, kernel_size, stride, exp_size,
                      padding=True, act_type="hswish", shortcut=True, name="bottleneck",
                   FUSE_BN=False):
        _, h, w, c = inputs.get_shape().as_list()
        x = self.ConvBlock(inputs, c, exp_size, kernel_size=1, stride=1, use_bias=True,
                           BN=True, padding=False, act_type=act_type, name="%s_Conv1x1"%name,
                           FUSE_BN=FUSE_BN) # 1*1*i*64*i*64

        x = self.DepthWise_Conv(x, in_channels=exp_size, channel_multiplier=1, kernel_size=kernel_size,
                                stride=stride, BN=True, use_bias=False, padding=padding, act_type=act_type, name="DWC_%s"%name,
                                FUSE_BN=True)
        if "bottleneck128_2" in name:
            self.x = tf.identity(x)
        x = self.ConvBlock(x, in_channels=exp_size, out_channels=out_channels, kernel_size=1, stride=1,
                           use_bias=True, BN=True, padding=False, act_type="linear", name="%s_last"%name,
                           FUSE_BN=FUSE_BN)
        if shortcut:
            if c == out_channels and stride == 1:
                x += inputs
            elif c!=out_channels and stride == 1:
                inputs = self.ConvBlock(inputs, c, out_channels, kernel_size=1, stride=1,
                                        BN=True, use_bias=True, padding=False, act_type="linear", name="%s_conv"%name,
                                        FUSE_BN=FUSE_BN)
                x += inputs
            else:
                Warning('stride = %d cannot shortcut, RESET shortcut False'%stride)
                
        return self.activation(x, act_type=act_type)

    def DepthWise_with_jsonParams(self, x, in_channels, channel_multiplier, kernel_size, stride=1,
                                   name="Depthwise", use_bias=False, act_type="relu"):
        w_name = "%s_w"%name
        gama_name, beta_name, mean_name, variance_name = self.get_batchnorm_name(name="%s_BN"%name)

        weight = self.TurboParams[w_name]["value"]
        gama = self.TurboParams[gama_name]["value"]
        beta = self.TurboParams[beta_name]["value"]
        mean = self.TurboParams[mean_name]["value"]
        variance = self.TurboParams[variance_name]["value"]
        esp = 0.001#0.0010000000474974513
        
        weight_var = tf.compat.v1.Variable(initial_value=weight, dtype=tf.float32, name="%s_w" % name)

        x = tf.compat.v1.nn.depthwise_conv2d(x, filter=weight_var, strides=[1, stride, stride, 1], padding="VALID")
        
        gama_var = tf.compat.v1.Variable(initial_value=gama, dtype=tf.float32, name="%s_gama" % name)
        beta_var = tf.compat.v1.Variable(initial_value=beta, dtype=tf.float32, name="%s_beta" % name)
        pop_mean_var = tf.compat.v1.Variable(initial_value=mean, dtype=tf.float32, name="%s_pop_mean" % name)
        pop_variance_var = tf.compat.v1.Variable(initial_value=variance, dtype=tf.float32, name="%s_pop_variance" % name)

        x = tf.nn.batch_normalization(x, pop_mean_var, pop_variance_var, beta_var, gama_var, esp)
        x = self.activation(x, act_type=act_type)
        return x

    def ConvBlock_with_jsonParams(self, x, in_channels, out_channels, kernel_size, stride=1,
                              name="", use_bias=True, act_type="relu"):
        w_name = "%s_w" % name
        weight = self.TurboParams[w_name]["value"]
        weight_var = tf.compat.v1.Variable(initial_value=weight, dtype=tf.float32, name="%s_w" % name)
        x = tf.compat.v1.nn.conv2d(x, filter=weight_var, strides=[1, stride, stride, 1], padding="VALID")
        if use_bias:
            bias = self.TurboParams["%s_b"%name]["value"]
            bias_var = tf.compat.v1.Variable(initial_value=bias, dtype=tf.float32, name="%s_b" % name)
            x = tf.compat.v1.nn.bias_add(x, bias_var)
        x = self.activation(x, act_type=act_type)
        return x
    
########################################################################################################################
    def process_params(self, params):
        new_params = {}
        for name, param in params.items():
            new_param = {}
            value = np.array(param["value"], dtype=np.float32)
            shape = param["shape"]
            # value = value.reshape(shape)
            new_param["value"] = value
            new_param["shape"] = shape
            new_params[name] = new_param
        return new_params

    def get_batchnorm_name(self, name="conv_BN"):
        keynames = self.TurboParams.keys()
        if "%s_gama"%name in keynames:
            gama_name = "%s_gama"%name
            beta_name = "%s_beta"%name
            mean_name = "%s_pop_mean"%name
            variance_name = "%s_pop_variance"%name
        else:
            gama_name = "%s/gamma"%name
            beta_name = "%s/beta"%name
            mean_name = "%s/moving_mean"%name
            variance_name = "%s/moving_variance"%name
        return gama_name, beta_name, mean_name, variance_name

    def fuse_w(self, w, gama, variance, esp):
        channel = w.shape[3]
        for i in range(channel):
            w[..., i] = w[..., i] * gama[i] / np.sqrt(variance[i]+esp)
        return w.copy()

    def fuse_b(self, b, gama, beta, mean, variance, esp):
        return gama*(b - mean)/np.sqrt(variance + esp) + beta

    def depwise_fuse_w(self, w, gama, variance, esp):
        channel = w.shape[2]
        for i in range(channel):
            w[:,:,i,:] = w[:,:,i,:] * gama[i] / np.sqrt(variance[i] + esp)
        return w.copy()

    def depwise_fuse_b(self, b, gama, beta, mean, variance, esp):
        return self.fuse_b(b, gama, beta, mean, variance, esp)# 目测和conv的计算方式一样

    def ConvBlock_FuseBN_with_jsonParams(self, x, in_channels, out_channels, kernel_size, stride=1,
                                            name="ConvBlock", use_bias=True, act_type="relu"):
        w_name = "%s_w"%name
        gama_name, beta_name, mean_name, variance_name = self.get_batchnorm_name(name="%s_BN"%name)

        weight = self.TurboParams[w_name]["value"]
        gama = self.TurboParams[gama_name]["value"]
        beta = self.TurboParams[beta_name]["value"]
        mean = self.TurboParams[mean_name]["value"]
        # print(mean)
        variance = self.TurboParams[variance_name]["value"]
        esp = 0.001#0.0010000000474974513

        fused_weight = self.fuse_w(weight, gama, variance, esp=esp)
        # warp_weight = self.cut_weight(fused_weight, percent=0.4)
        weight_var = tf.compat.v1.Variable(initial_value=fused_weight, dtype=tf.float32, name="%s_w" % name)

        x = tf.compat.v1.nn.conv2d(x, filter=weight_var, strides=[1, stride, stride, 1], padding="VALID")

        bias = np.zeros(shape=self.TurboParams[w_name]["shape"][3], dtype=np.float32)
        fused_bias = self.fuse_b(bias, gama, beta, mean, variance, esp=esp)
        bias_var0 = tf.compat.v1.Variable(initial_value=fused_bias, dtype=tf.float32, name="%s_b0" % name)
        x = tf.compat.v1.nn.bias_add(x, bias_var0)
        # print(fused_bias)

        if use_bias:
            bias_name = "%s_b"%name
            bias_ = self.TurboParams[bias_name]["value"]
            bias_var1 = tf.compat.v1.Variable(initial_value=bias_, name="%s_b1" % name)
            x = tf.compat.v1.nn.bias_add(x, bias_var1)
        # else:
        #     bias = np.zeros(shape=self.TurboParams[w_name]["shape"][3], dtype=np.float32)
        # fused_bias = self.fuse_b(bias, gama, beta, mean, variance, esp=esp)

        x = self.activation(x, act_type=act_type)
        return x

    def DepthWise_FuseBN_with_jsonParams(self, x, in_channels, channel_multiplier, kernel_size, stride,
                                     name="DepthWise_Conv", use_bias=True, act_type="relu"):
        w_name = "%s_w" % name
        gama_name, beta_name, mean_name, variance_name = self.get_batchnorm_name(name="%s_BN" % name)

        weight = self.TurboParams[w_name]["value"]
        gama = self.TurboParams[gama_name]["value"]
        beta = self.TurboParams[beta_name]["value"]
        mean = self.TurboParams[mean_name]["value"]
        variance = self.TurboParams[variance_name]["value"]
        esp = 0.001# 0.0010000000474974513

        fused_weight = self.depwise_fuse_w(weight, gama, variance, esp=esp)
        # warp_weight = self.cut_weight(fused_weight, percent=0.4)
        weight_var = tf.compat.v1.Variable(initial_value=fused_weight, dtype=tf.float32, name="%s_w" % name)

        x = tf.compat.v1.nn.depthwise_conv2d(x, filter=weight_var, strides=[1, stride, stride, 1], padding="VALID")

        bias = np.zeros(shape=self.TurboParams[w_name]["shape"][2], dtype=np.float32)
        fused_bias = self.depwise_fuse_b(bias, gama, beta, mean, variance, esp=esp)
        bias_var0 = tf.compat.v1.Variable(initial_value=fused_bias, name="%s_b0" % name)
        x = tf.compat.v1.nn.bias_add(x, bias_var0)
        # if "bottleneck128_2" in name:
        #     self.x = tf.identity(x)
        if use_bias:
            bias_name = "%s_b" % name
            bias = self.TurboParams[bias_name]["value"]
            bias_var1 = tf.compat.v1.Variable(initial_value=bias, name="%s_b1" % name)
            x = tf.compat.v1.nn.bias_add(x, bias_var1)

        # else:
        #     bias = np.zeros(shape=self.TurboParams[w_name]["shape"][3], dtype=np.float32)
        # fused_bias = self.depwise_fuse_b(bias, gama, beta, mean, variance, esp=esp)
        x = self.activation(x, act_type=act_type)
        return x

    def DeConvBlock_FuseBN_with_jsonParams(self, x, in_channels, out_shape, kernel_size, stride=1,
                                                         name="DeConvBlock", use_bias=True, act_type="relu"):
        # 该函数尚未验证
        w_name = "%s_w" % name
        gama_name, beta_name, mean_name, variance_name = self.get_batchnorm_name(name="%s_BN" % name)

        weight = self.TurboParams[w_name]["value"]
        gama = self.TurboParams[gama_name]["value"]
        beta = self.TurboParams[beta_name]["value"]
        mean = self.TurboParams[mean_name]["value"]
        variance = self.TurboParams[variance_name]["value"]
        esp = 0.0010000000474974513

        fused_weight = self.depwise_fuse_w(weight, gama, variance, esp=esp)
        # warp_weight = self.cut_weight(fused_weight, percent=0.0)
        weight_var = tf.compat.v1.Variable(initial_value=fused_weight, dtype=tf.float32, name="%s_w" % name)

        x = tf.compat.v1.nn.conv2d_transpose(x, filter=weight_var, output_shape=out_shape,
                                             strides=[1, stride, stride, 1], padding="VALID")
        if use_bias:
            bias_name = "%s_b"%name
            bias = self.TurboParams[bias_name]["value"]
            bias_var = tf.compat.v1.Variable(initial_value=bias, name="%s_b" % name)
            x = tf.compat.v1.nn.bias_add(x, bias_var)
        # else:
        #     bias = np.zeros(shape=self.TurboParams[weight_name]["shape"][2], dtype=np.float32) # 不确定是inchannel还是outchannel
        # fused_bias = self.fuse_b(bias, gama, beta, mean, variance, esp=esp)

        x = self.activation(x, act_type=act_type)
        return x

    def save_pb_with_fuseBN(self, sess, outnodes, dstPath, dstname):
        sess.run(tf.compat.v1.global_variables_initializer())
        # saver.save(sess, "./centerface_onnx")
        frozen = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def, outnodes)#,["heatmap","538","539","540"]["heatmap", "scale", "offset", "landmark"]
        graph_io.write_graph(frozen, dstPath, dstname, as_text=False)



def pb2jsonParams(pbfile, paramfile):
    sess = tf.compat.v1.Session()
    graph = tf.compat.v1.GraphDef()
    with open(pbfile, "rb") as f:
        graph.ParseFromString(f.read())
        tf.import_graph_def(graph, name="")
    print("...")


class TurboBASE(object):
    def __init__(self, cfg):
        if cfg is None:
            self.istrain = True
            self.BN_type = "BN"
            # self.scale = 2
            self.c_dim = 3
            self.load_premodel = False
            self.weight_decay = 0.995
            self.FUSE_BN=False
        else:
            self.cfg = cfg
            self.BN_type = self.cfg.BN_type
            self.istrain = self.cfg.istrain
            self.c_dim = self.cfg.c_dim
            self.load_premodel = self.cfg.load_premodel
            self.weight_decay = self.cfg.weight_decay
            self.FUSE_BN = False
        self.params_count = 0
        self.gpu_count = 0
        self.lrplaceholder = tf.compat.v1.placeholder(dtype=tf.float32, name="learningRate")

    def get_latest_count(self, path):
        names = os.listdir(path)
        counts = [int(name.split(".")[0].split("-")[-1]) for name in names]
        return max(counts)

    def load_TurboLIU_Params(self, jsonParams):
        f = open(jsonParams, "r")
        Params = json.load(f)
        f.close()
        self.TurboParams = self.process_params(Params)

    def calc_param(self, x):
        _, h, w, c = x.get_shape().as_list()
        self.gpu_count += h * w * c

    def _variable_on_cpu(self, shape, weight_decay=0.995, l2_loss=True, name="var"):
        with tf.device("/cpu:0"):
            w = tf.compat.v1.get_variable(shape=shape, initializer=tf.random_normal_initializer(0.0, 0.01),
                                          dtype=tf.float32, name=name)
            # tf.summary.histogram("%s_w" % name, w)
            if l2_loss:
                tf.compat.v1.add_to_collection(name="weights_l2_loss", value=(1 - weight_decay) * tf.nn.l2_loss(w))
        return w

    def load_param(self, name, l2_loss=False):
        value = self.TurboParams[name]["value"]
        with tf.device("/cpu:0"):
            v = tf.compat.v1.Variable(initial_value=value, trainable=True, name=name)
        if l2_loss:
            tf.compat.v1.add_to_collection(name="weights_l2_loss", value=(1 - self.weight_decay) * tf.nn.l2_loss(v))
        return v

    def get_weight(self, *args, **kwargs):
        name = kwargs["name"]
        if self.load_premodel:
            return self.load_param(name=name, l2_loss=True)
        kernel_size = kwargs["kernel_size"]
        in_channels = kwargs["in_channels"]
        out_channels = kwargs["out_channels"]
        w_shape = [kernel_size, kernel_size, in_channels, out_channels]
        w = self._variable_on_cpu(w_shape, name="%s" % name)
        self.params_count += kernel_size * kernel_size * in_channels * out_channels
        return w

    def get_bias(self, *args, **kwargs):
        name = kwargs["name"]
        if self.load_premodel:
            return self.load_param(name=name, l2_loss=False)
        out_channels = kwargs["out_channels"]
        b_shape = [out_channels]
        b = self._variable_on_cpu(b_shape, l2_loss=False, name="%s" % name)
        self.params_count += out_channels
        return b

    def calc_loss(self, *args, **kwargs):
        loss_type = kwargs["loss_type"]
        x = kwargs["x"]
        y = kwargs["y"]
        if loss_type == "L1":
            return tf.reduce_sum(tf.abs(x - y), name="L1_loss")
        elif loss_type == "L2":
            return tf.nn.l2_loss((x - y), name="L2_loss")

    def activation(self, *args, **kwargs):
        act_type = kwargs["act_type"]
        act_type = act_type.lower()
        if act_type == "relu":
            return tf.nn.relu(args[0])
        elif act_type == "relu6":
            return tf.nn.relu6(args[0])
        elif act_type == "lrelu":
            slope = kwargs["slope"]
            y = slope * args[0]
            return tf.maximum(args[0], y)
        elif act_type == "prelu":
            return tf.nn.leaky_relu(args[0], alpha=0.2)
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
            return self.hard_swish(args[0])
        elif act_type == "hsigmoid":
            return self.hard_sigmoid(args[0])
        elif act_type == "sigmoid":
            return tf.nn.sigmoid(args[0])
        elif act_type == "linear":
            return args[0]
        else:
            return NotImplementedError

    def calc_l2_loss(self, weight, weight_decay):
        _, _, _, outchannel = weight.get_shape().as_list()
        return (weight_decay) * tf.reduce_sum(tf.square(weight)) / outchannel

    def calc_l1_loss(self, weight, weight_decay):
        _, _, _, outchannel = weight.get_shape().as_list()
        return (weight_decay) * tf.reduce_sum(tf.abs(weight)) / outchannel

    def batch_norm(self, *args, **kwargs):
        # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        b, h, w, c = args[0].get_shape().as_list()
        name = kwargs["name"]
        decay = 0.995
        epsilon = 1e-5
        if self.load_premodel:
            gama = self.load_param("%s_gama"%name)
            beta = self.load_param("%s_beta"%name)
            pop_mean = self.load_param("%s_pop_mean"%name)
            pop_variance = self.load_param("%s_pop_variance"%name)
        else:
            with tf.device("/cpu:0"):
                gama = tf.compat.v1.get_variable(shape=c, initializer=tf.ones_initializer, trainable=True, name="%s_gama" % name)
                beta = tf.compat.v1.get_variable(shape=c, initializer=tf.zeros_initializer, trainable=True, name="%s_beta" % name)
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

    def instance_norm(self, *args,
                      **kwargs):  # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        b, h, w, c = args[0].get_shape().as_list()
        name = kwargs["name"]
        decay = 0.99
        epsilon = 1e-3
        shape = [b, 1, 1, c]
        gama = tf.get_variable(shape=shape, initializer=tf.ones_initializer, name="%s_gama" % name)
        beta = tf.get_variable(shape=shape, initializer=tf.zeros_initializer, name="%s_beta" % name)
        pop_mean = tf.get_variable(shape=shape, initializer=tf.zeros_initializer, trainable=True,
                                   name="%s_pop_mean" % name)
        pop_variance = tf.get_variable(shape=shape, initializer=tf.ones_initializer, trainable=True,
                                       name="%s_pop_variance" % name)

        if kwargs["training"] == True:
            average_mean, average_varance = tf.nn.moments(args[0], axes=[1, 2], keep_dims=True, name=name)

            train_mean = tf.assign(pop_mean, pop_mean * decay + average_mean * (1 - decay))
            train_variance = tf.assign(pop_variance, pop_variance * decay + average_varance * (1 - decay))
            # tf.train.ExponentialMovingAverage
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, [train_mean, train_variance])
            # with tf.control_dependencies([train_mean, train_variance]):
            return tf.nn.batch_normalization(args[0], average_mean, average_varance, beta, gama, epsilon)
        else:
            return tf.nn.batch_normalization(args[0], pop_mean, pop_variance, beta, gama, epsilon)

        # return layers.instance_norm(args[0], kwargs["name"], trainable=kwargs["training"])

    def hard_sigmoid(self, x):
        return tf.nn.relu6((x + 3) / 6)

    def hard_swish(self, x):
        return x * self.hard_sigmoid(x)

    def global_average_pooling(self, x, name="GAP"):
        return tf.compat.v1.reduce_mean(x, axis=[1, 2], keep_dims=True, name="Global_Average_Pooling_%s" % name)

    def DropOut(self, x, droprate, training):
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

    def ConvBlock(self, x, in_channels, out_channels, kernel_size, stride=1, name="ConvBlock",
                  BN=True, use_bias=True, padding=True, act_type="relu"):

        if padding:
            pdsz = kernel_size // 2
            x = tf.pad(x, [[0, 0], [pdsz, pdsz], [pdsz, pdsz], [0, 0]], name="pad")

        if (not self.istrain) and self.FUSE_BN and BN:
            return self.ConvBlock_FuseBN_with_jsonParams(x, in_channels, out_channels, kernel_size, stride=stride,
                                                         name=name, use_bias=use_bias, act_type=act_type)

        weight = self.get_weight(kernel_size=kernel_size, in_channels=in_channels,
                                        out_channels=out_channels, name="%s_w"%name)

        x = tf.compat.v1.nn.conv2d(x, filter=weight, strides=[1, stride, stride, 1], padding="VALID")
        if BN:
            if self.BN_type == "BN":
                x = self.batch_norm(x, training=self.istrain, name="%s_BN" % name)
            elif self.BN_type == "IN":
                x = self.instance_norm(x, training=self.istrain, name="%s_IN" % name)
            else:
                raise NotImplementedError('[ERROR] BN type [%s] is not implemented!' % self.BN_type)
        if use_bias:
            bias = self.get_bias(out_channels=out_channels, name="%s_b"%name)
            x = tf.nn.bias_add(x, bias)
        x = self.activation(x, act_type=act_type)
        return x

    def DeConvBlock(self, x, in_channels, out_channels, kernel_size, stride=1, scale=2, name="DeConvBlock",
                    BN=True, use_bias=True, padding=False, act_type="relu"):
        b, h, w, c = x.get_shape().as_list()
        out_shape = [b, h * scale, w * scale, out_channels]
        if padding:
            pdsz = kernel_size // 2
            x = tf.pad(x, [[0, 0], [pdsz, pdsz], [pdsz, pdsz], [0, 0]], name="pad")

        if (not self.istrain) and self.FUSE_BN and BN:
            return self.DeConvBlock_FuseBN_with_jsonParams(x, in_channels, out_shape, kernel_size, stride=stride,
                                                           name=name, use_bias=use_bias, act_type=act_type)
        weight = self.get_weight(kernel_size=kernel_size, in_channels=out_channels,
                                        out_channels=in_channels, name="%s_w"%name)
        x = tf.compat.v1.nn.conv2d_transpose(x, filter=weight, output_shape=out_shape,
                                   strides=[1, stride, stride, 1], padding="VALID")
        if BN:
            if self.BN_type == "BN":
                x = self.batch_norm(x, training=self.istrain, name="%s_BN"%name)
            elif self.BN_type == "IN":
                x = self.instance_norm(x, name="%s_IN" % name, training=self.istrain)
            else:
                raise NotImplementedError('[ERROR] BN type [%s] is not implemented!' % self.BN_type)
        if use_bias:
            bias = self.get_bias(out_channels=in_channels, name="%s_b"%name)
            x = tf.compat.v1.nn.bias_add(x, bias)
        x = self.activation(x, act_type=act_type)
        return x

    def Full_Connected_Block(self, x, outnum):
        return NotImplementedError

    def DepthWise_Conv(self, x, in_channels, channel_multiplier, kernel_size, stride=1, name="DepthWise_Conv",
                       use_bias=True, BN=True, padding=True, act_type="relu"):
        if padding:
            pdsz = kernel_size // 2
            x = tf.pad(x, [[0, 0], [pdsz, pdsz], [pdsz, pdsz], [0, 0]], name="pad")

        if (not self.istrain) and self.FUSE_BN and BN:
            return self.DepthWise_FuseBN_with_jsonParams(x, in_channels, channel_multiplier, kernel_size, stride=stride,
                                                         name=name, use_bias=use_bias, act_type=act_type)
        weight = self.get_weight(kernel_size=kernel_size, in_channels=in_channels,
                                        out_channels=channel_multiplier, name="%s_w"%name)
        x = tf.nn.depthwise_conv2d(x, filter=weight, strides=[1, stride, stride, 1], padding="VALID")
        # print("after depth", x)
        if BN:
            if self.BN_type == "BN":
                x = self.batch_norm(x, training=self.istrain, name="%s_BN" % name)
            elif self.BN_type == "IN":
                x = self.instance_norm(x, name="%s_IN" % name, training=self.istrain)
            else:
                raise NotImplementedError('[ERROR] BN type [%s] is not implemented!' % self.BN_type)
        if use_bias:
            bias = self.get_bias(out_channels=in_channels*channel_multiplier, name="%s_b"%name)
            self.params_count += (in_channels * channel_multiplier - channel_multiplier)
            x = tf.nn.bias_add(x, bias)
        x = self.activation(x, act_type=act_type)
        return x

    def BottleNeck(self, inputs, out_channels, kernel_size, stride, exp_size,
                   padding=True, act_type="hswish", shortcut=True, name="bottleneck"):
        _, h, w, c = inputs.get_shape().as_list()
        x = self.ConvBlock(inputs, c, exp_size, kernel_size=1, stride=1, use_bias=True,
                           BN=True, padding=False, act_type=act_type, name="%s_Conv1x1" % name)  # 1*1*i*64*i*64

        x = self.DepthWise_Conv(x, in_channels=exp_size, channel_multiplier=1, kernel_size=kernel_size,
                                stride=stride, BN=True, use_bias=False, padding=padding, act_type=act_type,
                                name="DWC_%s" % name)
        # if "bottleneck128_2" in name:
        #     self.x = tf.identity(x)
        x = self.ConvBlock(x, in_channels=exp_size, out_channels=out_channels, kernel_size=1, stride=1,
                           use_bias=True, BN=True, padding=False, act_type="linear", name="%s_last" % name)
        if shortcut:
            if c == out_channels and stride == 1:
                x += inputs
            elif c != out_channels and stride == 1:
                inputs = self.ConvBlock(inputs, c, out_channels, kernel_size=1, stride=1,
                                        BN=True, use_bias=True, padding=False, act_type="linear", name="%s_conv" % name)
                x += inputs
            else:
                Warning('stride = %d cannot shortcut, RESET shortcut False' % stride)

        return self.activation(x, act_type=act_type)

    def SqzResBlock(self, x, outdim, name="ExpResBlock"):
        _, h, w, c = x.get_shape().as_list()
        cs = max(c // 4, 16)
        ix = self.ConvBlock(x, in_channels=c, out_channels=cs, kernel_size=1, stride=1,
                            BN=True, use_bias=False, padding=False, act_type="relu", name="%s_1" % name)
        ix = self.ConvBlock(ix, in_channels=cs, out_channels=cs, kernel_size=3, stride=1,
                            BN=True, use_bias=False, padding=True, act_type="relu", name="%s_2" % name)
        ix = self.ConvBlock(ix, in_channels=cs, out_channels=outdim, kernel_size=1, stride=1,
                            BN=True, use_bias=False, padding=False, act_type="relu", name="%s_3" % name)
        if c == outdim:
            ix = ix + x
        else:
            x = self.ConvBlock(x, in_channels=c, out_channels=outdim, kernel_size=3, stride=1,
                               BN=True, use_bias=False, padding=True, act_type="relu", name="%s_uniform" % name)
            ix = ix + x
        return tf.nn.relu(ix)

    def ResBlock(self, x, outdim, name="ResBlock"):
        _, h, w, c = x.get_shape().as_list()
        ix = self.ConvBlock(x, in_channels=c, out_channels=outdim, kernel_size=3, stride=1,
                            BN=True, use_bias=False, padding=True, act_type="relu", name="%s_1" % name)
        ix = self.ConvBlock(ix, in_channels=outdim, out_channels=outdim, kernel_size=3, stride=1,
                            BN=True, use_bias=False, padding=True, act_type="relu", name="%s_2" % name)
        if c == outdim:
            ix = ix + x
        else:
            ix = self.ConvBlock(x, in_channels=c, out_channels=outdim, kernel_size=3, stride=1,
                                BN=True, use_bias=False, padding=True, act_type="relu", name="%s_uniform" % name)
            ix = ix + x
        return tf.nn.relu(ix)

    def DepthWise_with_jsonParams(self, x, in_channels, channel_multiplier, kernel_size, stride=1,
                                  name="Depthwise", use_bias=False, act_type="relu"):
        w_name = "%s_w" % name
        gama_name, beta_name, mean_name, variance_name = self.get_batchnorm_name(name="%s_BN" % name)

        weight = self.TurboParams[w_name]["value"]
        gama = self.TurboParams[gama_name]["value"]
        beta = self.TurboParams[beta_name]["value"]
        mean = self.TurboParams[mean_name]["value"]
        variance = self.TurboParams[variance_name]["value"]
        esp = 0.001  # 0.0010000000474974513

        weight_var = tf.compat.v1.Variable(initial_value=weight, dtype=tf.float32, name="%s_w" % name)

        x = tf.compat.v1.nn.depthwise_conv2d(x, filter=weight_var, strides=[1, stride, stride, 1], padding="VALID")

        gama_var = tf.compat.v1.Variable(initial_value=gama, dtype=tf.float32, name="%s_gama" % name)
        beta_var = tf.compat.v1.Variable(initial_value=beta, dtype=tf.float32, name="%s_beta" % name)
        pop_mean_var = tf.compat.v1.Variable(initial_value=mean, dtype=tf.float32, name="%s_pop_mean" % name)
        pop_variance_var = tf.compat.v1.Variable(initial_value=variance, dtype=tf.float32,
                                                 name="%s_pop_variance" % name)

        x = tf.nn.batch_normalization(x, pop_mean_var, pop_variance_var, beta_var, gama_var, esp)
        x = self.activation(x, act_type=act_type)
        return x

    # def ConvBlock_with_jsonParams(self, x, in_channels, out_channels, kernel_size, stride=1,
    #                               name="", use_bias=True, act_type="relu"):
    #     w_name = "%s_w" % name
    #     weight = self.TurboParams[w_name]["value"]
    #     weight_var = tf.compat.v1.Variable(initial_value=weight, dtype=tf.float32, name="%s_w" % name)
    #     x = tf.compat.v1.nn.conv2d(x, filter=weight_var, strides=[1, stride, stride, 1], padding="VALID")
    #     if use_bias:
    #         bias = self.TurboParams["%s_b" % name]["value"]
    #         bias_var = tf.compat.v1.Variable(initial_value=bias, dtype=tf.float32, name="%s_b" % name)
    #         x = tf.compat.v1.nn.bias_add(x, bias_var)
    #     x = self.activation(x, act_type=act_type)
    #     return x

    ########################################################################################################################
    def process_params(self, params):
        new_params = {}
        for name, param in params.items():
            new_param = {}
            name = name.split(":")[0]
            value = np.array(param["value"], dtype=np.float32)
            shape = param["shape"]
            # value = value.reshape(shape)
            new_param["value"] = value
            new_param["shape"] = shape
            new_params[name] = new_param
        return new_params

    def get_batchnorm_name(self, name="conv_BN"):
        keynames = self.TurboParams.keys()
        gama_name = "%s_gama" % name
        beta_name = "%s_beta" % name
        mean_name = "%s_pop_mean" % name
        variance_name = "%s_pop_variance" % name
        assert gama_name in keynames
        assert beta_name in keynames
        assert mean_name in keynames
        assert variance_name in keynames

        return gama_name, beta_name, mean_name, variance_name

    def fuse_w(self, w, gama, variance, esp):
        channel = w.shape[3]
        for i in range(channel):
            w[..., i] = w[..., i] * gama[i] / np.sqrt(variance[i] + esp)
        return w.copy()

    def fuse_b(self, b, gama, beta, mean, variance, esp):
        return gama * (b - mean) / np.sqrt(variance + esp) + beta

    def depwise_fuse_w(self, w, gama, variance, esp):
        channel = w.shape[2]
        for i in range(channel):
            w[:, :, i, :] = w[:, :, i, :] * gama[i] / np.sqrt(variance[i] + esp)
        return w.copy()

    def depwise_fuse_b(self, b, gama, beta, mean, variance, esp):
        return self.fuse_b(b, gama, beta, mean, variance, esp)  # 目测和conv的计算方式一样

    def ConvBlock_FuseBN_with_jsonParams(self, x, in_channels, out_channels, kernel_size, stride=1,
                                         name="ConvBlock", use_bias=True, act_type="relu"):
        w_name = "%s_w" % name
        gama_name, beta_name, mean_name, variance_name = self.get_batchnorm_name(name="%s_BN" % name)

        weight = self.TurboParams[w_name]["value"]
        gama = self.TurboParams[gama_name]["value"]
        beta = self.TurboParams[beta_name]["value"]
        mean = self.TurboParams[mean_name]["value"]
        # print(mean)
        variance = self.TurboParams[variance_name]["value"]
        esp = 1e-5  # 0.0010000000474974513

        fused_weight = self.fuse_w(weight, gama, variance, esp=esp)
        # warp_weight = self.cut_weight(fused_weight, percent=0.4)
        weight_var = tf.compat.v1.Variable(initial_value=fused_weight, dtype=tf.float32, name="%s_w" % name)

        x = tf.compat.v1.nn.conv2d(x, filter=weight_var, strides=[1, stride, stride, 1], padding="VALID")

        bias = np.zeros(shape=self.TurboParams[w_name]["shape"][3], dtype=np.float32)
        fused_bias = self.fuse_b(bias, gama, beta, mean, variance, esp=esp)
        bias_var0 = tf.compat.v1.Variable(initial_value=fused_bias, dtype=tf.float32, name="%s_b0" % name)
        x = tf.compat.v1.nn.bias_add(x, bias_var0)
        # print(fused_bias)

        if use_bias:
            bias_name = "%s_b" % name
            bias_ = self.TurboParams[bias_name]["value"]
            bias_var1 = tf.compat.v1.Variable(initial_value=bias_, name="%s_b1" % name)
            x = tf.compat.v1.nn.bias_add(x, bias_var1)
        x = self.activation(x, act_type=act_type)
        return x

    def DepthWise_FuseBN_with_jsonParams(self, x, in_channels, channel_multiplier, kernel_size, stride,
                                         name="DepthWise_Conv", use_bias=True, act_type="relu"):
        w_name = "%s_w" % name
        gama_name, beta_name, mean_name, variance_name = self.get_batchnorm_name(name="%s_BN" % name)

        weight = self.TurboParams[w_name]["value"]
        gama = self.TurboParams[gama_name]["value"]
        beta = self.TurboParams[beta_name]["value"]
        mean = self.TurboParams[mean_name]["value"]
        variance = self.TurboParams[variance_name]["value"]
        esp = 0.00001  # 0.0010000000474974513

        fused_weight = self.depwise_fuse_w(weight, gama, variance, esp=esp)
        # warp_weight = self.cut_weight(fused_weight, percent=0.4)
        weight_var = tf.compat.v1.Variable(initial_value=fused_weight, dtype=tf.float32, name="%s_w" % name)

        x = tf.compat.v1.nn.depthwise_conv2d(x, filter=weight_var, strides=[1, stride, stride, 1], padding="VALID")

        bias = np.zeros(shape=self.TurboParams[w_name]["shape"][2], dtype=np.float32)
        fused_bias = self.depwise_fuse_b(bias, gama, beta, mean, variance, esp=esp)
        bias_var0 = tf.compat.v1.Variable(initial_value=fused_bias, name="%s_b0" % name)
        x = tf.compat.v1.nn.bias_add(x, bias_var0)
        # if "bottleneck128_2" in name:
        #     self.x = tf.identity(x)
        if use_bias:
            bias_name = "%s_b" % name
            bias = self.TurboParams[bias_name]["value"]
            bias_var1 = tf.compat.v1.Variable(initial_value=bias, name="%s_b1" % name)
            x = tf.compat.v1.nn.bias_add(x, bias_var1)

        x = self.activation(x, act_type=act_type)
        return x

    def DeConvBlock_FuseBN_with_jsonParams(self, x, in_channels, out_shape, kernel_size, stride=1,
                                           name="DeConvBlock", use_bias=True, act_type="relu"):
        # 该函数尚未验证
        w_name = "%s_w" % name
        gama_name, beta_name, mean_name, variance_name = self.get_batchnorm_name(name="%s_BN" % name)

        weight = self.TurboParams[w_name]["value"]
        gama = self.TurboParams[gama_name]["value"]
        beta = self.TurboParams[beta_name]["value"]
        mean = self.TurboParams[mean_name]["value"]
        variance = self.TurboParams[variance_name]["value"]
        esp = 0.00001

        fused_weight = self.depwise_fuse_w(weight, gama, variance, esp=esp)
        # warp_weight = self.cut_weight(fused_weight, percent=0.0)
        weight_var = tf.compat.v1.Variable(initial_value=fused_weight, dtype=tf.float32, name="%s_w" % name)

        x = tf.compat.v1.nn.conv2d_transpose(x, filter=weight_var, output_shape=out_shape,
                                             strides=[1, stride, stride, 1], padding="VALID")

        bias = np.zeros(shape=self.TurboParams[w_name]["shape"][3], dtype=np.float32)
        fused_bias = self.fuse_b(bias, gama, beta, mean, variance, esp=esp)
        bias_var0 = tf.compat.v1.Variable(initial_value=fused_bias, dtype=tf.float32, name="%s_b0" % name)

        x = tf.compat.v1.nn.bias_add(x, bias_var0)
        if use_bias:
            bias_name = "%s_b" % name
            bias = self.TurboParams[bias_name]["value"]
            bias_var1 = tf.compat.v1.Variable(initial_value=bias, name="%s_b1" % name)
            x = tf.compat.v1.nn.bias_add(x, bias_var1)
        x = self.activation(x, act_type=act_type)
        return x

    def get_lr_value(self, ep):
        ex = 0
        for i in self.cfg.lr_steps:
            if ep > i:
                ex += 1
        lr = self.cfg.learning_rate * (self.cfg.lr_gama ** ex)
        return lr

    def save_pb_with_fuseBN(self, sess, outnodes, dstPath, dstname):
        sess.run(tf.compat.v1.global_variables_initializer())
        # saver.save(sess, "./centerface_onnx")
        frozen = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                        outnodes)  # ,["heatmap","538","539","540"]["heatmap", "scale", "offset", "landmark"]
        graph_io.write_graph(frozen, dstPath, dstname, as_text=False)

    def save(self, sess, name):
        # import json
        weights = dict()
        vars_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        params = sess.run(vars_list)
        for var,value in zip(vars_list, params):
            weight = dict()
            # value = sess.run(var)
            # print(var.name, var.shape)
            # weight["node_name"] = var.name
            weight["value"] = value.tolist()
            weight["shape"] = list(value.shape)
            weight["dtype"] = str(value.dtype)
            weights[var.name] = weight
        f = open(name, "w")
        file = json.dumps(weights)
        f.write(file)
        f.close()

if __name__ == '__main__':
    pb = r'E:\Object_Detection\Face_Detection\PFLD-master\PFLD-tiny-bn.pb'
    paramfile = r'E:\Object_Detection\Face_Detection\PFLD-master\PFLD-tiny-bn.json'
    pb2jsonParams(pb, paramfile)
