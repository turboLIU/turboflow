'''
@Time : 2020/12/29 14:58 
@Author : TurboLIU
@File : Basic_Model_tf2.py.py 
@Software: PyCharm
'''

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np
import json
import os




class TurboBase(tf.Module):
    def __init__(self, cfg):
        super(TurboBase, self).__init__()
        self.cfg = cfg
        self.eps = 1e-5
        self.variables_collections = []
        self.weight_l2_loss = []
        self.param_count = 0
        self.nodes = []
        self.keys = []
        self.actv = Activation()
        self.FUSE_BN = False

    def get_onfig(self, cfg):
        self.cfg = cfg

    def get_latest_count(self, path):
        names = os.listdir(path)
        counts = [int(name.split(".")[0].split("-")[-1]) for name in names]
        return max(counts)

    def load_TurboLIU_Params(self, jsonParams):
        f = open(jsonParams, "r")
        Params = json.load(f)
        f.close()
        self.TurboParams = self.process_params(Params)
        self.cfg.TurboParams = self.TurboParams
        self.keys = self.TurboParams.keys()

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

    def load_param(self, name, l2_loss=False):
        v =tf.Variable(initial_value=self.TurboParams[name]["value"], trainable=True, name=name)
        if l2_loss:
            self.weight_l2_loss.append((1-self.cfg.weight_decay)*tf.nn.l2_loss(v))
        return v

    def GetVariable(self, shape, name, trainable=True, weight_decay=0.995, l2_loss=False):
        if name in self.keys:
            v = tf.Variable(initial_value=self.TurboParams[name]["value"], trainable=trainable, name=name)
        else:
            v = tf.Variable(initial_value=tf.random.normal(shape=shape, mean=0.0, stddev=0.1, dtype=tf.float32),
                               trainable=trainable, name=name)
        if l2_loss:
            self.weight_l2_loss.append((1-weight_decay)*tf.nn.l2_loss(v))
        return v

    def get_weight_l2_loss(self):
        for node in self.nodes:
            if isinstance(node, ConvBlock_):
                self.weight_l2_loss.append((1-self.cfg.weight_decay) * tf.nn.l2_loss(node.w))

    def get_variables(self):
        return self.variables

    def ConvBlock(self, in_channels, out_channels, kernel_size, stride=1, padding=True, name="ConvBlock",
                  BN=True, use_bias=True, act_type="relu",):
        cb = ConvBlock_(self, in_channels, out_channels, kernel_size, stride=stride, padding=padding, name=name,
                  BN=BN, use_bias=use_bias, act_type=act_type)
        self.nodes.append(cb)
        return cb

    def DepthWiseConv(self, in_channels, channel_multiplier, kernel_size, stride=1, padding=True, name="DepthWise_Conv",
                       use_bias=True, BN=True, act_type="relu"):
        db = DepthWiseConv_(self, in_channels=in_channels, channel_multiplier=channel_multiplier,
                            kernel_size=kernel_size, stride=stride, padding=padding, name=name,
                            use_bias=use_bias, BN=BN, act_type=act_type)
        self.nodes.append(db)
        return db

    def DeConvBlock(self, in_channels, out_channels, kernel_size, stride=1, padding=True, scale=2, name="DeConvBlock",
                    BN=True, use_bias=True, act_type="relu"):
        dc = DeConvBlock_(self, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, scale=scale, name=name,
                         BN=BN, use_bias=use_bias, act_type=act_type)
        self.nodes.append(dc)
        return dc

    def ResBlock(self, **kwargs):
        pass

    def BottleNeck(self, **kwargs):
        pass

    def ExpBlock(self, **kwargs):
        pass

    def fuse_w(self, w, gama, variance, esp):
        channel = w.shape[3]
        for i in range(channel):
            w[..., i] = w[..., i] * gama[i] / np.sqrt(variance[i] + esp)
        return w.copy()

    def fuse_b(self, b, gama, beta, mean, variance, esp):
        ##  test beta
        # beta1 = np.zeros_like(beta)
        return gama * (b - mean) / np.sqrt(variance + esp) + beta

    def depwise_fuse_w(self, w, gama, variance, esp):
        channel = w.shape[2]
        for i in range(channel):
            w[:, :, i, :] = w[:, :, i, :] * gama[i] / np.sqrt(variance[i] + esp)
        return w.copy()

    def depwise_fuse_b(self, b, gama, beta, mean, variance, esp):
        return self.fuse_b(b, gama, beta, mean, variance, esp)  # 目测和conv的计算方式一样

    def get_lr_WarmUpValue(self, step, stride):
        init_lr = self.cfg.learning_rate
        end = self.cfg.lr_steps[0]
        total_step = end*stride
        lr = step*init_lr/total_step + 1e-7
        return lr

    def get_lr_CosValue(self, ep, epoch):
        lr = 0.5 * (1 + np.cos(ep * np.pi / epoch)) * self.cfg.learning_rate
        return lr

    def get_lr_value(self, ep):
        ex = 0
        for i in self.cfg.lr_steps:
            if ep > i:
                ex += 1
        lr = self.cfg.learning_rate * (self.cfg.lr_gama ** ex)
        return lr

    def calc_l2_loss(self, weight, weight_decay):
        _, _, _, outchannel = weight.get_shape().as_list()
        return (weight_decay) * tf.reduce_sum(tf.square(weight)) / outchannel

    def calc_l1_loss(self, weight, weight_decay):
        _, _, _, outchannel = weight.get_shape().as_list()
        return (weight_decay) * tf.reduce_sum(tf.abs(weight)) / outchannel

    def save_pb_with_fuseBN(self, net, outnodes, dstPath, dstname):
        # saver.save(sess, "./centerface_onnx")
        net_func = tf.function(lambda x:net(x))
        net_func = net_func.get_concrete_function(x=tf.TensorSpec(net.inputs[0].shape, net.inputs[0].dtype))
        frozen_func = convert_variables_to_constants_v2(net_func)
        frozen_func.graph.as_graph_def()
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                          logdir=dstPath,
                          name=dstname,
                          as_text=False)


    def pb2onnx(self, pbmodel, inputs, outputs, onnxmodel):
        '''
        python - m tf2onnx.convert \
        --input pbmodel \
        --inputs Placeholder:0 \
        --outputs Sigmoid:0, Sigmoid_1:0, Exp:0 \
        --output ./centerhand_288x288f3.onnx
        '''
        pass

    def save(self, name):
        weights = dict()
        for var in self.variables:
            weight = dict()
            weight["value"] = var.numpy().tolist()
            weight["shape"] = list(var.shape)
            weight["dtype"] = str(var.dtype)
            weights[var.name] = weight
        f = open(name, "w")
        file = json.dumps(weights)
        f.write(file)
        f.close()

    def forward(self, x):
        pass

class Activation(tf.Module):
    def __init__(self):
        self.activation = {}
        self.activation["relu"] = self.relu
        self.activation["relu6"] = self.relu6
        self.activation["lrelu"] = self.lrelu
        self.activation["prelu"] = self.prelu
        self.activation["tanh"] = self.tanh
        self.activation["selu"] = self.selu
        self.activation["hswish"] = self.hswish
        self.activation["hsigmoid"] = self.hsigmoid
        self.activation["sigmoid"] = self.sigmoid
        self.activation["linear"] = self.linear

    def check_type(self, act_type):
        if act_type.lower() not in self.activation.keys():
            raise NotImplementedError
    def relu(self, x, **kwargs):
        return tf.nn.relu(x)
    def lrelu(self, x, **kwargs):
        slope = kwargs["slope"]
        y = slope * x
        return tf.maximum(x, y)
    def relu6(self, x, **kwargs):
        return tf.nn.relu6(x)
    def prelu(self, x, **kwargs):
        alpha = kwargs["alpha"]
        return tf.nn.leaky_relu(x, alpha)
    def tanh(self,x,**kwargs):
        return tf.nn.tanh(x)
    def hswish(self, x, **kwargs):
        return x * self.hsigmoid(x, **kwargs)
    def hsigmoid(self, x, **kwargs):
        return tf.nn.relu6((x+3)/6)
    def sigmoid(self, x, **kwargs):
        return tf.nn.sigmoid(x)
    def linear(self, x, **kwargs):
        return x
    def selu(self, x, **kwargs):
        '''
        权重必须服从（0，1）的正态分布  W ~~ N(0,1)
        '''
        with tf.name_scope('elu') as scope:
            alpha = 1.6732632423543772848170429916717
            scale = 1.0507009873554804934193349852946
            return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))
    def __call__(self,x, act_type, **kwargs):
        return self.activation[act_type.lower()](x, **kwargs)

class BatchNorm(tf.Module):
    def __init__(self, cls:object, depth, decay=0.995, name="BN"):
        super(BatchNorm, self).__init__()
        self._name = name
        if cls.cfg.load_premodel:
            self.gama = cls.load_param(name="%s_gama" % name)
            self.beta = cls.load_param(name="%s_beta"%name)
            self.pop_mean = cls.load_param(name="%s_pop_mean" % name)
            self.pop_variance = cls.load_param(name="%s_pop_variance" % name)
        else:
            self.gama = tf.Variable(initial_value=tf.ones(shape=[depth]), name="%s_gama" % name, trainable=True)
            self.beta = tf.Variable(initial_value=tf.zeros(shape=[depth]), name="%s_beta" % name, trainable=True)
            self.pop_mean = tf.Variable(initial_value=tf.zeros(shape=[depth]), name="%s_pop_mean" % name, trainable=False)
            self.pop_variance = tf.Variable(initial_value=tf.ones(shape=[depth]), name="%s_pop_variance" % name, trainable=False)
        self.decay = decay
        self.eps = 1e-5
        # self._add_to_variable_collection()
        cls.param_count += depth*4

    def __call__(self, x):
        average_mean, average_varance = tf.nn.moments(x, axes=[0, 1, 2], keepdims=False)
        self.pop_mean.assign(self.pop_mean * self.decay + average_mean * (1-self.decay))
        self.pop_variance.assign(self.pop_variance * self.decay + average_varance * (1-self.decay))
        return tf.nn.batch_normalization(x, average_mean, average_varance, self.beta, self.gama, self.eps)

    def _add_to_variable_collection(self):
        self.variables_collections.append(self.gama)
        self.variables_collections.append(self.beta)
        self.variables_collections.append(self.pop_mean)
        self.variables_collections.append(self.pop_variance)

class InstanceNorm(tf.Module):
    def __init__(self, cls:TurboBase, depth, decay=0.995, name="IN"):
        super(InstanceNorm, self).__init__()
        self._name = name
        if cls.cfg.load_premodel:
            self.gama = cls.load_param(name="%s_gama" % name)
            self.beta = cls.load_param(name="%s_beta"%name)
            self.pop_mean = cls.load_param(name="%s_pop_mean" % name)
            self.pop_variance = cls.load_param(name="%s_pop_variance" % name)
        else:
            self.gama = tf.Variable(initial_value=tf.ones(shape=[cls.cfg.batchsize, 1, 1, depth]), name="%s_gama" % name, trainable=True)
            self.beta = tf.Variable(initial_value=tf.zeros(shape=[cls.cfg.batchsize, 1, 1, depth]), name="%s_beta" % name, trainable=True)
            self.pop_mean = tf.Variable(initial_value=tf.zeros(shape=[cls.cfg.batchsize, 1, 1, depth]), name="%s_pop_mean" % name, trainable=False)
            self.pop_variance = tf.Variable(initial_value=tf.ones(shape=[cls.cfg.batchsize, 1, 1, depth]), name="%s_pop_variance" % name, trainable=False)
        self.decay = decay
        self.eps = 1e-5
        # self._add_to_variable_collection()
        cls.param_count += cls.cfg.batchsize*depth*4

    def __call__(self, x):
        average_mean, average_varance = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        self.pop_mean.assign(self.pop_mean * self.decay + average_mean * (1-self.decay))
        self.pop_variance.assign(self.pop_variance * self.decay + average_varance * (1-self.decay))
        return tf.nn.batch_normalization(x, average_mean, average_varance, self.beta, self.gama, self.eps)

    def _add_to_variable_collection(self):
        self.variables_collections.append(self.gama)
        self.variables_collections.append(self.beta)
        self.variables_collections.append(self.pop_mean)
        self.variables_collections.append(self.pop_variance)


class ConvBlock_(tf.Module):
    def __init__(self, cls:object, in_channels, out_channels, kernel_size, stride=1, padding=True, name="ConvBlock",
                  BN=True, use_bias=True, act_type="relu"):
        super(ConvBlock_, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self._name = name
        self.use_bias = use_bias
        self.FUSE_BN = cls.FUSE_BN
        self.act_type = act_type
        # self.istrain = istrain
        self.actv = cls.actv
        self.actv.check_type(act_type)
        self.eps = 1e-5

        if (not cls.cfg.istrain) and cls.FUSE_BN and BN:
            gama_name, beta_name, mean_name, var_name = cls.get_batchnorm_name("%s_BN"%name)
            w_name = "%s_w"%name
            weight = cls.TurboParams[w_name]["value"]
            gama = cls.TurboParams[gama_name]["value"]
            beta = cls.TurboParams[beta_name]["value"]
            mean = cls.TurboParams[mean_name]["value"]
            # print(mean)
            variance = cls.TurboParams[var_name]["value"]
            fused_weight = cls.fuse_w(weight, gama, variance, esp=self.eps)
            self.w = tf.Variable(initial_value=fused_weight, name="%s_w"%name)
            bias = np.zeros(shape=cls.TurboParams[w_name]["shape"][3], dtype=np.float32)
            fused_bias = cls.fuse_b(bias, gama, beta, mean, variance, esp=self.eps)
            if use_bias:
                bias_name = "%s_b" % name
                bias_ = cls.TurboParams[bias_name]["value"]
                fused_bias += bias_
            self.b = tf.Variable(initial_value=fused_bias, name="%s_b" % name)
            self.BN=None
        else:
            if BN:
                if cls.cfg.BN_type == "BN":
                    self.BN = BatchNorm(cls, out_channels, name="%s_BN" % name)
                elif cls.cfg.BN_type == "IN":
                    self.BN = InstanceNorm(cls, out_channels, name="%s_IN" % name)
                else:
                    raise ValueError("BN_type: %s not recognised!" % BN)
            else:
                self.BN = None
            self.w = cls.GetVariable(shape=[kernel_size, kernel_size, in_channels, out_channels], name="%s_w" % name, l2_loss=True)
            if use_bias:
                self.b = cls.GetVariable(shape=[out_channels], l2_loss=False, name="%s_b"%name)
                cls.param_count += out_channels

            cls.param_count = kernel_size*kernel_size*in_channels*out_channels
        # self.calc_count = 0
        # self._add_to_variable_collection()

    def __call__(self, x):
        if self.padding:
            pdsz = self.kernel_size//2
            x = tf.pad(x, [[0, 0], [pdsz, pdsz], [pdsz, pdsz], [0, 0]], name="pad")
        x = tf.nn.conv2d(x, self.w, strides=[1, self.stride, self.stride, 1], padding="VALID", name=self.name)
        if self.BN:
            x = self.BN(x)
        if self.use_bias or self.FUSE_BN:
            x = tf.nn.bias_add(x, self.b)
        x = self.actv(x, act_type=self.act_type)
        return x

    def _add_to_variable_collection(self):
        self.variables_collections.append(self.w)
        if self.use_bias or self.FUSE_BN:
            self.variables_collections.append(self.b)


class DepthWiseConv_(tf.Module):
    def __init__(self, cls:object, in_channels, channel_multiplier, kernel_size, stride=1, padding=True, name="DepthWise_Conv",
                       use_bias=True, BN=True, act_type="relu"):
        super(DepthWiseConv_, self).__init__()
        self._name = name
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.channel_multiplier = channel_multiplier
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.FUSE_BN = cls.FUSE_BN
        self.actv = cls.actv
        self.actv.check_type(act_type)
        self.act_type = act_type
        self.eps = 1e-5

        if (not cls.cfg.istrain) and cls.FUSE_BN and BN:
            gama_name, beta_name, mean_name, var_name = cls.get_batchnorm_name("%s_BN" % name)
            w_name = "%s_w" % name
            weight = cls.TurboParams[w_name]["value"]
            gama = cls.TurboParams[gama_name]["value"]
            beta = cls.TurboParams[beta_name]["value"]
            mean = cls.TurboParams[mean_name]["value"]
            # print(mean)
            variance = cls.TurboParams[var_name]["value"]
            fused_weight = cls.depwise_fuse_w(weight, gama, variance, esp=self.eps)
            self.w = tf.Variable(initial_value=fused_weight, name="%s_w" % name)
            bias = np.zeros(shape=cls.TurboParams[w_name]["shape"][2], dtype=np.float32)
            fused_bias = cls.depwise_fuse_b(bias, gama, beta, mean, variance, esp=self.eps)
            if use_bias:
                bias_name = "%s_b" % name
                bias_ = cls.TurboParams[bias_name]["value"]
                fused_bias += bias_
            self.b = tf.Variable(initial_value=fused_bias, name="%s_b" % name)
            self.BN = None
        else:
            if BN:
                if cls.cfg.BN_type == "BN":
                    self.BN = BatchNorm(cls, in_channels * channel_multiplier, name="%s_BN" % name)
                elif cls.cfg.BN_type == "IN":
                    self.BN = InstanceNorm(cls, in_channels * channel_multiplier, name="%s_IN" % name)
                else:
                    raise ValueError("BN_type: %s not recognised!" % self.cfg.BN_type)
            else:
                self.BN = None
            self.w = cls.GetVariable(shape=[kernel_size, kernel_size, in_channels, channel_multiplier],
                                     name="%s_w" % name, l2_loss=True)
            if use_bias:
                self.b = cls.GetVariable(shape=[in_channels * channel_multiplier], l2_loss=False, name="%s_b"%name)
                cls.param_count += in_channels * channel_multiplier

            cls.param_count = kernel_size*kernel_size*in_channels*channel_multiplier

        self.calc_count = 0
        # self._add_to_variable_collection()

    def __call__(self, x):
        if self.padding:
            pdsz = self.kernel_size//2
            x = tf.pad(x, [[0, 0], [pdsz, pdsz], [pdsz, pdsz], [0, 0]], name="pad")
        x = tf.nn.depthwise_conv2d(x, filter=self.w, strides=[1, self.stride, self.stride, 1], padding="VALID")
        if self.BN:
            x = self.BN(x)
        if self.use_bias or self.FUSE_BN:
            x = tf.nn.bias_add(x, self.b)
        x = self.actv(x, act_type=self.act_type)
        return x

    def _add_to_variable_collection(self):
        self.variables_collections.append(self.w)
        if self.use_bias or self.FUSE_BN:
            self.variables_collections.append(self.b)

class DeConvBlock_(tf.Module):
    def __init__(self, cls:object, in_channels, out_channels, kernel_size, stride=1, padding=True, scale=2, name="DeConvBlock",
                    BN=True, use_bias=True, act_type="relu"):
        super(DeConvBlock_, self).__init__()
        self._name = name
        self.kernel_size = kernel_size
        self.in_channels = out_channels
        self.out_channels = in_channels
        self.scale = scale
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.FUSE_BN = cls.FUSE_BN
        self.actv = cls.actv
        self.actv.check_type(act_type)
        self.act_type = act_type
        self.eps = 1e-5

        if (not cls.cfg.istrain) and cls.FUSE_BN and BN:
            gama_name, beta_name, mean_name, var_name = cls.get_batchnorm_name("%s_BN" % name)
            w_name = "%s_w" % name
            weight = cls.TurboParams[w_name]["value"]
            gama = cls.TurboParams[gama_name]["value"]
            beta = cls.TurboParams[beta_name]["value"]
            mean = cls.TurboParams[mean_name]["value"]
            # print(mean)
            variance = cls.TurboParams[var_name]["value"]
            fused_weight = cls.depwise_fuse_w(weight, gama, variance, esp=self.eps)
            self.w = tf.Variable(initial_value=fused_weight, name="%s_w" % name)
            bias = np.zeros(shape=cls.TurboParams[w_name]["shape"][3], dtype=np.float32)
            fused_bias = cls.depwise_fuse_b(bias, gama, beta, mean, variance, esp=self.eps)
            if use_bias:
                bias_name = "%s_b" % name
                bias_ = cls.TurboParams[bias_name]["value"]
                fused_bias += bias_
            self.b = tf.Variable(initial_value=fused_bias, name="%s_b" % name)
            self.BN = None
        else:
            if BN:
                if cls.cfg.BN_type == "BN":
                    self.BN = BatchNorm(cls, self.out_channels, name="%s_BN" % name)
                elif cls.cfg.BN_type == "IN":
                    self.BN = InstanceNorm(cls, self.out_channels, name="%s_IN" % name)
                else:
                    raise ValueError("BN_type: %s not recognised!" % self.cfg.BN_type)
            else:
                self.BN = None
            self.w = cls.GetVariable(shape=[kernel_size, kernel_size, self.in_channels, self.out_channels],
                                      name="%s_w" % name, l2_loss=True)
            if use_bias:
                self.b = cls.GetVariable(shape=[self.out_channels], name="%s_b"%name)
                cls.param_count += in_channels

            cls.param_count = kernel_size*kernel_size*in_channels*out_channels

        self.calc_count = 0
        # self._add_to_variable_collection()

    def __call__(self, x):
        b, h, w, c = x.shape
        out_shape = [b, h * self.scale, w * self.scale, c]
        if self.padding:
            pdsz = self.kernel_size//2
            x = tf.pad(x, [[0,0],[pdsz, pdsz],[pdsz, pdsz],[0,0]], name="pad")
        x = tf.nn.conv2d_transpose(x, filters=self.w, output_shape=out_shape, strides=[1, self.stride, self.stride, 1], padding="VALID")
        if self.BN:
            x = self.BN(x)
        if self.use_bias or self.FUSE_BN:
            x = tf.nn.bias_add(x, self.b)
        x = self.actv(x, act_type=self.act_type)
        return x

    def _add_to_variable_collection(self):
        self.variables_collections.append(self.w)
        if self.use_bias or self.FUSE_BN:
            self.variables_collections.append(self.b)

if __name__ == '__main__':
    from config import config_net
    cfg = config_net()
    x = tf.ones([1,15,15,3])
    t = TurboBase(cfg)
    f1 = t.ConvBlock_(3, 5, 3, 1, BN=True, name="testconv1")
    f2 = t.ConvBlock_(5, 10, 3, 1, BN=True, name="testconv2")
    f2(f1(x))
    pass

