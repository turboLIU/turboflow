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

def pb2jsonParams(pbfile, paramfile):
    sess = tf.compat.v1.Session()
    graph = tf.compat.v1.GraphDef()
    with open(pbfile, "rb") as f:
        graph.ParseFromString(f.read())
        tf.import_graph_def(graph, name="")
    print("...")

class TurboBase(tf.Module):
    def __init__(self, cfg):
        super(TurboBase, self).__init__()
        self.cfg = cfg
        self.eps = 1e-5
        self.variables_collections = []

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

    def load_TurboParams(self):
        self.TurboParams = {}
        self.keys = self.TurboParams.keys()
        pass

    def GetVariable(self, shape, name, trainable=True):
        if name in self.keys:
            return tf.Variable(initial_value=self.TurboParams[name]["value"], trainable=trainable, name=name)
        else:
            return tf.Variable(initial_value=tf.random.normal(shape=shape, mean=0.0, stddev=0.1, dtype=tf.float32),
                               trainable=trainable, name=name)

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

    def save_pb_with_fuseBN(self, net, outnodes, dstPath, dstname):
        # saver.save(sess, "./centerface_onnx")
        net_func = tf.function(lambda x:net(x))
        net_func = net_func.get_concrete_function(x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
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
        for var in self.variables_collections:
            weight = dict()
            weight["value"] = var.numpy().tolist()
            weight["shape"] = list(var.shape)
            weight["dtype"] = str(var.dtype)
            weights[var.name] = weight
        f = open(name, "w")
        file = json.dumps(weights)
        f.write(file)
        f.close()

class BatchNorm(TurboBase, tf.Module):
    def __init__(self, depth, decay=0.995, name="BN"):
        super(BatchNorm, self).__init__(name=name)
        self.gama = self.GetVariable([depth], "%s_gama" % name, trainable=True)
        self.beta = self.GetVariable([depth], "%s_beta" % name, trainable=True)
        self.pop_mean = self.GetVariable([depth], "%s_pop_mean" % name, trainable=False)
        self.pop_variance = self.GetVariable([depth], "%s_pop_variance" % name, trainable=False)
        self.decay = decay
        self._add_to_variable_collection()

    def __call__(self, x):
        average_mean, average_varance = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=False)
        self.pop_mean.assign(self.pop_mean * self.decay + average_mean * (1-self.decay))
        self.pop_variance.assign(self.pop_variance * self.decay + average_varance * (1-self.decay))
        return tf.nn.batch_normalization(x, average_mean, average_varance, self.beta, self.gama, self.eps)

    def _add_to_variable_collection(self):
        self.variables_collections.append(self.gama)
        self.variables_collections.append(self.beta)
        self.variables_collections.append(self.pop_mean)
        self.variables_collections.append(self.pop_variance)

class InstanceNorm(TurboBase, tf.Module):
    def __init__(self, depth, decay=0.995, name="IN"):
        super(InstanceNorm, self).__init__(name=name)
        self.gama = self.GetVariable([self.cfg.batchsize, 1, 1, depth], "%s_gama" % name, trainable=True)
        self.beta = self.GetVariable([self.cfg.batchsize, 1, 1, depth], "%s_beta" % name, trainable=True)
        self.pop_mean = self.GetVariable([self.cfg.batchsize, 1, 1, depth], "%s_pop_mean" % name, trainable=False)
        self.pop_variance = self.GetVariable([self.cfg.batchsize, 1, 1, depth], "%s_pop_variance" % name, trainable=False)
        self.decay = decay
        self._add_to_variable_collection()

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

class ConvBlock(TurboBase, tf.Module):
    def __init__(self, in_channels, out_channels, kernel_size, name="ConvBlock",
                  BN=True, FUSE_BN=True, use_bias=True):
        super(ConvBlock, self).__init__(name=name)
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.name = name
        self.use_bias = use_bias
        self.FUSE_BN = FUSE_BN

        if (not self.cfg.istrain) and self.FUSE_BN and BN:
            gama_name, beta_name, mean_name, var_name = self.get_batchnorm_name("%s_BN"%name)
            w_name = "%s_w"%name
            weight = self.TurboParams[w_name]["value"]
            gama = self.TurboParams[gama_name]["value"]
            beta = self.TurboParams[beta_name]["value"]
            mean = self.TurboParams[mean_name]["value"]
            # print(mean)
            variance = self.TurboParams[var_name]["value"]
            fused_weight = self.fuse_w(weight, gama, variance, esp=self.eps)
            self.w = tf.Variable(initial_value=fused_weight, name="%s_w"%name)
            bias = np.zeros(shape=self.TurboParams[w_name]["shape"][3], dtype=np.float32)
            fused_bias = self.fuse_b(bias, gama, beta, mean, variance, esp=self.eps)
            if use_bias:
                bias_name = "%s_b" % name
                bias_ = self.TurboParams[bias_name]["value"]
                fused_bias += bias_
            self.b = tf.Variable(initial_value=fused_bias, name="%s_b" % name)
            self.BN=None
        else:
            if BN:
                if self.cfg.BN_type == "BN":
                    self.BN = BatchNorm(out_channels, name="%s_BN" % name)
                elif self.cfg.BN_type == "IN":
                    self.BN = InstanceNorm(out_channels, name="%s_IN" % name)
                else:
                    raise ValueError("BN_type: %s not recognised!" % self.cfg.BN_type)
            else:
                self.BN = None
            self.w = self.GetVariable(shape=[kernel_size, kernel_size, in_channels, out_channels], name="%s_w" % name)
            if use_bias:
                self.b = self.GetVariable(shape=[out_channels], name="%s_b"%name)
                self.param_count += out_channels

            self.param_count = kernel_size*kernel_size*in_channels*out_channels
        self.calc_count = 0
        self._add_to_variable_collection()

    def __call__(self, x, stride=1, padding=True, BN=True, act_type="relu"):
        if padding:
            pdsz = self.kernel_size//2
            x = tf.pad(x, [[0, 0], [pdsz, pdsz], [pdsz, pdsz], [0, 0]], name="pad")
        x = tf.nn.conv2d(x, self.w, strides=[1, stride, stride, 1], padding="VALID", name=self.name)
        if self.BN:
            x = self.BN(x)
        if self.use_bias or self.FUSE_BN:
            x = tf.nn.bias_add(x, self.b)
        x = self.activation(x, act_type=act_type)
        return x

    def _add_to_variable_collection(self):
        self.variables_collections.append(self.w)
        if self.use_bias or self.FUSE_BN:
            self.variables_collections.append(self.b)

class DepthWise_Conv(TurboBase, tf.Module):
    def __init__(self, in_channels, channel_multiplier, kernel_size, name="DepthWise_Conv",
                       use_bias=True, BN=True, FUSE_BN=False):
        super(DepthWise_Conv, self).__init__(name=name)
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.channel_multiplier = channel_multiplier
        self.use_bias = use_bias
        self.FUSE_BN = FUSE_BN

        if (not self.cfg.istrain) and self.FUSE_BN and BN:
            gama_name, beta_name, mean_name, var_name = self.get_batchnorm_name("%s_BN" % name)
            w_name = "%s_w" % name
            weight = self.TurboParams[w_name]["value"]
            gama = self.TurboParams[gama_name]["value"]
            beta = self.TurboParams[beta_name]["value"]
            mean = self.TurboParams[mean_name]["value"]
            # print(mean)
            variance = self.TurboParams[var_name]["value"]
            fused_weight = self.depwise_fuse_w(weight, gama, variance, esp=self.eps)
            self.w = tf.Variable(initial_value=fused_weight, name="%s_w" % name)
            bias = np.zeros(shape=self.TurboParams[w_name]["shape"][2], dtype=np.float32)
            fused_bias = self.depwise_fuse_b(bias, gama, beta, mean, variance, esp=self.eps)
            if use_bias:
                bias_name = "%s_b" % name
                bias_ = self.TurboParams[bias_name]["value"]
                fused_bias += bias_
            self.b = tf.Variable(initial_value=fused_bias, name="%s_b" % name)
            self.BN = None
        else:
            if BN:
                if self.cfg.BN_type == "BN":
                    self.BN = BatchNorm(channel_multiplier, name="%s_BN" % name)
                elif self.cfg.BN_type == "IN":
                    self.BN = InstanceNorm(channel_multiplier, name="%s_IN" % name)
                else:
                    raise ValueError("BN_type: %s not recognised!" % self.cfg.BN_type)
            else:
                self.BN = None
            self.w = self.GetVariable(shape=[kernel_size, kernel_size, in_channels, channel_multiplier], name="%s_w" % name)
            if use_bias:
                self.b = self.GetVariable(shape=[channel_multiplier], name="%s_b"%name)
                self.param_count += channel_multiplier

            self.param_count = kernel_size*kernel_size*in_channels*channel_multiplier

        self.calc_count = 0
        self._add_to_variable_collection()

    def __call__(self, x, stride=1, padding=True, act_type="relu"):
        if padding:
            pdsz = self.kernel_size//2
            x = tf.pad(x, [[0, 0], [pdsz, pdsz], [pdsz, pdsz], [0, 0]], name="pad")
        x = tf.nn.depthwise_conv2d(x, filter=self.w, strides=[1, stride, stride, 1], padding="VALID")
        if self.BN:
            x = self.BN(x)
        if self.use_bias or self.FUSE_BN:
            x = tf.nn.bias_add(x, self.b)
        x = self.activation(x, act_type=act_type)
        return x

    def _add_to_variable_collection(self):
        self.variables_collections.append(self.w)
        if self.use_bias or self.FUSE_BN:
            self.variables_collections.append(self.b)

class DeConvBlock(TurboBase, tf.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale=2, name="DeConvBlock",
                    BN=True, use_bias=True, FUSE_BN=False):
        super(DeConvBlock, self).__init__(name=name)
        self.kernel_size = kernel_size
        self.in_channels = out_channels
        self.out_channels = in_channels
        self.scale = scale
        self.use_bias = use_bias
        self.FUSE_BN = FUSE_BN

        if (not self.cfg.istrain) and self.FUSE_BN and BN:
            gama_name, beta_name, mean_name, var_name = self.get_batchnorm_name("%s_BN" % name)
            w_name = "%s_w" % name
            weight = self.TurboParams[w_name]["value"]
            gama = self.TurboParams[gama_name]["value"]
            beta = self.TurboParams[beta_name]["value"]
            mean = self.TurboParams[mean_name]["value"]
            # print(mean)
            variance = self.TurboParams[var_name]["value"]
            fused_weight = self.depwise_fuse_w(weight, gama, variance, esp=self.eps)
            self.w = tf.Variable(initial_value=fused_weight, name="%s_w" % name)
            bias = np.zeros(shape=self.TurboParams[w_name]["shape"][3], dtype=np.float32)
            fused_bias = self.depwise_fuse_b(bias, gama, beta, mean, variance, esp=self.eps)
            if use_bias:
                bias_name = "%s_b" % name
                bias_ = self.TurboParams[bias_name]["value"]
                fused_bias += bias_
            self.b = tf.Variable(initial_value=fused_bias, name="%s_b" % name)
            self.BN = None
        else:
            if BN:
                if self.cfg.BN_type == "BN":
                    self.BN = BatchNorm(self.out_channels, name="%s_BN" % name)
                elif self.cfg.BN_type == "IN":
                    self.BN = InstanceNorm(self.out_channels, name="%s_IN" % name)
                else:
                    raise ValueError("BN_type: %s not recognised!" % self.cfg.BN_type)
            else:
                self.BN = None
            self.w = self.GetVariable(shape=[kernel_size, kernel_size, self.in_channels, self.out_channels], name="%s_w" % name)
            if use_bias:
                self.b = self.GetVariable(shape=[self.out_channels], name="%s_b"%name)
                self.param_count += in_channels

            self.param_count = kernel_size*kernel_size*in_channels*out_channels

        self.calc_count = 0
        self._add_to_variable_collection()

    def __call__(self, x, stride=1, padding=True, act_type="relu"):
        b, h, w, c = x.shape
        out_shape = [b, h * self.scale, w * self.scale, c]
        if padding:
            pdsz = self.kernel_size//2
            x = tf.pad(x, [[0,0],[pdsz, pdsz],[pdsz, pdsz],[0,0]], name="pad")
        x = tf.nn.conv2d_transpose(x, filters=self.w, output_shape=out_shape, strides=[1, stride, stride, 1], padding="VALID")
        if self.BN:
            x = self.BN(x)
        if self.use_bias or self.FUSE_BN:
            x = tf.nn.bias_add(x, self.b)
        x = self.activation(x, act_type=act_type)
        return x

    def _add_to_variable_collection(self):
        self.variables_collections.append(self.w)
        if self.use_bias or self.FUSE_BN:
            self.variables_collections.append(self.b)

if __name__ == '__main__':
    pb = r'E:\Object_Detection\Face_Detection\PFLD-master\PFLD-tiny-bn.pb'
    paramfile = r'E:\Object_Detection\Face_Detection\PFLD-master\PFLD-tiny-bn.json'
    pb2jsonParams(pb, paramfile)
