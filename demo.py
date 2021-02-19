'''
@Time : 2021/2/19 11:27 
@Author : TurboLIU
@File : demo.py.py 
@Software: PyCharm
'''
import os
import tensorflow as tf
from Basic_Model import TurboBASE

class PointsNet(TurboBASE):
    def __init__(self, cfg):
        super(PointsNet, self).__init__(cfg)

        self.parse_cfg(cfg)

        imageshape = [cfg.subBatchSize, cfg.dsize, cfg.dsize, cfg.c_dim]
        labelshape = [cfg.subBatchSize, cfg.clsNum]
        with tf.device("/cpu:0"):
            self.imageplaceholder = tf.compat.v1.placeholder(dtype=tf.float32, shape=imageshape, name="image")
            self.labelplaceholder = tf.compat.v1.placeholder(dtype=tf.float32, shape=labelshape, name="label")
        self.losses = []
        if self.load_premodel:
            model_dir = "demoNet"
            json_param_path = self.cfg.checkpoint_dir + "/" + model_dir
            if cfg.counter:
                count = cfg.counter
            else:
                count = self.get_latest_count(json_param_path)
            json_param_name = json_param_path + "/" + model_dir + "-" + str(count)+".json"
            self.load_TurboLIU_Params(jsonParams=json_param_name)
            self.counter = count
            print("loading model from %d"%self.counter)
        else:
            self.counter = 0

    def parse_cfg(self, cfg=None):
        if cfg is None:
            self.c_dim = 3
            self.istrain = True
            self.modif = False
            self.ptsNum = 4
            self.BN_type = "BN"
            self.weight_decay = 0.995

        else:
            self.c_dim = cfg.c_dim
            self.istrain = cfg.istrain
            self.ptsNum = cfg.ptsNum
            self.BN_type = cfg.BN_type
            self.weight_decay = self.cfg.weight_decay


    def build(self, x):
        x = self.ConvBlock(x, in_channels=self.c_dim, out_channels=16, kernel_size=3, stride=2,
                           BN=True, use_bias=False, padding=True, act_type="relu", name="conv1")
        x = self.ConvBlock(x, in_channels=16, out_channels=32, kernel_size=3, stride=2,
                           BN=True, use_bias=False, padding=True, act_type="relu", name="conv2")
        x = self.BottleNeck(x, out_channels=64, kernel_size=3, stride=2, exp_size=256,
                            padding=True, act_type="relu", shortcut=True, name="bottleneck128_2")

        S1 = self.BottleNeck(x, out_channels=32, kernel_size=3, stride=1, exp_size=2 * 64,
                             padding=True, act_type="relu", shortcut=True, name="S1")
        # self.x = tf.identity(S1)
        S2 = self.ConvBlock(S1, in_channels=32, out_channels=32, kernel_size=3, stride=2,
                            BN=True, use_bias=False, padding=True, act_type="relu", name="S2")
        S3 = self.ConvBlock(S2, in_channels=32, out_channels=128, kernel_size=3, stride=1,
                            BN=True, use_bias=False, padding=False, act_type="relu", name="S3")
        x = self.global_average_pooling(S3, name="GAP")

        feat = self.ConvBlock(x, in_channels=128, out_channels=386, kernel_size=1, stride=1,
                           BN=True, use_bias=True, padding=False, act_type="relu", name="FC")
        cls = self.ConvBlock(feat, in_channels=386, out_channels=2, kernel_size=1, stride=1,
                             BN=False, use_bias=True, padding=False, act_type="sigmoid", name="class")
        self.cls = tf.squeeze(cls, axis=[1, 2], name="cls")

    def train_step(self):
        # tf.get_variable_scope().reuse_variables()
        for i in range(self.cfg.ngpu):
            with tf.device('/gpu:%d' % i):
                # with tf.name_scope("tower_%d" % i):
                self.build(self.imageplaceholder[i * self.cfg.subBatchSize: (i + 1) * self.cfg.subBatchSize])
                l2_regularization_loss = tf.reduce_sum(tf.compat.v1.get_collection("weights_l2_loss"))
                self.clsloss = - self.labelplaceholder[:, 0:2] * tf.compat.v1.log(self.cls + 1e-5) \
                               - (1 - self.labelplaceholder[:, 0:2]) * tf.compat.v1.log(1 - self.cls + 1e-5)
                self.clsloss = tf.reduce_sum(self.clsloss) / self.cfg.batchsize
                self.losses.append(self.clsloss + l2_regularization_loss)
                tf.compat.v1.get_variable_scope().reuse_variables()
        tf.compat.v1.summary.scalar('loss/clsloss', self.clsloss)
        tf.summary.scalar("loss/l2_loss", l2_regularization_loss)
        self.merged_summary = tf.compat.v1.summary.merge_all()
        self.avg_loss = tf.reduce_mean(self.losses)
        print("This Net has Params num is %f MB" % (self.params_count * 4 / 1024 / 1024))  # float32

    def test(self):
        self.cfg.batchsize = 1
        self.istrain = False
        self.FUSE_BN = True
        testshape = [self.cfg.batchsize, self.cfg.dsize, self.cfg.dsize, self.cfg.c_dim]
        self.imageplaceholder = tf.compat.v1.placeholder(dtype=tf.float32, shape=testshape, name="input")
        self.build(self.imageplaceholder)
        return self.cls

    def save_json_model(self, sess, save_path, count):
        model_dir = "demoNet"
        if not os.path.exists(os.path.join(save_path, model_dir)):
            os.mkdir(os.path.join(save_path, model_dir))
        model_name = model_dir + "-" + str(count) + ".json" # name-count.json
        self.save(sess, os.path.join(save_path, model_dir, model_name))
        print("%s is saved"%model_name)