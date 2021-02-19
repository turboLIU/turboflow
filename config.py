import os

class config:
    def __init__(self):
        # queue config
        self.Process_num = 1
        self.maxsize = 500
        self.ngpu = 1
        self.epoch = 1000

        # self.checkpoint_dir = "./jmodel"
        # if not os.path.exists(self.checkpoint_dir):
        #     os.mkdir(self.checkpoint_dir)
        # self.log_dir = "./log"
        # if not os.path.exists(self.log_dir):
        #     os.mkdir(self.log_dir)
        # self.result = "./result"
        # if not os.path.exists(self.result):
        #     os.mkdir(self.result)



class config_net(config):
    def __init__(self):
        super(config_net, self).__init__()
        self.dsize = 640
        self.downScale = 4
        self.minSize = 0# 像素个数
        self.c_dim = 3
        self.clsNum = 1
        self.channelSize = 3
        self.heat_weight = 1
        self.off_weight = 1
        self.size_weight = 0.1
        self.landmark_weight = 0
        self.landmarknum = 98
        self.heatmap = "CenterNet" # Turbo, CenterNet


        self.subBatchSize = 1
        self.batchsize = self.subBatchSize * self.ngpu
        self.istrain = True
        self.istest = not self.istrain
        self.BN = True
        self.BN_type = "BN" # "BN" # or "IN"
        self.act_type = "relu"
        self.loss_type = "L2"
        self.lr_gama = 0.1
        self.learning_rate = 1e-2
        self.lr_steps = [ 7, 20, 50, 100, 200, 400, 600, 800]
        # self.lr_values = [0.001, self.learning_rate, ]

        self.load_premodel = False
        self.counter = 0
        self.weight_decay = 0.995
        self.mean = 0.
        self.variance = 255.0

        # self.model = "CenterNet"  # "mobileSmall" or "mobileLarge"
        # self.model_logdir = "%s/%s" % (self.log_dir, self.model)
        # if not os.path.exists(self.model_logdir):
        #     os.mkdir(self.model_logdir)
        # self.model_result = "%s/%s" % (self.result, self.model)
        # if not os.path.exists(self.model_result):
        #     os.mkdir(self.model_result)

        # augment params
        self.perspective = 0.5
        self.cutout = 0.5
        self.blur = 0.1
        self.noise = 0.1
        self.rotate = 0.5



