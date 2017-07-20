import os
import math
import torch


class NetOption(object):
    def __init__(self):
        #  ------------ General options ----------------------------------------
        self.save_path = ""  # log path
        # self.data_path = "/home/dataset/imagenet/"  # path for loading data set
        self.data_set = "mnist"  # options: cifar10 | cifar100 | mnist
        self.manualSeed = 1  # manually set RNG seed
        self.nGPU = 1  # number of GPUs to use by default
        self.GPU = 0  # default gpu to use, options: range(nGPU)

        # ------------- Data options -------------------------------------------
        self.nThreads = 4  # number of data loader threads

        # ------------- Training options ---------------------------------------
        self.testOnly = False  # run on validation set only
        self.tenCrop = False  # Ten-crop testing

        # ---------- Optimization options --------------------------------------
        self.useDefaultSetting = False
        self.nEpochs = 10  # number of total epochs to train
        self.trainBatchSize = 50  # mini-batch size for training
        self.testBatchSize = 1000  # mini-batch size for testing

        self.LR = 0.01  # initial learning rate
        self.lrPolicy = "multistep"  # options: multistep | linear | exp | fixed
        self.momentum = 0.9  # momentum
        self.weightDecay = 0  # weight decay 1e-4
        self.gamma = 0.94  # gamma for learning rate policy (step)
        self.step = 2.0  # step for linear or exp learning rate policy
        self.ratio = [0.6, 0.8] # learning rate decay for multi-step lrPolicy

        # ---------- Model options ---------------------------------------------
        self.netType = "LeNet5"  # options: ResNet | PreResNet | LeNet5
        self.experimentID = "epochs10-bs50-lr0.01-momentum0.9-weightdecay0.0001-01"
        self.depth = 20  # resnet depth: (n-2)%6==0
        self.shortcutType = "B"

        # ---------- Resume or Retrain options ---------------------------------------------
        self.retrain = None  # path to model to retrain with
        self.resume = None  # path to directory containing checkpoint
        self.resumeEpoch = 0  # manual epoch number for resume

        # ---------- Memory Reduction options ----------------------------------
        self.nClasses = 10  # number of classes in the dataset
        self.wideFactor = 1  # wide factor for wide-resnet

        # ---------- Visualization options -------------------------------------
        self.drawNetwork = False
        self.onlineBoard = False
        self.drawInterval = 30

        # check parameters
        self.paramscheck()

    def paramscheck(self):
        torch_version = torch.__version__
        torch_version_split = torch_version.split("_")

        if torch_version_split[0] != "0.1.10":
            self.drawNetwork = False
            print "|===>DrawNetwork is unsupported by PyTorch with version: ", torch_version

        if self.netType == "LeNet":
            self.save_path = "log_%s_%s_%s/" % (self.netType, self.data_set, self.experimentID)
        else:
            self.save_path = "log_%s_%s_%d_%s/" % (self.netType, self.data_set,
                                                   self.depth, self.experimentID)

        if self.useDefaultSetting:
            print("|===> Use Default Setting")
            if self.data_set == "cifar10" or self.data_set == "cifar100":
                if self.nEpochs == 160:
                    self.LR = 0.5
                    self.lrPolicy = "exp"
                    self.momentum = 0.9
                    self.weightDecay = 1e-4
                    self.step = 2.0
                    self.gamma = math.pow(0.001 / self.LR, 1.0/math.floor(self.nEpochs/self.step))
                else:
                    self.LR = 0.1
                    self.lrPolicy = "multistep"
                    self.momentum = 0.9
                    self.weightDecay = 1e-4
            else:
                assert False, "invalid data set"

        if self.data_set == "cifar10" or self.data_set == "mnist":
            self.nClasses = 10
        elif self.data_set == "cifar100":
            self.nClasses = 100
