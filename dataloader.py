import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import transform as auxtransform
import os


class DataLoader(object):
    def __init__(self, dataset, train_batch_size, test_batch_size, n_threads=4, ten_crop=False):
        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.n_threads = n_threads
        self.ten_crop = ten_crop
        if self.dataset == "cifar10" or self.dataset == "cifar100":
            print "|===>Creating Cifar Data Loader"
            self.train_loader, self.test_loader = self.cifar(dataset=self.dataset)
        elif self.dataset == "mnist":
            print "|===>Creating MNIST Data Loader"
            self.train_loader, self.test_loader = self.mnist()
        else:
            assert False, "invalid data set"

    def getloader(self):
        return self.train_loader, self.test_loader

    def mnist(self):
        norm_mean = [0.1307]
        norm_std = [0.3081]
        train_loader = torch.utils.data.DataLoader(
            dsets.MNIST("/home/dataset/mnist", train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(norm_mean, norm_std)
                        ])),
            batch_size=self.train_batch_size, shuffle=True,
            num_workers=self.n_threads,
            pin_memory=False
        )
        test_loader = torch.utils.data.DataLoader(
            dsets.MNIST("/home/dataset/mnist", train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)
            ])),
            batch_size=self.test_batch_size, shuffle=True,
            num_workers=self.n_threads,
            pin_memory=False
        )
        return train_loader, test_loader


    def cifar(self, dataset="cifar10"):
        assert False, "Write it in yourself"
        return train_loader, test_loader
