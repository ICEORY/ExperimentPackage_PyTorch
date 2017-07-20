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
        if dataset == "cifar10":
            norm_mean = [0.49139968, 0.48215827, 0.44653124]
            norm_std = [0.24703233, 0.24348505, 0.26158768]
        elif dataset == "cifar100":
            norm_mean = [0.50705882, 0.48666667, 0.44078431]
            norm_std = [0.26745098, 0.25568627, 0.27607843]

        else:
            assert False, "Invalid cifar dataset"
        # data_root = "../data"
        data_root = "/home/dataset/cifar"
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)])

        if self.ten_crop:
            test_transform = transforms.Compose([
                auxtransform.TenCrop(28, transforms.Normalize(norm_mean, norm_std))])
            print "use TenCrop()"
        else:
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)])

        # cifar10 data set
        if self.dataset == "cifar10":
            train_dataset = dsets.Cifar10(root=data_root,
                                          train=True,
                                          transform=train_transform,
                                          download=True)

            test_dataset = dsets.CIFAR10(root=data_root,
                                         train=False,
                                         transform=test_transform)
        elif self.dataset == "cifar100":
            train_dataset = dsets.Cifar100(root=data_root,
                                           train=True,
                                           transform=train_transform,
                                           download=True)

            test_dataset = dsets.CIFAR100(root=data_root,
                                          train=False,
                                          transform=test_transform)
        else:
            assert False, "invalid data set"

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.train_batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=self.n_threads)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.test_batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=self.n_threads)
        return train_loader, test_loader
