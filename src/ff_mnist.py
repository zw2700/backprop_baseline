import numpy as np
import torch

from src import utils


class FF_MNIST(torch.utils.data.Dataset):
    def __init__(self, opt, partition, num_classes=10):
        self.opt = opt
        self.mnist = utils.get_MNIST_partition(opt, partition)
        self.num_classes = num_classes
        self.uniform_label = torch.ones(self.num_classes) / self.num_classes

    def __getitem__(self, index):
        sample, class_label = self.mnist[index]

        inputs = {"sample": sample}
        labels = {"class_labels": class_label}
        return inputs, labels

    def __len__(self):
        return len(self.mnist)
