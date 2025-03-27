import torch
import torchvision
import os
import numpy as np
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
import argparse
#---------------------------------------------- device:cpu dtype:float32-----------------------------------------------

class Cifar10_FewshotDataset(Dataset):
    def __init__(self, args):
        self.shots = args["shots"]
        self.prepocess = args["preprocess"]
        self.all_train = CIFAR10(os.path.expanduser("../dataset"), transform=None, download=True, train=True)
        self.new_train_data = self.construct_few_shot_data()

    def __len__(self):
        return len(self.new_train_data)

    def __getitem__(self, idx):
        return {"image": self.new_train_data[idx][0], "label": self.new_train_data[idx][1]}

    def construct_few_shot_data(self):
        new_train_data = []
        train_shot_count = {}
        all_indices = list(range(len(self.all_train)))
        np.random.shuffle(all_indices)

        for index in all_indices:
            label = self.all_train[index][1]
            if label not in train_shot_count:
                train_shot_count[label] = 0

            if train_shot_count[label] < self.shots:
                tmp = self.all_train[index]
                tmp_data = [self.prepocess(tmp[0]), tmp[1]]  # convert img to compatible tensors
                new_train_data.append(tmp_data)
                train_shot_count[label] += 1

        return new_train_data

def load_train_cifar10(batch_size=1, shots=16, preprocess=None):
    args = {"shots": shots, "preprocess": preprocess}
    train_data = Cifar10_FewshotDataset(args)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_data, train_loader

class Cifar10_TestDataset(Dataset):
    def __init__(self, args):
        self.prepocess = args["preprocess"]
        self.all_test = CIFAR10(os.path.expanduser("../dataset"), transform=self.prepocess, download=True, train=False)

    def __len__(self):
        return len(self.all_test)

    def __getitem__(self, idx):
        return {"image": self.all_test[idx][0], "label": self.all_test[idx][1]}

def load_test_cifar10(batch_size=1, preprocess=None):
    args = {"preprocess": preprocess}
    test_data = Cifar10_TestDataset(args)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)
    return test_data, test_loader

print(os.path.expanduser("../dataset"))
