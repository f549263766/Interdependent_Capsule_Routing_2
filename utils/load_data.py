"""
@author: QYZ
@time: 2021/09/08
@file: load_data.py
@describe: This file aims to design a function for loading dataset.
"""
import os
import argparse
import numpy as np
import torchvision.transforms as transforms
from torch.utils import data
from torchvision import datasets

from utils.affnish import AffNISH
from utils.tools import DATA_NAME
from utils.multi_mnist import MultiMNIST
from utils.smallnorb import SmallNORB, Standardize
from utils.tools import split_dataset


def worker_init_fn(worker_id):
    """
    This function aims to set the initial seed before load DataLoader.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class LoadData:
    """
    This function aims to achieve loading data into models.
    Args:
        args: The data parameter settings.
    """

    def __init__(self, args):
        # Initial the setting of parameter
        self.args = args
        self.data_name = args.data_name
        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

    def select_data(self):
        """
        This function aims to select a set as data and then load corresponding transformation.
        :return: train_iterator, valid_iterator, test_iterator.
        """
        # Determine whether the input is correct
        if self.data_name not in DATA_NAME:
            raise ValueError("Please input the correct name of data, such as MNIST!")

        # Select proper transform for data
        train_transform, test_transform = self.transform_setting()

        # Load the selected data from destination folder
        if self.data_name in [
            "MNIST",
            "CIFAR10",
            "FashionMNIST",
        ]:
            universal = getattr(datasets, self.data_name)
            train_data = universal(root=self.data_path,
                                   train=True,
                                   download=True,
                                   transform=transforms.Compose(train_transform))
            test_data = universal(root=self.data_path,
                                  train=False,
                                  download=True,
                                  transform=transforms.Compose(test_transform))
            train_data, valid_data = split_dataset(train_data, random_indices=True)
        elif self.data_name == "SVHN":
            train_data = datasets.SVHN(root=os.path.join(self.data_path, self.data_name),
                                       split='train',
                                       download=True,
                                       transform=transforms.Compose(train_transform))
            test_data = datasets.SVHN(root=os.path.join(self.data_path, self.data_name),
                                      split='test',
                                      download=True,
                                      transform=transforms.Compose(test_transform))
            train_data, valid_data = split_dataset(train_data, random_indices=True)
        elif self.data_name == "affNIST":
            train_data = AffNISH(root=self.data_path,
                                 train=True,
                                 transform=None,
                                 download=True,
                                 mode="single",
                                 num=1)
            test_data = AffNISH(root=self.data_path,
                                train=False,
                                transform=None,
                                download=True,
                                mode=self.args.affNIST_mode,
                                num=1)
            train_data, valid_data = split_dataset(train_data, random_indices=True)
        elif self.data_name == "MultiMNIST":
            train_data = MultiMNIST(root=self.data_path,
                                    train="train",
                                    transform=None,
                                    download=True,
                                    n_multi=self.args.n_multi)
            valid_data = MultiMNIST(root=self.data_path,
                                    train="train",
                                    transform=None,
                                    download=True,
                                    n_multi=self.args.n_multi)
            test_data = MultiMNIST(root=self.data_path,
                                   train="train",
                                   transform=None,
                                   download=True,
                                   n_multi=self.args.n_multi)
        elif self.data_name == "smallNORB":
            train_data = SmallNORB(root=self.data_path,
                                   train="train",
                                   transform=transforms.Compose(train_transform),
                                   download=True,
                                   mode=self.args.smallNORB_mode)
            test_data = SmallNORB(root=self.data_path,
                                  train="test",
                                  transform=transforms.Compose(test_transform),
                                  download=True,
                                  mode=self.args.smallNORB_mode)
            valid_data = SmallNORB(root=self.data_path,
                                   train="validation",
                                   transform=transforms.Compose(test_transform),
                                   download=True,
                                   mode=self.args.smallNORB_mode)
        else:
            train_data, valid_data, test_data = None, None, None

        # Load data set into DataLoader
        train_iterator = data.DataLoader(dataset=train_data,
                                         batch_size=self.batch_size,
                                         num_workers=self.num_workers,
                                         shuffle=True,
                                         pin_memory=True,
                                         worker_init_fn=worker_init_fn)
        valid_iterator = data.DataLoader(dataset=valid_data,
                                         batch_size=self.batch_size,
                                         num_workers=self.num_workers,
                                         shuffle=False)
        test_iterator = data.DataLoader(dataset=test_data,
                                        batch_size=self.batch_size,
                                        num_workers=self.num_workers,
                                        shuffle=False)
        return train_iterator, valid_iterator, test_iterator

    def transform_setting(self):
        """
        This function aims to set transform for selected data.
        Return:
            train_transform, test_transform.
        """
        if self.data_name == "MNIST":
            train_transform = [
                transforms.RandomCrop((32, 32), padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.13066047,), (0.30810780,))
            ]
            test_transform = [
                transforms.Pad(np.maximum(0, (32 - 28) // 2)),
                transforms.ToTensor(),
                transforms.Normalize((0.13066047,), (0.30810780,))
            ]
        elif self.data_name == "SVHN":
            train_transform = [
                transforms.RandomCrop(size=32, padding=4),
                transforms.ColorJitter(brightness=0, contrast=0),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4376821, 0.4437697, 0.47280442),
                                     std=(0.19803012, 0.20101562, 0.19703614))
            ]
            test_transform = [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4376821, 0.4437697, 0.47280442),
                                     std=(0.19803012, 0.20101562, 0.19703614))
            ]
        elif self.data_name == "CIFAR10":
            train_transform = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
            ]
            test_transform = [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
            ]
        elif self.data_name == "FashionMNIST":
            train_transform = [
                transforms.RandomCrop(28, padding=2),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.286, std=0.352)
            ]
            test_transform = [
                transforms.ToTensor(),
                transforms.Normalize(mean=0.286, std=0.352)
            ]
        elif self.data_name == "smallNORB":
            train_transform = [
                transforms.ToPILImage(),
                transforms.RandomCrop((32, 32)),
                transforms.ColorJitter(brightness=0. / 255., contrast=0.),
                transforms.ToTensor(),
                Standardize(),
            ]
            test_transform = [
                transforms.ToPILImage(),
                transforms.CenterCrop((32, 32)),
                transforms.ToTensor(),
                Standardize(),
            ]
        else:
            train_transform, test_transform = None, None

        return train_transform, test_transform


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Load Data")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch Size refers to the number of training samples in each Batch")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Setting learning rate for this model")
    parser.add_argument("--data_name", type=str, default="smallNORB",
                        help="Selecting the dataset for model")
    parser.add_argument("--data_path", type=str, default="../datasets/",
                        help="Setting the directory path to find the data")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Setting the number of workers to load the dataset")
    parser.add_argument("--affNIST_mode", type=str, default="all",
                        help="Setting the mode for generating affNIST dataset")
    parser.add_argument("--smallNORB_mode", type=str, default="azimuth",
                        help="Setting the mode for generating smallNORB dataset")
    parser.add_argument("--n_multi", type=int, default=1000,
                        help="Setting the number of generating MultiMNIST dataset in each digits")
    arg = parser.parse_args()
    Train_iteration, Valid_iteration, Test_iteration = LoadData(arg).select_data()
    print('Number of training examples:', Train_iteration.dataset.__len__())
    print('Number of validation examples:', Valid_iteration.dataset.__len__())
    print('Number of testing examples:', Test_iteration.dataset.__len__())
    # show the picture
    images, labels = iter(Test_iteration).__next__()
    print(labels)
    print(images.shape)
