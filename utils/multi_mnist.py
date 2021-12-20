"""
@author: QYZ
@time: 2021/09/10
@file: multi_mnist.py
@describe: This file aims to create a MultiMNIST dtaset.
"""
import os

import numpy as np
import torch
import torchvision
import torch.utils.data as data
from torchvision import datasets

from utils.tools import one_hot_encode, imshow


def pad_dataset(img):
    """
    Padding the images.
    Args:
        img: the input images data.
    Return:
        the images after padding.
    """
    return np.pad(img, [(0, 0), (4, 4), (4, 4)])


def pre_process(img, label):
    """
    Normalizing the images by dividing 255, it's pixel value will put in 0-1 range.
    Generating the corresponding one-hot labels.
    Args:
        img: the input images data.
        label: the corresponding labels of input images.
    Return:
        the images after normalizing and the corresponding one-hot labels.
    """
    return (img / 255)[..., None].astype('float32'), one_hot_encode(label, length=10)


def shift_images(images, shifts, max_shift):
    """
    Randomly shifting the digital of the images.
    Args:
        images: the input images data.
        shifts: the value of randomly shifting.
        max_shift: the max-value of shifting.
    Return:
        the images after shifting
    """
    lenI = images.shape[1]
    images_sh = np.pad(images, ((0, 0), (max_shift, max_shift), (max_shift, max_shift), (0, 0)))
    shifts = max_shift - shifts
    batches = np.arange(len(images))[:, None, None]
    images_sh = images_sh[
        batches, np.arange(lenI + max_shift * 2)[None, :, None], (shifts[:, 0, None] + np.arange(0, lenI))[:, None, :]]
    images_sh = images_sh[batches, (shifts[:, 1, None] + np.arange(0, lenI))[..., None], np.arange(lenI)[None, None]]
    return images_sh


def merge_with_image(images, labels, i, shift, n_multi=1000):
    """
    The same image randomly fuses multiple other categories of images.
    Args:
        images: Images to be fused.
        labels: The label of the image to be fused.
        i: Index of the list of images.
        shift: Random translation of fused image.
        n_multi: Number of fused images.
    Returns:
        merged, merged_labels
    """
    base_image = images[i]
    base_label = labels[i]
    base_indexes = np.arange(len(images))
    indexes = np.arange(len(images))[np.bitwise_not((labels == base_label).all(axis=-1))]
    indexes = base_indexes * indexes
    indexes = np.setdiff1d(indexes, 0)
    indexes = np.random.choice(indexes, n_multi, replace=False)
    top_images = images[indexes]
    top_labels = labels[indexes]
    shifts = np.random.randint(-shift, shift + 1, (n_multi + 1, 2))
    images_sh = shift_images(np.concatenate((base_image[None], top_images), axis=0), shifts, shift)
    base_sh = images_sh[0]
    top_sh = images_sh[1:]
    merged = np.clip(base_sh + top_sh, 0, 1)
    merged_labels = base_label + top_labels
    return merged, merged_labels


class MultiMNIST(data.IterableDataset):
    """
    Creat a class for generating MultiMNIST dataset, which's train set and test set are a data set generated
    by two overlapping numbers.
    Args:
        root: The directory of MNIST.
        train: whether to use train set or test set.
        transform: Apply data enhancement.
        download: whether to download the dataset.
        n_multi: The number of generating MultiMNIST in each digits.
    """
    raw_folder = "raw"
    processed_folder = "processed"
    train_file = "train_MultiMNIST.pt"
    test_file = "test_MultiMNIST.pt"

    def __init__(self, root, train="train", transform=None, download=False, n_multi=1000):
        # Initialize the __init__ parameters for input parameters
        self.root = root
        self.train = train
        self.transform = transform
        self.n_multi = n_multi

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train == "train":
            data_file = self.train_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.root, "MultiMNIST", self.processed_folder, data_file))

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        # Merge two digits into a image
        if self.train == "train":
            while True:
                i = np.random.randint(len(self.data))
                j = np.random.randint(len(self.data))
                while np.all(self.data[i] == self.data[j]) or torch.equal(self.targets[i], self.targets[j]):
                    j = np.random.randint(len(self.data))
                base = shift_images(self.data[i:i + 1], np.random.randint(-4, 4 + 1, (1, 2)), 4)[0]
                top = shift_images(self.data[j:j + 1], np.random.randint(-4, 4 + 1, (1, 2)), 4)[0]
                merged = np.clip(np.add(base, top), 0, 1)
                merged = np.transpose(merged, (2, 0, 1))
                yield merged, self.targets[i] + self.targets[j]
        elif self.train == "valid":
            for i in range(len(self.data)):
                j = np.random.randint(len(self.data))
                while torch.equal(self.targets[i], self.targets[j]):
                    j = np.random.randint(len(self.data))
                base = shift_images(self.data[i:i + 1], np.random.randint(-4, 4 + 1, (1, 2)), 4)[0]
                top = shift_images(self.data[j:j + 1], np.random.randint(-4, 4 + 1, (1, 2)), 4)[0]
                merged = np.clip(np.add(base, top), 0, 1)
                merged = np.transpose(merged, (2, 0, 1))
                yield merged, self.targets[i] + self.targets[j]
        elif self.train == "test":
            for i in range(len(self.data)):
                X_merged, y_merged = merge_with_image(self.data, self.targets, i, 4, self.n_multi)
                X_merged = np.transpose(X_merged, (0, 3, 1, 2))
                yield X_merged, y_merged
        else:
            raise ValueError("please check the input parameter \"train\".")

    def _check_exists(self):
        """
        Check if processed files exists.
        Returns:
            True or False.
        """
        files = (
            self.train_file,
            self.test_file
        )
        fpaths = [os.path.exists(os.path.join(self.root, "MultiMNIST", self.processed_folder, f))
                  for f in files]
        return False not in fpaths

    def download(self):
        """
        Download the MultiMNIST data if it doesn't exist in processed_folder already.
        """
        import errno

        if self._check_exists():
            return

        try:
            os.makedirs(os.path.join(self.root, "MultiMNIST", self.raw_folder))
            os.makedirs(os.path.join(self.root, "MultiMNIST", self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # Loading MNIST basic dataset
        mnist_train_base = datasets.MNIST(root=self.root,
                                          train=True,
                                          download=False,
                                          transform=None)
        mnist_test_base = datasets.MNIST(root=self.root,
                                         train=False,
                                         download=False,
                                         transform=None)

        # Processing and saving as torch files
        print('Processing MultiMNIST dataset...')
        # Padding 4 pixels in images
        train_data = pad_dataset(mnist_train_base.data)
        test_data = pad_dataset(mnist_test_base.data)
        # Normalize, img = img /255.0; Target => one-hot labels
        train_data, train_target = pre_process(train_data, mnist_train_base.targets)
        test_data, test_target = pre_process(test_data, mnist_test_base.targets)
        # Saving classification MultiMNIST dataset
        with open(os.path.join(self.root, "MultiMNIST", self.processed_folder,
                               self.train_file), 'wb') as f:
            torch.save((train_data, train_target), f)
        with open(os.path.join(self.root, "MultiMNIST", self.processed_folder,
                               self.test_file), 'wb') as f:
            torch.save((test_data, test_target), f)
        print('Done!')


if __name__ == "__main__":
    dataset = MultiMNIST(root="../datasets/",
                         train="test",
                         transform=None,
                         download=True,
                         n_multi=1)
    Train_DataLoader = data.DataLoader(dataset=dataset,
                                       batch_size=64,
                                       num_workers=0,
                                       shuffle=False)
    print('Number of training examples:', dataset.__len__())
    train_images, train_labels = iter(Train_DataLoader).__next__()
    print("the batch size shape of train dataset:{}".format(train_images.shape))
    print("the corresponding train labels shape:{}".format(train_labels.shape))
    imshow(torchvision.utils.make_grid(train_images.reshape(-1, 1, 36, 36)), True, name="MultiMNIST_test.png")
