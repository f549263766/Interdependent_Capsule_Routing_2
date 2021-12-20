"""
@author: QYZ
@time: 2021/09/10
@file: smallnorb.py
@describe: This file aims to create a smallnorb dataset.
"""
import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from utils.tools import AZIMUTH, ELEVATION


def view_point(file_name, exp):
    """
    Judge this image is azimuth or elevation.
    Args:
        file_name: The input image's name.
        exp: The mode of deal with way.
    Returns:
        True or False
    """
    azimuth = file_name[9:11]
    elevation = file_name[12:14]
    if exp == 'azimuth':
        if azimuth in AZIMUTH:
            return True
        else:
            return False
    else:
        if elevation in ELEVATION:
            return True
        else:
            return False


def generate_smallnorb(data_path, mode, train, shuffle=True):
    """
    Generate smallNORB dataset.
    Args:
        data_path: The directory of smallNORB dataset.
        mode: The mode of smallNORB.
        train: Select the train or test data.
        shuffle: Whether to shuffle dataset.
    Returns:
        img_data and labels.
    """
    img_data, labels = [], []
    # Get path for each class folder
    for class_label_idx, class_name in enumerate(os.listdir(data_path)):
        class_path = os.path.join(data_path, class_name)
        # Get name of each file per class and respective class name/label index
        for _, file_name in enumerate(os.listdir(class_path)):
            img = np.load(os.path.join(data_path, class_name, file_name))
            # Out ← [H, W, C] ← [C, H, W]
            if img.shape[0] < img.shape[1]:
                img = np.moveaxis(img, 0, -1)
            if mode == "azimuth":
                if view_point(file_name, mode) and (train == 'train'):
                    img_data.extend([img])
                    labels.append(class_label_idx)
                elif not view_point(file_name, mode) and (train == 'test'):
                    img_data.extend([img])
                    labels.append(class_label_idx)
                elif view_point(file_name, mode) and (train == 'valid'):
                    img_data.extend([img])
                    labels.append(class_label_idx)
                else:
                    pass
            elif mode == "elevation":
                if view_point(file_name, mode) and (train == 'train'):
                    img_data.extend([img])
                    labels.append(class_label_idx)
                elif not view_point(file_name, mode) and (train == 'test'):
                    img_data.extend([img])
                    labels.append(class_label_idx)
                elif view_point(file_name, mode) and (train == 'valid'):
                    img_data.extend([img])
                    labels.append(class_label_idx)
                else:
                    pass
            else:
                img_data.extend([img])
                labels.append(class_label_idx)

    img_data = np.array(img_data, dtype=np.uint8)
    labels = np.array(labels)

    # shuffle the dataset
    if shuffle:
        idx = np.random.permutation(img_data.shape[0])
        img_data = img_data[idx]
        labels = labels[idx]
    return img_data, labels


def random_split(data_set, labels, n_classes, n_samples_per_class):
    """
    Creates a class-balanced validation set from a training set.
    Args:
        data_set: The input dataset for splitting.
        labels: The index value of labels.
        n_classes: The number of classes.
        n_samples_per_class: The number of samples class.
    Return:
        train dataset and validation dataset.
    """
    train_X, train_Y, valid_X, valid_Y = [], [], [], []

    for c in range(n_classes):
        # Get indices of all class 'c' samples
        c_idx = (np.array(labels) == c).nonzero()[0]
        # Get n unique class 'c' samples
        valid_samples = np.random.choice(c_idx, n_samples_per_class[c], replace=False)
        # Get remaining samples of class 'c'
        train_samples = np.setdiff1d(c_idx, valid_samples)
        # Assign class c samples to validation, and remaining to training
        train_X.extend(data_set[train_samples])
        train_Y.extend(labels[train_samples])
        valid_X.extend(data_set[valid_samples])
        valid_Y.extend(labels[valid_samples])

    return {'train': np.stack(train_X), 'valid': np.stack(valid_X)}, \
           {'train': np.stack(train_Y), 'valid': np.stack(valid_Y)}


class SmallNORB(data.Dataset):
    """
    Creat a class for generating s dataset, which's train set and test set are smallNORB train set,
    smallNORB test set if mode equal all, separately.
    Args:
        root: The directory of smallNORB.
        train: whether to use train set , test set or validation set.
        transform: Apply data enhancement.
        download: whether to download the dataset.
        mode: Select classification test dataset or test robust dataset. ["classification", "azimuth", "elevation"]
    """

    raw_folder = "raw"
    processed_folder = "processed"
    train_file = "train_classification.pt"
    validation_file = "validation_classification.pt"
    test_file = "test_classification.pt"
    train_file_azimuth = "train_azimuth.pt"
    validation_file_azimuth = "validation_azimuth.pt"
    test_file_azimuth = "test_azimuth.pt"
    train_file_elevation = "train_elevation.pt"
    validation_file_elevation = "validation_elevation.pt"
    test_file_elevation = "test_elevation.pt"

    def __init__(self, root, train="train", transform=None, download=False, mode="classification"):
        # Initialize the __init__ parameters for input parameters
        self.root = root
        self.train = train
        self.transform = transform
        self.mode = mode
        self.suffix = ".pt"

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        try:
            data_file = self.train + "_" + self.mode + self.suffix
            self.data, self.targets = torch.load(os.path.join(self.root, "smallNORB", self.processed_folder, data_file))
        except FileNotFoundError:
            raise ValueError("Please check you input the parameters of \"train\" and \"mode\"!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform:
            image = self.transform(self.data[idx])
        else:
            image = self.data[idx]
        return image, self.targets[idx]

    def _check_exists(self):
        """
        Check if processed files exists.
        Returns:
            True or False.
        """
        files = (
            self.train_file,
            self.validation_file,
            self.test_file,
            self.train_file_azimuth,
            self.validation_file_azimuth,
            self.test_file_azimuth,
            self.train_file_elevation,
            self.validation_file_elevation,
            self.test_file_elevation
        )
        fpaths = [os.path.exists(os.path.join(self.root, "smallNORB", self.processed_folder, f))
                  for f in files]
        return False not in fpaths

    def download(self):
        """
        Download the smallNORB data if it doesn't exist in processed_folder already.
        """
        import errno

        if self._check_exists():
            return

        try:
            os.makedirs(os.path.join(self.root, "smallNORB", self.raw_folder))
            os.makedirs(os.path.join(self.root, "smallNORB", self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # Processing and saving as torch files
        print('Processing classification smallNORB dataset...')
        # Generating classification smallNORB dataset
        train_classification = generate_smallnorb(
            data_path=os.path.join(self.root, "smallNORB", self.raw_folder, "train"),
            mode="all",
            train="train",
            shuffle=True
        )
        test_classification = generate_smallnorb(
            data_path=os.path.join(self.root, "smallNORB", self.raw_folder, "test"),
            mode="all",
            train="test",
            shuffle=False
        )
        # Return data, labels dicts for new train set and class-balanced valid set
        datas, labels = random_split(data_set=test_classification[0],
                                     labels=test_classification[1],
                                     n_classes=5,
                                     n_samples_per_class=np.unique(
                                         test_classification[1], return_counts=True)[1] // 5)
        # Saving classification smallNORB dataset
        with open(os.path.join(self.root, "smallNORB", self.processed_folder,
                               self.train_file), 'wb') as f:
            torch.save(train_classification, f)
        with open(os.path.join(self.root, "smallNORB", self.processed_folder,
                               self.validation_file), 'wb') as f:
            torch.save((datas['valid'], labels['valid']), f)
        with open(os.path.join(self.root, "smallNORB", self.processed_folder,
                               self.test_file), 'wb') as f:
            torch.save(test_classification, f)

        # Processing and saving as torch files
        print('Processing azimuth smallNORB dataset...')
        # Generating azimuth smallNORB dataset
        train_robust_azimuth = generate_smallnorb(
            data_path=os.path.join(self.root, "smallNORB", self.raw_folder, "train"),
            mode="azimuth",
            train="train"
        )
        valid_robust_azimuth = generate_smallnorb(
            data_path=os.path.join(self.root, "smallNORB", self.raw_folder, "test"),
            mode="azimuth",
            train="valid"
        )
        test_robust_azimuth = generate_smallnorb(
            data_path=os.path.join(self.root, "smallNORB", self.raw_folder, "test"),
            mode="azimuth",
            train="test",
            shuffle=False
        )
        # Saving azimuth smallNORB dataset
        with open(os.path.join(self.root, "smallNORB", self.processed_folder,
                               self.train_file_azimuth), 'wb') as f:
            torch.save(train_robust_azimuth, f)
        with open(os.path.join(self.root, "smallNORB", self.processed_folder,
                               self.validation_file_azimuth), 'wb') as f:
            torch.save(valid_robust_azimuth, f)
        with open(os.path.join(self.root, "smallNORB", self.processed_folder,
                               self.test_file_azimuth), 'wb') as f:
            torch.save(test_robust_azimuth, f)

        # Processing and saving as torch files
        print('Processing elevation smallNORB dataset...')
        # Generating elevation smallNORB dataset
        train_robust_elevation = generate_smallnorb(
            data_path=os.path.join(self.root, "smallNORB", self.raw_folder, "train"),
            mode="elevation",
            train="train"
        )
        valid_robust_elevation = generate_smallnorb(
            data_path=os.path.join(self.root, "smallNORB", self.raw_folder, "test"),
            mode="elevation",
            train="valid"
        )
        test_robust_elevation = generate_smallnorb(
            data_path=os.path.join(self.root, "smallNORB", self.raw_folder, "test"),
            mode="elevation",
            train="test",
            shuffle=False
        )
        # Saving azimuth smallNORB dataset
        with open(os.path.join(self.root, "smallNORB", self.processed_folder,
                               self.train_file_elevation), 'wb') as f:
            torch.save(train_robust_elevation, f)
        with open(os.path.join(self.root, "smallNORB", self.processed_folder,
                               self.validation_file_elevation), 'wb') as f:
            torch.save(valid_robust_elevation, f)
        with open(os.path.join(self.root, "smallNORB", self.processed_folder,
                               self.test_file_elevation), 'wb') as f:
            torch.save(test_robust_elevation, f)
        print('Done!')


class Standardize(object):
    """
    Standardizes a 'PIL Image' such that each channel gets zero mean and unit variance.
    """

    def __call__(self, img):
        return (img - img.mean(dim=(1, 2), keepdim=True)) \
               / torch.clamp(img.std(dim=(1, 2), keepdim=True), min=1e-8)

    def __repr__(self):
        return self.__class__.__name__ + '()'


if __name__ == "__main__":
    train_data = SmallNORB(root="../datasets",
                           train="train",
                           transform=transforms.Compose([
                               transforms.ToPILImage(),
                               transforms.RandomCrop((32, 32)),
                               transforms.ColorJitter(brightness=0 / 255., contrast=0),
                               transforms.ToTensor(),
                               Standardize(),
                           ]),
                           download=True,
                           mode="classification")
    val_data = SmallNORB(root="../datasets",
                         train="validation",
                         transform=transforms.Compose([
                             transforms.ToPILImage(),
                             transforms.CenterCrop((32, 32)),
                             transforms.ToTensor(),
                             Standardize(),
                         ]),
                         download=False,
                         mode="classification")
    test_data = SmallNORB(root="../datasets",
                          train="test",
                          transform=transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.CenterCrop((32, 32)),
                              transforms.ToTensor(),
                              Standardize(),
                          ]),
                          download=False,
                          mode="elevation")
    Train_DataLoader = data.DataLoader(dataset=train_data,
                                       batch_size=64,
                                       num_workers=4,
                                       pin_memory=True,
                                       shuffle=True)
    Valid_DataLoader = data.DataLoader(dataset=val_data,
                                       batch_size=64,
                                       num_workers=4,
                                       pin_memory=True,
                                       shuffle=True)
    Test_DataLoader = data.DataLoader(dataset=test_data,
                                      batch_size=64,
                                      num_workers=4,
                                      pin_memory=True,
                                      shuffle=False)
    print('Number of training examples:', train_data.__len__())
    train_images, train_labels = iter(Train_DataLoader).__next__()
    print("the batch size shape of train dataset:{}".format(train_images.shape))
    print('Number of validation examples:', val_data.__len__())
    val_images, val_labels = iter(Valid_DataLoader).__next__()
    print("the batch size shape of validation dataset:{}".format(val_images.shape))
    print('Number of test examples:', test_data.__len__())
    test_images, test_labels = iter(Test_DataLoader).__next__()
    print("the batch size shape of test dataset:{}".format(test_images.shape))
    print(val_labels)
