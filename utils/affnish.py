"""
@author: QYZ
@time: 2021/09/08
@file: affnish.py
@describe: This file aims to design a affnish dataset function.
"""
import os
import numpy as np
import scipy.io as spio
import torch
import torchvision
import torch.utils.data as data
from torchvision import datasets
from scipy.io.matlab import mio5_params

from utils.tools import one_hot_encode, imshow


def place_random(trainX):
    """
    Randomly placing 28x28 mnist image on 40x40 background
    Args:
        trainX: the input MNIST dataset.
    Return:
        new randomly place MNIST.
    """
    trainX_new = []
    for img in trainX:
        img_new = np.zeros((40, 40, 1), dtype=np.float32)
        x = np.random.randint(12, size=1)[0]
        y = np.random.randint(12, size=1)[0]

        img_new[y:y + 28, x:x + 28, :] = img
        trainX_new.append(img_new)

    return np.array(trainX_new)


def randomly_mnist(mnist_data):
    """
    load MNIST dataset and preprocess data to generate randomly place mnist dataset.
    Args:
        mnist_data: the input mnist dataset.
    Return:
        randomly place data and label.
    """
    # Extract data and labels from dataset
    mnist_data, mnist_labels = mnist_data.data, mnist_data.targets
    # Randomly place MNIST in 40x40 black background
    random_mnist = place_random(mnist_data[..., None])
    # Normalize the data
    random_mnist_datas = (random_mnist / 255).astype(np.float32).reshape(-1, 1, 40, 40)
    # Encode the label as one-hot encode
    random_mnist_labels = one_hot_encode(mnist_labels, 10)

    return torch.Tensor(random_mnist_datas), torch.Tensor(random_mnist_labels)


class AffNISH(data.Dataset):
    """
    Creat a class for generating affNIST dataset, which's train set and test set are randomly place MNIST,
    affNIST, separately.
    Args:
        root: The directory of MNIST.
        train: whether to use train set or test set.
        transform: Apply data enhancement.
        download: whether to download the dataset.
        mode: Select single test mat or test all mat.
        num: The serial number of test mat.
    """

    raw_folder = "raw"
    processed_folder = "processed"
    train_file = "train_random_mnist.pt"
    test_file = "test_affnish.pt"
    test_all_file = "test_all_affnish.pt"

    def __init__(self, root, train=True, transform=None, download=False, mode="single", num=1):
        # Initialize the __init__ parameters for input parameters
        self.root = root
        self.train = train
        self.transform = transform
        self.mode = mode
        self.num = num

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.train_file
        else:
            if self.mode == "single":
                data_file = self.test_file
            else:
                data_file = self.test_all_file
        self.data, self.targets = torch.load(os.path.join(self.root, "affNIST", self.processed_folder, data_file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        if self.transform is not None:
            if img.shape[0] == 3:
                img = np.transpose(img.numpy, (1, 2, 0))
            else:
                img = img.squeeze().numpy()
            img = self.transform(img)

        return img, target

    def _check_exists(self):
        """
        Check if processed files exists.
        Returns:
            True or False.
        """
        files = (
            self.train_file,
            self.test_file,
            self.test_all_file
        )
        fpaths = [os.path.exists(os.path.join(self.root, "affNIST", self.processed_folder, f))
                  for f in files]
        return False not in fpaths

    def download(self):
        """
        Download the affnish data if it doesn't exist in processed_folder already.
        Returns:
            None
        """
        import errno

        if self._check_exists():
            return

        try:
            os.makedirs(os.path.join(self.root, "affNIST", self.raw_folder))
            os.makedirs(os.path.join(self.root, "affNIST", self.processed_folder))
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
        # Processing and saving as torch files
        print('Processing Randomly Place MNIST...')
        # Generating randomly place mnist dataset
        random_train_mnist_data, random_train_mnist_label = randomly_mnist(mnist_train_base)
        random_train_mnist = (random_train_mnist_data, random_train_mnist_label)
        # Saving randomly mnist dataset
        with open(os.path.join(self.root, "affNIST", self.processed_folder,
                               self.train_file), 'wb') as f:
            torch.save(random_train_mnist, f)

        # Processing and saving as torch files
        print('Processing AffNIST...')
        test_affNIST_data, test_affNIST_label = self.generate_affNIST(
            img_path=os.path.join(self.root, "affNIST", self.raw_folder),
            name="test_batches",
            num=self.num)
        test_affNIST = (test_affNIST_data, test_affNIST_label)
        test_affNIST_all_data, test_affNIST_all_label = self.generate_affNIST(
            img_path=os.path.join(self.root, "affNIST", self.raw_folder),
            name="test_batches",
            num=None)
        test_affNIST_all = (test_affNIST_all_data, test_affNIST_all_label)
        # Saving affNIST dataset
        with open(os.path.join(self.root, "affNIST", self.processed_folder,
                               self.test_file), 'wb') as f:
            torch.save(test_affNIST, f)
        with open(os.path.join(self.root, "affNIST", self.processed_folder,
                               self.test_all_file), 'wb') as f:
            torch.save(test_affNIST_all, f)
        print('Done!')

    def generate_affNIST(self, img_path, name, num):
        """
        Generate the affNIST dataset.
        Args:
            img_path: The directory where the data set is stored.
            name: Select the name of dataset, train, valid or test.
            num: Select any label dataset.
        Return:
            packaged affNIST data and label
        """
        if num is not None:
            path = f"{img_path}/{name}/{num}.mat"
        else:
            path = f"{img_path}/test.mat"
        print(path)
        dataset = self.load_mat(path)
        label_set = dataset['affNISTdata']['label_int'].reshape((-1)).astype(np.int32)
        image_set = dataset['affNISTdata']['image'].transpose().reshape((-1, 1, 40, 40)).astype(np.float32)
        image_set = (image_set / 255).astype(np.float32)
        label_set = one_hot_encode(label_set, 10)
        return torch.Tensor(image_set), label_set

    def load_mat(self, filename):
        """
        This function should be called instead of direct spio.loadmat
        as it cures the problem of not properly recovering python dictionaries
        from mat files. It calls the function check keys to cure all entries
        which are still mat-objects
        Args:
            filename: the directory of affnish dataset.
        Returns:
            Dict mat dataset.
        """
        data_mat = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
        return self._check_keys(data_mat)

    def _check_keys(self, dicts):
        """
        Checks if entries in dictionary are mat-objects. If yes
        to dict is called to change them to nested dictionaries
        Args:
            dicts: the input mat for checking if mat-object.
        Returns:
            Dictionary nested MAT data.
        """
        for key in dicts:
            if isinstance(dicts[key], spio.matlab.mio5_params.mat_struct):
                dicts[key] = self._todict(dicts[key])
        return dicts

    def _todict(self, matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        Args:
            matobj: the input keys of dicts.
        Returns:
            Dictionary nested MAT data.
        """
        dicts = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                dicts[strg] = self._todict(elem)
            else:
                dicts[strg] = elem
        return dicts


if __name__ == "__main__":
    import torchvision.transforms as transforms
    train_data = AffNISH(root="../datasets",
                         train=True,
                         transform=transforms.ToTensor(),
                         download=True,
                         mode="single",
                         num=1)
    test_data = AffNISH(root="../datasets",
                        train=False,
                        transform=None,
                        download=True,
                        mode="all",
                        num=1)
    Train_DataLoader = data.DataLoader(dataset=train_data,
                                       batch_size=64,
                                       num_workers=4,
                                       shuffle=True)
    Test_DataLoader = data.DataLoader(dataset=test_data,
                                      batch_size=64,
                                      num_workers=4,
                                      shuffle=True)
    print('Number of training examples:', train_data.__len__())
    train_images, train_labels = iter(Train_DataLoader).__next__()
    print("the batch size shape of train dataset:{}".format(train_images.shape))
    imshow(torchvision.utils.make_grid(train_images), True, name="random_mnist.png")
    print('Number of test examples:', test_data.__len__())
    test_images, test_labels = iter(Test_DataLoader).__next__()
    print("the batch size shape of test dataset:{}".format(test_images.shape))
    imshow(torchvision.utils.make_grid(test_images), True, name="affNIST.png")
