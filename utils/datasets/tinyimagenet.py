import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# mean and std of tinyimagenet dataset
TINYIMAGENET_MEAN = (0.4802, 0.4481, 0.3975)
TINYIMAGENET_STD = (0.2302, 0.2265, 0.2262)


def get_tinyimagenet_training_dataloader(data_dir, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        data_dir: path to tinyimagenet training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(TINYIMAGENET_MEAN, TINYIMAGENET_STD)
    ])
    tinyimagenet_training = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform_train)
    tinyimagenet_training_loader = DataLoader(
        tinyimagenet_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return tinyimagenet_training_loader


def get_tinyimagenet_test_dataloader(data_dir, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        path: path to tinyimagenet test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: tinyimagenet_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(TINYIMAGENET_MEAN, TINYIMAGENET_STD)
    ])
    tinyimagenet_test = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform_test)
    tinyimagenet_test_loader = DataLoader(
        tinyimagenet_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return tinyimagenet_test_loader
