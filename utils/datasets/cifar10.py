import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# mean and std of cifar10 dataset
# CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
# CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)


def get_cifar10_training_dataloader(data_path, batch_size, workers, shuffle):
    """ return training dataloader
    Args:
        data_path: path to cifar10 training python dataset
        batch_size: dataloader batchsize
        workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    cifar10_training = datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=transform_train)
    cifar10_training_loader = DataLoader(
        cifar10_training, shuffle=shuffle, num_workers=workers, batch_size=batch_size)

    return cifar10_training_loader


def get_cifar10_test_dataloader(data_path, batch_size, workers, shuffle):
    """ return training dataloader
    Args:
        path: path to cifar10 test python dataset
        batch_size: dataloader batchsize
        workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar10_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    cifar10_test = datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=transform_test)
    cifar10_test_loader = DataLoader(
        cifar10_test, shuffle=shuffle, num_workers=workers, batch_size=batch_size)

    return cifar10_test_loader


def build_cifar10_data(data_path, batch_size, workers):
    """ return training dataloader and test dataloader
    Args:
        data_path: path to cifar10 dataset
        batch_size: dataloader batchsize
        workers: dataloader num_works
    Returns: cifar10_training_loader, cifar10_test_loader:torch dataloader object
    """

    cifar10_training_loader = get_cifar10_training_dataloader(
        data_path, batch_size, workers, shuffle=True)
    cifar10_test_loader = get_cifar10_test_dataloader(
        data_path, batch_size, workers, shuffle=False)

    return cifar10_training_loader, cifar10_test_loader