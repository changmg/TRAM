import random
import os
import numpy as np
import torch
import torch.nn as nn
import time
from tqdm import tqdm

from conf import settings
print_log = settings.LOGGER.info


def set_seed(seed=1005):
    """ set seed for reproducibility
    Args:
        seed: random seed
    Returns: 
        None
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_model(model, train_loader, optimizer, epoch, tensorboard_writer=None):
    """ train model on train dataset
    Args:
        model: model to train
        train_loader: train dataloader
        optimizer: optimizer
        epoch: current epoch
    """
    model.train()
    model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar('Loss/train_loss1', loss, epoch)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 50 == 0:
            print_log(f'Train Epoch {epoch}: [{(batch_idx + 1) * len(data)}/{len(train_loader.dataset)}]\tloss: {loss.item()}')
    print_log(f'Train Epoch {epoch}: [{len(train_loader.dataset)}/{len(train_loader.dataset)}]\tloss: {loss.item()}')


def test_model(model, test_loader, epoch=-1, tensorboard_writer=None):
    """ test model on test dataset
    Args:
        model: model to test
        test_loader: test dataloader
    Returns: 
        top1_acc: top1 accuracy
        # top5_acc: top5 accuracy
    """
    model.cuda()
    model.eval()
    correct_1 = 0.0
    test_loss = 0.0
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(tqdm(test_loader, desc="Testing model")):
            image = image.cuda()
            label = label.cuda()
            output = model(image)
            criterion = nn.CrossEntropyLoss()

            loss = criterion(output, label)
            test_loss += loss

            _, pred = output.topk(5, 1, largest=True, sorted=True)
            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()
            correct_1 += correct[:, :1].sum()

    acc1 = (correct_1 / len(test_loader.dataset)).item()
    test_loss = (test_loss / (batch_idx + 1)).item()
    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar('Loss/test_loss1', test_loss, epoch)
        tensorboard_writer.add_scalar('Accuracy/Top1', acc1, epoch)

    return acc1, test_loss


def report_time_and_speed(t_begin, epoch, epochs, len_train_loader):
    elapse_time = time.time() - t_begin
    speed_epoch = elapse_time / (epoch + 1)
    speed_batch = speed_epoch / len_train_loader
    eta = speed_epoch * epochs - elapse_time
    print_log("Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s\n".format(
        elapse_time, speed_epoch, speed_batch, eta))


def tensor_to_str(x: torch.Tensor, threshold: int=10) -> str:
    """ convert tensor to string
    Args:
        x: tensor
    Returns: 
        str_x: string representation of tensor
    """
    torch.set_printoptions(threshold=threshold)
    str_x = str(x.cpu().detach().squeeze())
    return str_x


def tensor_list_to_str(x_list: list, threshold: int=10) -> str:
    """ convert list of tensors to string
    Args:
        x_list: list of tensors
    Returns: 
        str_x: string representation of list of tensors
    """
    torch.set_printoptions(threshold=threshold)
    str_x = '[' + ', '.join([str(x.cpu().detach().squeeze()) for x in x_list]) + ']'
    return str_x