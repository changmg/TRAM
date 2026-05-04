import argparse
import os
import torch
import logging
import sys
import time

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils.datasets as datasets

import models.cifar10
import models.imagenet

from conf import settings
print_log = settings.LOGGER.info
from utils.common import set_seed, report_time_and_speed
from utils.common import test_model as test_model_nolambda
from quant.quant_model import QuantModel


def parse_args():
    parser = argparse.ArgumentParser(description='running parameters', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general parameters for data and model
    parser.add_argument('--seed', default=1005, type=int, help='random seed for results reproduction')
    parser.add_argument('--pretrained_weights', default='state_dicts/cifar10/resnet18_w8a8_ptq.pth', type=str, help='path to FP32 pretrained weights')
    parser.add_argument('--num_workers', default=16, type=int, help='number of workers for data loader')

    # quantization parameters
    parser.add_argument('--channel_wise', action='store_true', help='apply channel_wise quantization for weights')
    parser.add_argument('--nbits_w', default=8, type=int, help='bitwidth for weight quantization')
    parser.add_argument('--nbits_a', default=8, type=int, help='bitwidth for activation quantization')

    # retraining parameters
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate for retraining')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for retraining')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay for retraining')
    parser.add_argument('--epochs', default=10, type=int, help='number of epochs for retraining')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--lambd', default=1.0, type=float, help='factor of hardware cost')

    # log file path
    parser.add_argument('--log', default='./tmp/test.log', type=str, help='path to log file')

    return parser.parse_args()


def train_model(model, train_loader, optimizer, epoch, lambd: float, tensorboard_writer=None):
    model.train()
    model.cuda()
    criterion = torch.nn.CrossEntropyLoss()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)

        loss1 = criterion(output, target)
        loss2 = 0.0 if lambd == 0.0 else model.compute_hardware_loss()

        loss = loss1 + lambd * loss2
        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar('Loss/train_loss1', loss1, epoch)
            tensorboard_writer.add_scalar('Loss/train_loss2', loss2, epoch)
            tensorboard_writer.add_scalar('Loss/train_loss', loss, epoch)

        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 50 == 0:
            print_log(f'Train Epoch {epoch}: [{(batch_idx + 1) * len(data)}/{len(train_loader.dataset)}]\tloss1: {loss1.item()}\tloss2: {loss2}\tloss: {loss.item()}')
    print_log(f'Train Epoch {epoch}: [{len(train_loader.dataset)}/{len(train_loader.dataset)}]\tloss1: {loss1.item()}\tloss2: {loss2}\tloss: {loss.item()}')


def test_model(model, test_loader, lambd: float = 1.0, epoch=-1, tensorboard_writer=None):
    model.cuda()
    model.eval()
    correct_1 = 0.0
    test_loss1 = 0.0
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(tqdm(test_loader, desc="Testing model")):
            image = image.cuda()
            label = label.cuda()
            output = model(image)
            criterion = torch.nn.CrossEntropyLoss()

            loss1 = criterion(output, label)
            test_loss1 += loss1

            _, pred = output.topk(5, 1, largest=True, sorted=True)
            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()
            correct_1 += correct[:, :1].sum()
    
    acc1 = (correct_1 / len(test_loader.dataset)).item()
    loss1 = (test_loss1 / (batch_idx + 1)).item()
    _print = True if lambd != 0.0 else False
    loss2 = model.compute_hardware_loss(_print=_print).item()
    loss = loss1 + lambd * loss2
    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar('Accuracy/Top1', acc1, epoch)
        tensorboard_writer.add_scalar('Loss/test_loss1', loss1, epoch)
        tensorboard_writer.add_scalar('Loss/test_loss2', loss2, epoch)
        tensorboard_writer.add_scalar('Loss/test_loss', loss, epoch)

    # return acc1, acc5, loss1, loss2, loss
    return acc1, loss1, loss2, loss


def main():
    # parse arguments
    args = parse_args()

    # initialize logger
    assert args.log != '', 'Please specify the log file path using --log'
    if args.log != '':
        print(f'log file: {args.log}')
        os.makedirs(os.path.dirname(args.log), exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename=args.log, filemode='w')
    # else:
    #     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', stream=sys.stdout)
    torch.set_printoptions(precision=6)

    # print arguments
    print_log(args)

    # set seed
    set_seed(args.seed)

    # get date and time
    date_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    print_log(f'date_time: {date_time}')

    # setup tensorboard
    if args.log != '':
        tensorboard_log_path = f'{args.log}_{date_time}'
    else:
        tensorboard_log_path = f'./tmp/tensorboard_{date_time}'
    writer = SummaryWriter(tensorboard_log_path)
    print(f'tensorboard log path: {tensorboard_log_path}')

    # load dataset and model
    if args.pretrained_weights.find('cifar10') != -1:
        dataset_name, data_path = 'cifar10', settings.DATASET_PATHS['cifar10']
        print_log(f'loading dataset from {data_path}...')
        train_loader, test_loader = datasets.cifar10.build_cifar10_data(batch_size=args.batch_size, workers=args.num_workers, data_path=data_path)
        if args.pretrained_weights.find('vgg19') != -1:
            fp_model, model_name = models.cifar10.vgg.vgg19_bn(), 'vgg19'
        elif args.pretrained_weights.find('resnet18') != -1:
            fp_model, model_name = models.cifar10.resnet.resnet18(), 'resnet18'
        elif args.pretrained_weights.find('resnet34') != -1:
            fp_model, model_name = models.cifar10.resnet.resnet34(), 'resnet34'
        elif args.pretrained_weights.find('resnet50') != -1:
            fp_model, model_name = models.cifar10.resnet.resnet50(), 'resnet50'
        elif args.pretrained_weights.find('densenet161') != -1:
            fp_model, model_name = models.cifar10.densenet.densenet161(), 'densenet161'
        elif args.pretrained_weights.find('inception_v3') != -1:
            fp_model, model_name = models.cifar10.inception.inception_v3(), 'inception_v3'
        else:
            raise ValueError(f'Unknown model: {args.pretrained_weights}')
    else:
        raise ValueError(f'Unknown dataset: {args.data_path}')

    # quantize model
    wq_params = {'n_bits': args.nbits_w, 'channel_wise': args.channel_wise}
    aq_params = {'n_bits': args.nbits_a, 'channel_wise': False}
    print_log(f'quantization paramaters for weights: {wq_params}')
    print_log(f'quantization paramaters for activations: {aq_params}')
    q_model = QuantModel(model=fp_model, dataset_name=dataset_name, weight_quant_params=wq_params, act_quant_params=aq_params).cuda()

    # load the quantized model
    q_model.prepare_quantization_aware_training()
    q_model.load_state_dict(torch.load(args.pretrained_weights, map_location='cuda', weights_only=True))
    q_model.switch_train_eval_mode(train_mode=False)
    print_log(f'q_model (loaded quantized): (acc@1, acc@5, loss) = {test_model_nolambda(model=q_model, test_loader=test_loader)}')

    # add gamma parameters into the model
    num_max_discard_cols = 8
    num_init_discard_cols = 4
    print_log(f'Max discard cols: {num_max_discard_cols}, Initial discard cols: {num_init_discard_cols}')
    q_model.prepare_trainappmult(use_homogeneous_appmult=True, num_max_discard_cols=num_max_discard_cols, num_init_discard_cols=num_init_discard_cols)

    # obtain #macs per layer
    dummy_input, _ = next(iter(train_loader))
    q_model.compute_macs(dummy_input.cuda())

    # collect indicators
    q_model.set_fixed_appmult(use_fixed_appmult=True)
    q_model.switch_train_eval_mode(train_mode=False)
    print_log(f'q_model (before phase1): (acc@1, loss1, loss2, total_loss) = {test_model(model=q_model, test_loader=test_loader, lambd=args.lambd)}')

    # retraining strategy
    optimizer = torch.optim.SGD(q_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * 2)
    print_log(f'optimizer: {optimizer}')
    # print_log(f'scheduler: {scheduler}')

    # train (with DSE)
    q_model.set_fixed_appmult(use_fixed_appmult=False)
    t_begin = time.time()
    for epoch in range(args.epochs):
        print_log(f'AppTrain Epoch {epoch}: Learning rate: {optimizer.param_groups[0]["lr"]}')
        
        q_model.switch_train_eval_mode(train_mode=True)
        train_model(q_model, train_loader, optimizer, epoch, lambd=args.lambd, tensorboard_writer=writer)

        q_model.switch_train_eval_mode(train_mode=False)
        print_log(f'(acc@1, loss1, loss2, total_loss) = {test_model(q_model, test_loader, lambd=args.lambd, epoch=epoch, tensorboard_writer=writer)}')

        # scheduler.step()
        report_time_and_speed(t_begin, epoch, args.epochs, len(train_loader))

    # save stage 1 model
    model_filename = args.log.replace('.log', '.pth')
    print_log(f'saving phase 1 model to {model_filename} ...')
    torch.save({
        "model": q_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        # "scheduler": scheduler.state_dict(),
        "epoch": epoch,
    }, model_filename)


if __name__ == '__main__':
    main()