import argparse
import os
import torch
import logging
import sys
import time

from torch.utils.tensorboard import SummaryWriter

import utils.datasets as datasets

import models.cifar10
import models.imagenet
from conf import settings
print_log = settings.LOGGER.info
from utils.common import set_seed, train_model, test_model, report_time_and_speed
from quant.quant_model import QuantModel


def parse_args():
    parser = argparse.ArgumentParser(description='running parameters', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general parameters for data and model
    parser.add_argument('--seed', default=1005, type=int, help='random seed for results reproduction')
    # parser.add_argument('--pretrained_weights', default='./cifar10_models/state_dicts/vgg19_bn.pt', type=str, help='path to FP32 pretrained weights')
    parser.add_argument('--pretrained_weights', default='./state_dicts/cifar10/resnet18.pt', type=str, help='path to FP32 pretrained weights')
    # parser.add_argument('--pretrained_weights', default='./cifar10_models/state_dicts/resnet50.pt', type=str, help='path to FP32 pretrained weights')
    parser.add_argument('--num_workers', default=16, type=int, help='number of workers for data loader')

    # quantization parameters
    parser.add_argument('--channel_wise', action='store_true', help='apply channel_wise quantization for weights')
    parser.add_argument('--nbits_w', default=8, type=int, help='bitwidth for weight quantization')
    parser.add_argument('--nbits_a', default=8, type=int, help='bitwidth for activation quantization')

    # calibration parameters
    parser.add_argument('--num_samples', default=1024, type=int, help='size of the calibration dataset')

    # retraining parameters
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate for retraining')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for retraining')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay for retraining')
    parser.add_argument('--epochs', default=60, type=int, help='number of epochs for retraining')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')

    # log file path
    parser.add_argument('--log', default='', type=str, help='path to log file')

    return parser.parse_args()


def get_calibration_data(train_loader, num_samples):
    train_data = []
    for batch in train_loader:
        train_data.append(batch[0])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples]


def main():
    # parse arguments
    args = parse_args()

    # initialize logger
    if args.log != '':
        print(f'log file: {args.log}')
        os.makedirs(os.path.dirname(args.log), exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename=args.log, filemode='w')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', stream=sys.stdout)
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

    # load data
    if args.pretrained_weights.find('cifar10') != -1:
        dataset_name, data_path = 'cifar10', settings.DATASET_PATHS['cifar10']
        print_log(f'loading dataset from {data_path}...')
        train_loader, test_loader = datasets.cifar10.build_cifar10_data(batch_size=args.batch_size, workers=args.num_workers, data_path=data_path)
        if args.pretrained_weights.find('vgg19_bn') != -1:
            fp_model, model_name = models.cifar10.vgg.vgg19_bn(), 'vgg19_bn'
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
    elif args.pretrained_weights.find('imagenet') != -1:
        dataset_name, data_path = 'imagenet', settings.DATASET_PATHS['imagenet']
        t_begin = time.time()
        print_log(f'loading dataset from {data_path}...')
        train_loader, test_loader = datasets.imagenet.build_imagenet_data(batch_size=args.batch_size, workers=args.num_workers, data_path=data_path)
        print_log(f'loading dataset time: {time.time() - t_begin:.2f}s')
        if args.pretrained_weights.find('resnet18') != -1:
            fp_model, model_name = models.imagenet.resnet.resnet18(), 'resnet18'
        elif args.pretrained_weights.find('resnet50') != -1:
            fp_model, model_name = models.imagenet.resnet.resnet50(), 'resnet50'
        elif args.pretrained_weights.find('mnasnet') != -1:
            fp_model, model_name = models.imagenet.mnasnet.mnasnet(), 'mnasnet'
        else:
            raise ValueError(f'Unknown model: {args.pretrained_weights}')
    else:
        raise ValueError(f'Unknown dataset: {args.data_path}')

    # load model 
    fp_model.load_state_dict(torch.load(args.pretrained_weights, weights_only=True, map_location='cuda'))
    # print_log(f'fp_model: {fp_model}')
    print_log(f'fp_model: (acc@1, acc@5, loss) = {test_model(model=fp_model, test_loader=test_loader)}')

    # quantize model
    wq_params = {'n_bits': args.nbits_w, 'channel_wise': args.channel_wise}
    aq_params = {'n_bits': args.nbits_a, 'channel_wise': False}
    print_log(f'quantization paramaters for weights: {wq_params}')
    print_log(f'quantization paramaters for activations: {aq_params}')
    q_model = QuantModel(model=fp_model, dataset_name=dataset_name, weight_quant_params=wq_params, act_quant_params=aq_params).cuda()

    # post-training quantization
    cali_data = get_calibration_data(train_loader, args.num_samples)
    q_model.prepare_post_training_quantization()
    # print_log(f'current q_model: {q_model}')
    print_log(f'calibrating...')
    q_model.switch_train_eval_mode(train_mode=True)
    with torch.no_grad():
        _ = q_model(cali_data.cuda())
    q_model.switch_train_eval_mode(train_mode=False)
    print_log(f'q_model after ptq: (acc@1, acc@5, loss) = {test_model(q_model, test_loader)}')

    # prepare for quantization-aware training
    q_model.prepare_quantization_aware_training()
    # print_log(f'current q_model: {q_model}')
    q_model.switch_train_eval_mode(train_mode=False)
    print_log(f'q_model after ptq: (acc@1, acc@5, loss) = {test_model(q_model, test_loader)}')

    # save the quantized model (for w8a8, no retraining is needed)
    if args.nbits_w == 8 and args.nbits_a == 8:
        os.makedirs(os.path.dirname('state_dicts/'), exist_ok=True)
        torch.save(q_model.state_dict(), f'state_dicts/{dataset_name}/{model_name}_w{args.nbits_w}a{args.nbits_a}_ptq.pth')
        return
    
    # hyperparameters for retraining
    optimizer = torch.optim.SGD(q_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print_log(f'optimizer: {optimizer}')
    print_log(f'scheduler: {scheduler}')

    # train
    t_begin = time.time()
    for epoch in range(args.epochs):
        print_log(f'AppTrain Epoch {epoch}: Learning rate: {optimizer.param_groups[0]["lr"]}')
        q_model.switch_train_eval_mode(train_mode=True)
        train_model(q_model, train_loader, optimizer, epoch, writer)
        q_model.switch_train_eval_mode(train_mode=False)
        print_log(f'(acc@1, acc@5, loss) = {test_model(q_model, test_loader, epoch, writer)}')
        scheduler.step()
        report_time_and_speed(t_begin, epoch, args.epochs, len(train_loader))

    # save the quantized model
    os.makedirs(os.path.dirname('state_dicts/'), exist_ok=True)
    torch.save(q_model.state_dict(), f'state_dicts/{dataset_name}/{model_name}_w{args.nbits_w}a{args.nbits_a}_qat.pth')

    
if __name__ == '__main__':
    main()