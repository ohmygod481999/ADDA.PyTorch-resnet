import os
import sys
sys.path.append(os.path.abspath('.'))
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN, MNIST
from torchvision import transforms
from models.resnet50off import CNN, Discriminator
from core.trainer import train_target_cnn
from utils.utils import get_logger
from utils.altutils import get_office, get_ds


def run(args):
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    logger = get_logger(os.path.join(args.logdir, 'main.log'))
    logger.info(args)

    # data loaders
    dataset_root = os.environ["DATASETDIR"]
    source_loader = get_ds(os.path.join(dataset_root, "train"), args.batch_size)
    target_loader = get_ds(os.path.join(dataset_root, "test"), args.batch_size)

    # train source CNN
    source_cnn = CNN(in_channels=args.in_channels).to(args.device)
    if os.path.isfile(args.trained):
        c = torch.load(args.trained)
        source_cnn.load_state_dict(c['model'])
        logger.info('Loaded `{}`'.format(args.trained))

    # train target CNN
    target_cnn = CNN(in_channels=args.in_channels, target=True).to(args.device)
    target_cnn.load_state_dict(source_cnn.state_dict())
    for param in source_cnn.parameters():
        param.requires_grad = False
    for param in target_cnn.classifier.parameters():
        param.requires_grad = False
    discriminator = Discriminator(args=args).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        target_cnn.encoder.parameters(),
        lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
    d_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=args.d_lr, betas=args.betas, weight_decay=args.weight_decay)
    train_target_cnn(
        source_cnn, target_cnn, discriminator,
        criterion, optimizer, d_optimizer,
        source_loader, target_loader, target_loader,
        args=args)
