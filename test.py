import os
import sys
sys.path.append(os.path.abspath('.'))
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN, MNIST
from torchvision import transforms
from models.resnet50off import CNN, Discriminator
from core.trainer import train_target_cnn, validate
from utils.utils import get_logger
from utils.altutils import get_office, get_ds
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # NN
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--trained', type=str, default='')
    parser.add_argument('--slope', type=float, default=0.2)
    parser.add_argument('--model', type=str, default='default')
    # train
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--d_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=2.5e-5)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--betas', type=float, nargs='+', default=(.5, .999))
    # misc
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--logdir', type=str, default='outputs/garbage')
    # office dataset categories
    parser.add_argument('--src_cat', type=str, default='amazon')
    parser.add_argument('--tgt_cat', type=str, default='webcam')
    parser.add_argument('--message', '-m',  type=str, default='')
    args, unknown = parser.parse_known_args()

    dataset_root = os.environ["DATASETDIR"]
    test_loader = get_ds(os.path.join(dataset_root, "val"), 32)

    target_cnn = CNN(in_channels=3, target=True).to("cuda:0")
    c = torch.load(args.trained)
    target_cnn.load_state_dict(c['model'])
    criterion = nn.CrossEntropyLoss()
    validation = validate(
                target_cnn, test_loader, criterion, args=args)

    print(f"Accuracy: {validation['acc']}")