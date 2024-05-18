import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import ImageFile
from torch.optim import lr_scheduler
from torch.utils.data import Subset
from torchvision import models
from torchvision import transforms
from tqdm import *

from VisualMixer import AttentionBlock, rank, vfe

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE_BOX_SIZE = 14
VFE_BOX_SIZE = 14
IMAGE_SIZE_H = BASE_BOX_SIZE * 8
IMAGE_SIZE_W = BASE_BOX_SIZE * 8


def topk_accuracy(output, targets, k):
    values, indices = torch.topk(output, k=k, sorted=True)
    targets = torch.reshape(targets, [-1, 1])
    correct = (targets == indices) * 1
    top_k_accuracy = torch.sum(torch.max(correct, dim=1)[0])
    return top_k_accuracy


def fft(img):
    fft_image = np.fft.fft2(img)
    shifted_fft_image = np.fft.fftshift(fft_image)
    center_y, center_x = shifted_fft_image.shape[0] // 2, shifted_fft_image.shape[1] // 2
    size = 50
    region = shifted_fft_image[center_y - size // 2:center_y + size // 2, center_x - size // 2:center_x + size // 2]
    high_freq_magnitude = np.mean(np.abs(region))

    return high_freq_magnitude


def call_vfe(inputs):
    # print(f"Start calc vfe")
    v, _ = vfe(inputs, VFE_BOX_SIZE, VFE_BOX_SIZE)
    # print(f"Calc vfe: {v}")
    return v


def train(trainloader, model, rank_module, device, optimizer, schedule, criterion):
    train_acc = 0
    train_loss = 0
    model.train()
    rank_module.train()
    loop = tqdm((trainloader), total=(len(trainloader)))
    train_sample_num = 0
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i, (inputs, targets) in enumerate(loop):
            train_sample_num += inputs.shape[0]
            inputs = inputs.to(device)
            target = targets.to(device)
            inputs, channel_w, spatial_w = rank_module(inputs)
            d_inputs = inputs.detach()
            arg = [d_inputs[j] for j in range(inputs.shape[0])]
            v = pool.map(call_vfe, arg)
            v = [result for result in v]
            for j in range(inputs.shape[0]):
                inputs[j:, :, :] = rank(inputs[j:, :, :], channel_w, spatial_w, VFE_BOX_SIZE, v[j], [0.5, 1])
            output = model(inputs)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num = topk_accuracy(output, target, 1)
            train_loss += loss.item()
            train_acc += num.item()
            loop.set_description(f'Epoch:train [{i}/{len(trainloader)}]')
            loop.set_postfix(loss=train_loss / train_sample_num, acc=train_acc / train_sample_num)
            schedule.step()


def test(model, rank_module, testloader, best_acc, device, criterion, model_save_dir):
    model.eval()
    rank_module.eval()
    val_sample_num = 0
    loop = tqdm((testloader), total=len(testloader))
    val_loss = 0
    val_acc_num = 0
    val_steps = 0
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i, (inputs, target) in enumerate(loop):
            with torch.no_grad():
                val_sample_num += inputs.shape[0]
                inputs = inputs.to(device)
                target = target.to(device)
                inputs, channel_w, spatial_w = rank_module(inputs)
                d_inputs = inputs.detach()
                arg = [d_inputs[j] for j in range(inputs.shape[0])]
                v = pool.map(call_vfe, arg)
                v = [result for result in v]
                for j in range(inputs.shape[0]):
                    inputs[j:, :, :] = rank(inputs[j:, :, :], channel_w, spatial_w, VFE_BOX_SIZE, v[j], [0.5, 1])
                output = model(inputs)
                acc_num = topk_accuracy(output, target, 1)
                loss = criterion(output, target)
                val_acc_num += acc_num.item()
                val_loss += loss.item()
                val_steps += 1
            loop.set_description(f'Epoch:val [{i}/{len(testloader)}]')
            loop.set_postfix(loss=val_loss / val_steps, acc=val_acc_num / val_sample_num)
    if val_acc_num / val_sample_num > best_acc or best_acc == 0:
        best_acc = val_acc_num / val_sample_num
        torch.save(model.state_dict(), model_save_dir + "final_model.pth")
        torch.save(rank_module.state_dict(), model_save_dir + "rank.pth")
    return best_acc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers",
        default=2,
        type=int,
        metavar="N",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
    )
    parser.add_argument(
        "--batch-size",
        default=128,
        type=int,
        metavar="N",
    )
    parser.add_argument(
        "--evaluate",
        dest="evaluate",
        action="store_true",
    )
    parser.add_argument(
        "--model-save-dir",
        type=str,
        default="checkpoint",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./datas",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "--cls",
        type=int,
        default="1000"
    )

    return parser.parse_args()


def setEnv(args):
    cls = args.cls
    device = args.device
    model = models.resnet50(pretrained=False, num_classes=cls).to(device)
    model.to(device)
    rank_moudle = AttentionBlock(3, 3).to(device)
    data_root = args.data_root
    val_dataset = torchvision.datasets.CIFAR10(data_root, train=False, download=True,
                                               transform=torchvision.transforms.Compose([
                                                   transforms.Resize([IMAGE_SIZE_H, IMAGE_SIZE_W]),
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize(
                                                       mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                               ]))
    train_dataset = torchvision.datasets.CIFAR10(data_root, train=True, download=True,
                                                 transform=torchvision.transforms.Compose([
                                                     transforms.Resize([IMAGE_SIZE_H, IMAGE_SIZE_W]),
                                                     torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize(
                                                         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                 ]))
    pg = [p for p in model.parameters() if p.requires_grad]
    for p in rank_moudle.parameters():
        if p.requires_grad:
            pg.append(p)
    optimizer = optim.Adam(pg, lr=0.1)
    schedule = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.001)

    train_dataset = Subset(train_dataset, range(1000))
    val_dataset = Subset(val_dataset, range(1000))

    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              pin_memory=False,
                                              num_workers=args.workers)
    testloader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             pin_memory=False,
                                             num_workers=args.workers)
    criterion = nn.CrossEntropyLoss()
    return model, trainloader, testloader, criterion, schedule, rank_moudle, optimizer


if __name__ == '__main__':
    args = parse_args()
    model, trainloader, testloader, criterion, schedule, rank_module, optimizer = setEnv(args)
    best_acc = 0

    model.load_state_dict(torch.load('checkpointfinal_model.pth'))
    rank_module.load_state_dict(torch.load('checkpointrank.pth'))
    for i in range(args.epochs):
        train(trainloader, model, rank_module, args.device, optimizer, schedule, criterion)
        test(model, rank_module, testloader, best_acc, args.device, criterion, args.model_save_dir)
