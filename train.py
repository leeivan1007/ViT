# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DataLoader

from torchvision import datasets, transforms, models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.utils import save_image

from torchsummary import summary

# import spacy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import time
import math
from PIL import Image
import glob
from IPython.display import display
from transformer_package.models import ViT
from torch.utils.data.sampler import SubsetRandomSampler
import argparse

def get_timeString():

    time_stamp = time.time() 
    struct_time = time.localtime(time_stamp)
    timeString = time.strftime("%Y-%m-%d %H:%M:%S", struct_time)

    return timeString

    # %%
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate trained RL model on MNIST dataset.')
    parser.add_argument('--dataset', dest='dataset')
    parameter_args = parser.parse_args()

    dataset_name = parameter_args.dataset

    # %%

    # dataset_name = 'Mnist' # Mnist Fashion Cifar10  

    device = torch.device("cuda")
    torch.manual_seed(0)
    np.random.seed(0)

    NUM_EPOCHES = 5
    train_times = 1

    random_seed = 1
    batch_size = 128

    image_size = 32 if dataset_name == 'Cifar10' else 28
    channel_size = 3 if dataset_name == 'Cifar10' else 1
    patch_size = 8  if dataset_name == 'Cifar10' else 7
    LR = 5e-5 if dataset_name == 'Mnist' else 3e-4
    embed_size = 256
    num_heads = 8
    classes = 10
    num_layers = 1
    hidden_size = 256
    dropout = 0.2

    record_path = f'records/ViT/{dataset_name}'
    if not os.path.isdir(record_path):
        os.makedirs(record_path)

    record_txt = f'{record_path}/{get_timeString()}.txt'

    with open(record_txt, "a+") as outfile:
        outfile.writelines(f'Dataset: {dataset_name}')
        outfile.writelines('\n')

    # %%
    # Dataset
    data_dir = f'../data/{dataset_name}/'
    transform = transforms.Compose([transforms.ToTensor()])

    if dataset_name == 'Mnist':
        dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        valid_size = 1/6
    elif dataset_name == "Fashion":
        dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        valid_size = 0.1
    elif dataset_name == "Cifar10":
        dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        valid_size = 0.1

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4,)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=4,)

    # %%
    for t in range(train_times):

        model = ViT(image_size, channel_size, patch_size, embed_size, num_heads, classes, num_layers, hidden_size, dropout=dropout).to(device)
        # print(sum([param.nelement() for param in model.parameters()]))

        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)

        loss_hist = {}
        loss_hist["train accuracy"] = []
        loss_hist["train loss"] = []
        test_list = []
        train_list = []

        for epoch in range(1, NUM_EPOCHES+1):
            model.train()
            
            epoch_train_loss = 0
                
            y_true_train = []
            y_pred_train = []
                
            for batch_idx, (img, labels) in enumerate(trainloader):
                img = img.to(device)
                labels = labels.to(device)
                
                preds = model(img)
                
                loss = criterion(preds, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                y_pred_train.extend(preds.detach().argmax(dim=-1).tolist())
                y_true_train.extend(labels.detach().tolist())
                    
                epoch_train_loss += loss.item()
            
            loss_hist["train loss"].append(epoch_train_loss)
            
            total_correct = len([True for x, y in zip(y_pred_train, y_true_train) if x==y])
            total = len(y_pred_train)
            accuracy = total_correct * 100 / total
            
            accuracy = round(accuracy, 2)
            loss_hist["train accuracy"].append(accuracy)
            train_list.append(accuracy)

            with torch.no_grad():
                model.eval()
                
                y_true_test = []
                y_pred_test = []
                
                for batch_idx, (img, labels) in enumerate(testloader):
                    img = img.to(device)
                    labels = labels.to(device)
                
                    preds = model(img)
                    
                    y_pred_test.extend(preds.detach().argmax(dim=-1).tolist())
                    y_true_test.extend(labels.detach().tolist())
                    
                total_correct = len([True for x, y in zip(y_pred_test, y_true_test) if x==y])
                total = len(y_pred_test)
                test_accuracy = total_correct * 100 / total
                test_accuracy = round(test_accuracy, 2)
                
            test_list.append(test_accuracy)

            # print(f"Epoch: {epoch} | Train acc {accuracy} | Test acc: {test_accuracy}")
        train_info = f"Best train acc: {max(train_list)} at {train_list.index(max(train_list))}"
        test_info = f"Best test acc: {max(test_list)} at {test_list.index(max(test_list))}"
        info = f'T: {t+1} | {train_info} | {test_info}'
        print(info)

        with open(record_txt, "a+") as outfile:
            outfile.writelines(info)
            outfile.writelines('\n')

    # %%



