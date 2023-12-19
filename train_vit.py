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
import random

def get_timeString():

    time_stamp = time.time() 
    struct_time = time.localtime(time_stamp)
    timeString = time.strftime("%Y-%m-%d %H:%M:%S", struct_time)

    return timeString

def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# %%
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate trained RL model on MNIST dataset.')
    parser.add_argument('--dataset', dest='dataset')
    parser.add_argument('--random_seed', default=False, dest='random_seed')
    parser.add_argument('--slight_disturbance', action="store_true", dest='slight_disturbance')
    parameter_args = parser.parse_args()

    dataset_name = parameter_args.dataset
    if type(parameter_args.random_seed) == str:
        random_seed = int(parameter_args.random_seed)
    elif parameter_args.random_seed == False:
        random_seed = False
    folder_path = f'../..'
    slight_disturbance = parameter_args.slight_disturbance
    print(slight_disturbance)

    # %%

    # dataset_name = 'Mnist' # Mnist Fashion Cifar10  
    # random_seed = 1
    if random_seed == False: 
        random_seed = int(time.time())
        time.sleep(1)

    set_seed(random_seed)
    device = torch.device("cuda")

    NUM_EPOCHES = 300 # 25
    batch_size = 128

    image_size = 32 if dataset_name == 'Cifar10' else 28
    channel_size = 3 if dataset_name == 'Cifar10' else 1
    patch_size = 8  if dataset_name == 'Cifar10' else 7
    valid_size = 1/6  if dataset_name == 'Mnist' else 0.1
    LR = 5e-5 if dataset_name == 'Mnist' else 3e-4
    embed_size = 256
    num_heads = 8
    classes = 10
    num_layers = 1
    hidden_size = 256
    dropout = 0.2

    record_path = f'{folder_path}/records/ViT/{dataset_name}'
    model_path = f'{folder_path}/models/ViT/{dataset_name}'

    if not os.path.isdir(record_path):
        os.makedirs(record_path)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    record_txt = f'{record_path}/record.txt' # {get_timeString()}
    best_model = f'{model_path}/{random_seed}.pt'
    print(record_txt)

    # %%
    # Dataset
    data_dir = f'../data/{dataset_name}/'
    transform = transforms.Compose([transforms.ToTensor()])

    if dataset_name == 'Mnist':
        dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    elif dataset_name == "Fashion":
        dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    elif dataset_name == "Cifar10":
        dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    validloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=4)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # %%
    # for t in range(train_times):

    model = ViT(image_size, channel_size, patch_size, embed_size, num_heads, classes, num_layers, hidden_size, dropout=dropout).to(device)
    # print(sum([param.nelement() for param in model.parameters()]))

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
    time_String = get_timeString()

    train_list = []
    valid_list = []
    best_acc = 0 

    for epoch in range(1, NUM_EPOCHES+1):
        model.train()
        
        epoch_train_loss = 0
            
        y_true_train = []
        y_pred_train = []
            
        for batch_idx, (img, labels) in enumerate(trainloader):
            img = img.to(device)
            labels = labels.to(device)
            
            ########### shift test ###########
            if slight_disturbance:
                batsh_s, channels, width, height = img.shape
                shift_unit = 1
                s_u = shift_unit * 2
                copy_tensor = torch.zeros((batsh_s, channels, width+s_u, height+s_u)).to('cuda')
                init_pos = 1
                ver_pos = shift_unit + random.randint(-1, 1)
                her_pos = shift_unit + random.randint(-1, 1)
                copy_tensor[:, :, ver_pos: ver_pos+width, her_pos: her_pos+height] = img
                img = copy_tensor[:, :, init_pos: init_pos+width, init_pos: init_pos+height]
            ########### shift test ###########
            
            preds = model(img)
            
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            y_pred_train.extend(preds.detach().argmax(dim=-1).tolist())
            y_true_train.extend(labels.detach().tolist())
                
            epoch_train_loss += loss.item()
        
        
        total_correct = len([True for x, y in zip(y_pred_train, y_true_train) if x==y])
        total = len(y_pred_train)
        accuracy = total_correct * 100 / total
        
        accuracy = round(accuracy, 2)
        train_list.append(accuracy)

        with torch.no_grad():
            model.eval()
            
            y_true_test = []
            y_pred_test = []
            
            for batch_idx, (img, labels) in enumerate(validloader):
                img = img.to(device)
                labels = labels.to(device)
            
                preds = model(img)
                
                y_pred_test.extend(preds.detach().argmax(dim=-1).tolist())
                y_true_test.extend(labels.detach().tolist())
                
            total_correct = len([True for x, y in zip(y_pred_test, y_true_test) if x==y])
            total = len(y_pred_test)
            valid_accuracy = total_correct * 100 / total
            valid_accuracy = round(valid_accuracy, 2)
            
        valid_list.append(valid_accuracy)

        if valid_accuracy > best_acc:
            best_acc = valid_accuracy
            torch.save(model, best_model)

        print(f"Epoch: {epoch} | Train acc {accuracy} | Valid acc: {valid_accuracy}")

    model = torch.load(best_model)

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

    valid_info = f"Valid max acc: {max(valid_list)} at {valid_list.index(max(valid_list))}"
    info = f'Time: {time_String} | {valid_info} | Test acc: {test_accuracy} | Random seed: {random_seed}'
    print(info)

    with open(record_txt, "a+") as outfile:
        outfile.writelines(info)
        outfile.writelines('\n')

    # %%


    # %%
    


