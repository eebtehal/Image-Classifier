import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from collections import OrderedDict
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

import json

from PIL import Image
from __future__ import print_function, division
import argparse


def load_data(data_dir = "/.flowers"):
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    data_transforms =  {
    'train' : transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(30),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])]),
                                                            
    'valid' : transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])]),

    'test' : transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
             }


    # TODO: Load the datasets with ImageFolder
    image_datasets = { 'train' : datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                       'test' : datasets.ImageFolder(test_dir, transform=data_transforms['test']),
                       'valid' : datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = { 'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
                    'test' : torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=False),
                    'valid' : torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True)
    }
    return dataloaders

def get_model():
    model = models.densenet121(pretrained=True)
    return model

def build_model(hidden_layers, class_to_idx):
    # To geting the model
    model_1 = get_model()
    
    #To building the model
    for param in model_1.parameters():
        param.requires_grad = False
        
    input_size = model_1.classifier.in_features
    print("The input size: ", input_size)
    
    # Modifying the classifier
    model_1.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_layers)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_layers, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model_1.class_to_idx = class_to_idx
    
    return model_1
def train(model, epochs, learning_rate, criterion, optimizer, training_loader, validation_loader):
    
    start_time = time.time()
    
    model.train()
    print_every = 40
    steps = 0
    gpu_used = False
    
    
    if torch.cuda.is_available():
        gpu_used = True
        model.cuda()
    else:
        model.cpu()

    for e in range(epochs):
        running_loss = 0
        for inputs, labels in iter(training_loader):
            steps += 1

            if gpu_used:
                inputs = Variable(inputs.float().cuda())
                labels = Variable(labels.long().cuda()) 
            else:
                inputs = Variable(inputs)
                labels = Variable(labels) 

            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]

            if steps % print_every == 0:
                valid_loss, accuracy = validate(model, criterion, validation_loader)

                print("Epoch: {}/{} ".format(e+1, epochs),
                        "\nTrain loss: {:.4f} ".format(running_loss/print_every),
                        "\nValidation Loss: {:.4f} ".format(valid_loss),
                        "Validation Accuracy: {:.3f}%".format(accuracy*100))
    print("\nTraining process is now complete!!")
    total_time = time.time() - start_time
    print("\n The total time: {:.0f}m {:.0f}s".format(total_time//60, total_time % 60))

def validate(model, criterion, valid_loader):
    model.eval()
    accuracy = 0
    valid_loss = 0
    
    for images, labels in iter(valid_loader):
        if torch.cuda.is_available():
            images,labels = Variable(images.float().cuda(), volatile=True),Variable(labels.long().cuda(), volatile=True)
        else:
            images,labels = Variable(images, volatile=True),Variable(labels, volatile=True)
        
        # Do a forward-pass
        output = model.forward(images)
        
        # Based on the criterion we calculate the loss 
        valid_loss += criterion(output, labels).data[0]
        
        # get the probabilities from the e^x of the output
        ps = torch.exp(output).data 
        
        # Checking the number of output results most likely to match the labels
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()
        
    vali=valid_loss/len(valid_loader), accuracy/len(valid_loader)

    return vali    
  
def save_checkpoint():
    checkpoint_path = 'densenet121_checkpoint.pth'

    checkpoint = {
        'arch': 'densenet121',
        'learning_rate': learning_rate,
        'hidden_layers': hidden_layers,
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'class_to_idx' : model.class_to_idx
    }

    torch.save(checkpoint, checkpoint_path)

    
parser = argeparse.ArgumentParser(description='train.py')

parser.add_argument('data_dir', nargs='*', action="store", default="./flowers")
parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
parser.add_argument('--save_dir', dest="save_dir", action="store", default="classifier.pth")
parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
parser.add_argument('--structure', dest="structure", action="store", default="vgg16", type = str)
parser.add_argument('--hidden_layer', dest="hidden_layer", action]"store", type = int, default = 1024)

data_dir = parse.data_dir
power = parse.gpu
checkpoint_path = parse.save_dir
learning_rate = parse.learning_rate
epochs= parse.epochs
structure = parse.structure
hidden_layer = parse.hidden_layer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
class_to_idx = parse.class_to_idx


dataloaders = load_data(data_dir)
build_model(hidden_layer,class_to_idx)
train_model( model,epochs,learning_rate, criterion, optimizer ,dataloaders['train'], dataloaders['valid'])
save_checkpoint()








