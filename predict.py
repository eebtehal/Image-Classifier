import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image
import json
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import argparse


# Argparse config section

parser = argparse.ArgumentParser(description='Testing a Neural Network with the test sample')
parser.add_argument('--checkpoint_path', type=str,
                    help='path to recover and reload checkpoint',default='checkpoint.pth')
parser.add_argument('--image_path', type=str,
                    help='/path/to/image',default='flowers/test/71/image_04512.jpg')
parser.add_argument('--top_k', type=int,
                    help='top k: top categories by prob predictions',default=5)
parser.add_argument('--cat_to_name', type=str,
                    help='category name mapping',default='cat_to_name.json')
parser.add_argument('--device', type=str,
                    help='Choose -cuda- gpu or internal -cpu-',default='cuda')
parser.add_argument('--network', type=str,
                    help='Torchvision pretrained model. May choose densenet121 too', default='vgg19')
parser.add_argument('data_dir', type=str,
                    help='Path to root data directory', default='/flowers/')

args = parser.parse_args()
checkpoint_path = args.checkpoint_path
image_path = args.image_path
top_k = args.top_k
device = args.device
cat_to_name = args.cat_to_name
network = args.network
data_dir = args.data_dir



# Label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)



# Loading the checkpoint
def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    learning_rate = checkpoint['learning_rate']
    class_to_idx = checkpoint['class_to_idx']

    model = build_model(hidden_layers, class_to_idx)

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    
    '''
    img = Image.open(image)
    W, H = img.size # the W:width and H:height
    size = 224 
    # TODO: Process a PIL image for use in a PyTorch model
    
    if H > W:
        H,W = int(max(H * size / W, 1)),int(size)
    else:
        W,H = int(max(W * size / H, 1)),int(size)
        
        
    x0,y0 = ((W - size) / 2) ,((H - size) / 2)
    x1,y1 = (x0 + size),(y0 + size)
    
    
    resized_image = img.resize((W, H))
    
    # Crop
    cropped_image = img.crop((x0, y0, x1, y1))
    
    # Normalize
    np_image = np.array(cropped_image) / 255.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])     
    image_array = (np_image - mean) / std
    
    
    #As expected by PyTorch move the color channels to the first dimension

    image_array = np_image.transpose((2, 0, 1))
    
    return image_array
    # TODO: Process a PIL image for use in a PyTorch model

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    # Processing image
    model.eval()
    np_array = process_image(image_path)
    
    # Numpy -> Tensor
    tensor = torch.from_numpy(np_array) # tranfer to tensor
    
    use_gpu = False 
    
    if torch.cuda.is_available():
        use_gpu = True
        model = model.cuda()
        var_inputs = Variable(tensor.float().cuda(), volatile=True)
    else:
        model = model.cpu()
        var_inputs = Variable(tensor, volatile=True)
    np_array = process_image(image_path)
    
    
    # Adding batch of size 1 to image
    var_inputs = var_inputs.unsqueeze(0)
    output = model.forward(var_inputs)
    
    # Probs
    ps = torch.exp(output).data.topk(topk)
    
    # Top probs and Convert indices to classes
    
    probabilities = ps[0].cpu() if use_gpu else ps[0]
    classes = ps[1].cpu() if use_gpu else ps[1]
    class_to_idx_inverted = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()
    for label in classes.numpy()[0]:
        mapped_classes.append(class_to_idx_inverted[label])
    return probabilities.numpy()[0], mapped_classes


def main():
    model = load_checkpoint(checkpoint_path)
    image = process_image(image_path)
    probs, classes_list = predict(image_path, model)
    

    print("Class names:", classes_list)
    print("Probability:", probs)

    
if __name__== "__main__":
    main()
