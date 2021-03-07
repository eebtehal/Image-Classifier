import os
import argparse
import torch
from torchvision import datasets, transforms
import model_utils

def get_command_line_args():
    
    # Command Line arguments
    parser = argparse.ArgumentParser(description='Image Classifire')
    #Set directory to save checkpoints:
    
    # working directory
    parser.add_argument('-p', '--work_path', help='Set the working directory', default='/home/workspace/aipnd-project')
    parser.add_argument('-f', '--image_folder', help='Set the image directory', default='flowers')
    parser.add_argument('-t', '--train_folder', help='Set the training directory', default='train')
    parser.add_argument('-v', '--valid_folder', help='Set the validatig directory', default='valid')
    parser.add_argument('-s', '--test_folder', help='Set the testing directory', default='test')
    
    # Model hyperparameters
    
    parser.add_argument('-m', '--model', help='Choose the Model architecture',
                        choices=['vgg11', 'vgg13', 'vgg16', 'vgg19', 'densenet'], default='vgg16')
    parser.add_argument('-l', '--learning_r', type=float, help='Set the learning rate', default=0.003)
    parser.add_argument('-h1', '--h1', type=int, help='Set the headding layer 1', default=4000)
    parser.add_argument('-h2', '--h2', type=int, help='Set the headding layer 2', default=1000)
    parser.add_argument('-o', '--output_size', type=int, help='Set the output size', default=102)
    
    # training
    
    parser.add_argument('-ep', '--epochs', type=int, help='set the number of ephocs', default=10)
    parser.add_argument('-d', '--device', help='set the device of learning cpu or cuda', choices=['cpu', 'cuda'],
                        default='cuda')

    return parser.parse_args()



def checkpoint_save(arch, learning_rate, hidden_units, epochs, save_path, model, optimizer):
    ''' 
    to Save the checkpoint
    '''
    sstate = {
    'arch': 'densenet121',
    'learning_rate': learning_rate,
    'hidden_layers': hidden_layers,
    'epochs': epochs,
    'state_dict': model.state_dict(),
    'optimizer' : optimizer.state_dict(),
    'class_to_idx' : model.class_to_idx
          }

    torch.save(state, save_path)

def main():
    
    
    args = get_command_line_args() # Get Command Line Arguments
    used_gpu = torch.cuda.is_available() and args.gpu
    print("Data directory: {}".format(args.data_dir))
    if used_gpu:
        print("Training on GPU.")
    else:
        print("Training on CPU.")
    print("Architecture: {}".format(args.arch))
    if args.save_dir:
        print("Checkpoint save directory: {}".format(args.save_dir))
    print("Learning rate: {}".format(args.learning_rate))
    print("Hidden units: {}".format(args.hidden_units))
    print("Epochs: {}".format(args.epochs))
    
    
    data_loaders, class_to_idx = model_utils.get_loaders(args.data_dir) # To get loaders
    for key, value in data_loaders.items():
        print("{} data loader retrieved".format(key))
    
    
    model, optimizer, criterion = model_utils.build_model(args.arch, args.hidden_units, args.learning_rate) # To build the model
    model.class_to_idx = class_to_idx
    
    # To check if the GPU availiable and move
    if used_gpu: 
        print("GPU is availaible. Moving Tensors.")
        model.cuda()
        criterion.cuda()
    
    
    model_utils.train_model(model, args.epochs, criterion, optimizer,
                       data_loaders['training'], data_loaders['validation'], used_gpu) # Train the model
    
    # To Save the checkpoint
    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        save_path = args.save_dir + '/' + args.arch + '_checkpoint.pth'
    else:
        save_path = args.arch + '_checkpoint.pth'
    print("Will save checkpoint to {}".format(save_path))

    checkpoint_save(args.arch, args.learning_rate, args.hidden_units, args.epochs, save_path, model, optimizer)
    print("Checkpoint saved")

    
    test_loss, accuracy = model_utils.validate(model, criterion, data_loaders['testing'], used_gpu) # calculate accuracy
    print("Test Loss: {:.3f}".format(test_loss))
    print("Test Acc.: {:.3f}".format(accuracy))
          
if __name__ == "__main__":
    main()
