import argparse
import torch
import json
import predict_utils

def get_command_line_args():

    parser.set_defaults(gpu=False)
    parser.add_argument ('image_dir', help = 'Provide path to image. Mandatory argument', type = str)
    parser.add_argument ('load_dir', help = 'Provide path to checkpoint. Mandatory argument', type = str)
    parser.add_argument ('--top_k', help = 'Top K most likely classes. Optional', type = int)
    parser.add_argument ('--category_names', help = 'Mapping of categories to real names. JSON file name to be provided. Optional', type = str)
    parser.add_argument ('--GPU', help = "Option to use GPU. Optional", type = str)
    

    return parser.parse_args()


def main():
    
    args = get_command_line_args() # To input arguments
    used_gpu = torch.cuda.is_available() and args.gpu
    print("Input file: {}".format(args.input))
    print("Checkpoint file: {}".format(args.checkpoint))
    if args.top_k:
        print("Returning {} most likely classes".format(args.top_k))
    if args.category_names:
        print("Category names file: {}".format(args.category_names))
    if used_gpu:
        print("Using GPU.")
    else:
        print("Using CPU.")
    
    
    model = predict_utils.load_checkpoint(args.checkpoint) # To load the checkpoint
    print("Checkpoint loaded.")
    
    # To move tensors to GPU
    if used_gpu:
        model.cuda()
    
    # To Load categories file
    if args.category_names:
        with open(args.category_names, 'r') as f:
            categories = json.load(f)
            print("Category names loaded")
    
    results_to_show = args.top_k if args.top_k else 1
    
    
    print("Processing image")
    probabilities, classes = predict_utils.predict(args.input, model, used_gpu, results_to_show, args.top_k) # To Predict
    
    # Show the results
    # Print results
    if results_to_show > 1:
        print("Top {} Classes for '{}':".format(len(classes), args.input))

        if args.category_names:
            print("{:<30} {}".format("Flower", "Probability"))
            print("------------------------------------------")
        else:
            print("{:<10} {}".format("Class", "Probability"))
            print("----------------------")

        for i in range(0, len(classes)):
            if args.category_names:
                print("{:<30} {:.2f}".format(categories[classes[i]], probabilities[i]))
            else:
                print("{:<10} {:.2f}".format(classes[i], probabilities[i]))
    else:
        print("The most likely class is '{}': probability: {:.2f}" \
              .format(categories[classes[0]] if args.category_names else classes[0], probabilities[0]))
        
    
if __name__ == "__main__":
    main()