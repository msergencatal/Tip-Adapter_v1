# Imports several standard Python libraries and third-party modules. 
# 'os' is a module that provides a way of interacting with the operating system. It offers functions for file and directory manipulation, environment variables, and more.
import os
# 'random' is a module for generating pseudorandom numbers. It provides functions for various randomization tasks.
import random
# 'argparse' is a module for parsing command-line arguments in a user-friendly way. It allows you to define command-line interfaces for your scripts.
import argparse
# 'yaml' is a module for working with YAML (YAML Ain't Markup Language) files. YAML is a human-readable data serialization format.
import yaml
# 'tqdm' is a module for displaying progress bars in the terminal. Importing the tqdm module and specifically the tqdm class from it.
from tqdm import tqdm

# Imports some fundamental PyTorch modules and classes
# 'torch' is the main PyTorch module. It provides the fundamental data structures (tensors) and operations for building and training neural networks.
import torch
# 'torch.nn.functional' (imported as 'F') contains various functions that are not part of the 'torch.nn' module but are still essential for neural network operations. 
# Common functions like activation functions (e.g., relu, sigmoid) are found here.
import torch.nn.functional as F
# 'torch.nn' is PyTorch's neural network module. It includes classes for defining neural network architectures, layers, loss functions, and more.
import torch.nn as nn
# 'torchvision.transforms' provides a set of image transformations commonly used in computer vision tasks. 
# These transformations can be applied to images before feeding them into a neural network
import torchvision.transforms as transforms

# Imports the build_dataset function from the datasets/_init.py_ . 
from datasets import build_dataset
# Imports the build_data_loader function from the utils submodule(utils.py) within the datasets module.
from datasets.utils import build_data_loader
# Imports  the CLIP module
import clip
# imports all names (functions, classes, variables) from the utils.py
from utils import *

# function definition of 'get_arguments()'
def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()

    return args

# function definition of 'run_tip_adapter()'
# PARAMETERS
# cfg--> configurations of each datasets from configs file.(eurosat.yaml, food101.yaml ...)
# cache_keys--> Ftrain (CLIP vision encoder output of cache model)
# cache_values--> Ltrain (one hot encoded labels of cache model))
# val_features--> visual features from the validation set for hyperparameter searching.
# val_labels--> labels from the validation set for hyperparameter searching.
# test_features--> visual features from the test set (ftest)
# test_labels--> labels from the test set
# clip_weights--> CLIP textual features (Wc)
def run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights):
    
    print("\n-------- Searching hyperparameters on the val set. --------")

    # Zero-shot CLIP
    # calculates the logits using CLIP, computes the classification accuracy on validation data, and prints the accuracy.
    # calculates the dot product between 'val_features' and the transpose of 'clip_weights'. The result is scaled by 100.
    # 'clip_logits' (the predicted logits) and 'val_labels' (the true labels)
    clip_logits = 100. * val_features @ clip_weights
    acc = cls_acc(clip_logits, val_labels)
    print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    # take beta and alpha from cfg
    beta, alpha = cfg['init_beta'], cfg['init_alpha']

    # visual features*Ftrain
    affinity = val_features @ cache_keys
    # activation function of tip-adapter
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values        

    # Tip-adapter output logits
    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, val_labels)
    print("**** Tip-Adapter's val accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights)


    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP
    # calculate accuracy  with best_beta and best_alpha on test set
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter    
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))

# function definition of 'run_tip_adapter_F()'
# clip_model--> pre-trained CLIP model
# train_loader_F--> data from the training set
def run_tip_adapter_F(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, clip_model, train_loader_F):
    
    # Enable the cached keys to be learnable
    # `cache_keys` is a PyTorch tensor    [1024 x num of images]
    # The `nn.Linear` creates a linear layer with the specified input and output sizes.
    # - `cache_keys.shape[0]` is the input size (1024)
    # - `cache_keys.shape[1]` is the output size (number of images)
    # - `bias=False` means that the linear layer will not have a bias term.
    # .to(clip_model.dtype) sets the data type of the layer to match the data type of `clip_model`.
    # .cuda() moves the layer to the GPU (assuming you have a GPU available).
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(cache_keys.t())

    # The optimizer is created using the AdamW algorithm, which is an extension of Adam with weight decay.
    # adapter.parameters() --> provide the parameters (weights) of the adapter linear layer to the optimizer. 
    # This is necessary so that the optimizer knows which parameters to update during the optimization process.
    # lr=cfg['lr'] --> sets the learning rate for the optimizer. The learning rate is a hyperparameter that controls the step size during optimization.
    # eps=1e-4 --> is the epsilon parameter, a small value added to the denominator for numerical stability.
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    
    # optimizer --> is the AdamW optimizer that you created earlier.
    # train_epoch --> is the number of training epochs specified in the configuration.
    # len(train_loader_F) --> is the total number of batches in your training data loader for a single epoch.
    # The Cosine Annealing LR scheduler gradually reduces the learning rate in a cosine-shaped manner over the specified number of total training steps.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))

    # initialize the beta, alfa, best_acc and best_epoch.
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(cfg['train_epoch']):
        # Train
        # The model is set to training mode.
        adapter.train()
        # Variables for holding the num of correct samples and all samples.
        correct_samples, all_samples = 0, 0
        # Variable for holding the losses.
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                # Construct CLIP image features.
                image_features = clip_model.encode_image(images)
                # Normalize features with L2 norm.
                image_features /= image_features.norm(dim=-1, keepdim=True)

            # Calculate the affinity score with the trainable adapter.
            affinity = adapter(image_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha

            # calculate the cross-entropy loss between the tip_logits and the target labels. 
            loss = F.cross_entropy(tip_logits, target)
            # Calculate accuracy.
            acc = cls_acc(tip_logits, target)
            # Num of correct samples.
            correct_samples += acc / 100 * len(tip_logits)
            # Num of all samples.
            all_samples += len(tip_logits)
            # This is commonly done in training loops to keep track of the losses for each iteration (or batch) during an epoch.
            loss_list.append(loss.item())

            # Zeroing out the gradients ensures that the gradients from the previous batch do not accumulate and  
            # affect the parameter updates for the current batch, preventing unwanted interference between batches.
            optimizer.zero_grad()
            loss.backward()    #  Compute gradients with respect to the parameters.
            optimizer.step()    # Update the model parameters based on the computed gradients.
            scheduler.step()    # Update the learning rate schedule during training.

        # Retrieve the current learning rate from the learning rate scheduler.
        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Eval
        # Set the adapter model in evaluation mode.
        adapter.eval()

        # Calculate the accuracy
        affinity = adapter(test_features)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        clip_logits = 100. * test_features @ clip_weights
        tip_logits = clip_logits + cache_logits * alpha
        acc = cls_acc(tip_logits, test_labels)

        print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
        # Find the highest accuracy and save the weights for this accuracy value.
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(adapter.weight, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
        # end of one epoch.

    # Load back the weight for the best accuracy.
    adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    print(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    print("\n-------- Searching hyperparameters on the val set. --------")

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights, adapter=adapter)

    print("\n-------- Evaluating on the test set. --------")
   
    affinity = adapter(test_features)
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(max(best_acc, acc)))


def main():

    # Load config file
    args = get_arguments()
    # Check if the specified configuration file exists
    assert (os.path.exists(args.config))
    
    # open(args.config, 'r'): This part opens the file specified in args.config in read mode ('r'). It returns a file object.
    # yaml.load(..., Loader=yaml.Loader): This uses the yaml library to load the content of the opened file. The Loader=yaml.
    # Loader parameter specifies the YAML loader to use. The yaml.Loader is part of the PyYAML library and is used for loading YAML data.
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # If cfg['dataset'] is, for instance, 'my_dataset', then cache_dir would be './caches/my_dataset'.
    cache_dir = os.path.join('./caches', cfg['dataset'])
    # This code checks if the directory specified by cache_dir exists. If it does not exist, it creates it and any necessary parent directories. 
    # If it already exists, it does nothing, thanks to exist_ok=True. 
    os.makedirs(cache_dir, exist_ok=True)
    #  assigns the previously constructed cache_dir to the 'cache_dir' key in the configuration dictionary (cfg).
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # Prepare dataset
    # Subsequent calls to random functions in your script will produce the same sequence of random numbers every time you run the script.
    random.seed(1)
    torch.manual_seed(1)
    
    print("Preparing dataset.")
    dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])

    val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)

    # transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC): Randomly crops and resizes the input image. 
    # The size is set to 224 pixels, and the crop scale is randomly chosen between 0.5 and 1. Bicubic interpolation is used for resizing.
    # transforms.RandomHorizontalFlip(p=0.5): Randomly flips the input image horizontally with a probability of 0.5.
    # transforms.ToTensor(): Converts the input image to a PyTorch tensor. This is a necessary step before feeding the image into a deep learning model.
    # transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)): 
    # Normalizes the tensor by subtracting the mean and dividing by the standard deviation. 
    # The specified mean and std values are often precomputed based on the dataset statistics to standardize the input data.
    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)
    train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=True)

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

    # ------------------------------------------ Tip-Adapter ------------------------------------------
    run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights)

    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
    run_tip_adapter_F(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, clip_model, train_loader_F)
           
# This means that the main() function will only be called if the script is executed directly. 
# If the script is imported as a module elsewhere, the main() function won't be automatically executed.
if __name__ == '__main__':
    main()
