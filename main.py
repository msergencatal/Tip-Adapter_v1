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
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    
    affinity = val_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, val_labels)
    print("**** Tip-Adapter's val accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights)


    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP
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
def run_tip_adapter_F(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, clip_model, train_loader_F):
    
    # Enable the cached keys to be learnable
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(cache_keys.t())
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            affinity = adapter(image_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha

            loss = F.cross_entropy(tip_logits, target)

            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Eval
        adapter.eval()

        affinity = adapter(test_features)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        clip_logits = 100. * test_features @ clip_weights
        tip_logits = clip_logits + cache_logits * alpha
        acc = cls_acc(tip_logits, test_labels)

        print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(adapter.weight, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    
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
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # Prepare dataset
    random.seed(1)
    torch.manual_seed(1)
    
    print("Preparing dataset.")
    dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])

    val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)

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
           

if __name__ == '__main__':
    main()
