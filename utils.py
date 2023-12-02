from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

import clip


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model):
    # gradients are not added to the computation graph, which can save memory and improve performance.
    # It's common when working with pre-trained models during evaluation, as you usually don't need to compute gradients for weights during this phase. 
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def build_cache_model(cfg, clip_model, train_loader_cache):
    # create cache keys and values for the first run. load_cache=False in the yaml file. For later running, they can be set as True for faster hyperparameter tuning. 
    if cfg['load_cache'] == False:    
        cache_keys = []        # Initialize an empty list of keys 
        cache_values = []      # initialize an empty list of values
        
        # Perform operations that you don't want to track for gradient computation
        # This is often used during inference or updating weights without computing gradients, such as in evaluation or testing.
        with torch.no_grad():  
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):  # Iterates over the specified number of augmentations (cfg['augment_epoch']).
                train_features = []    # Create an empty list to store image_features

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()     # Move the tensor to the GPU. Now 'images' is on the GPU, and operations involving 'images' will be executed on the GPU.                     
                    image_features = clip_model.encode_image(images)  # Extract image features from the CLIP vision encoder.[1 x 512]
                    train_features.append(image_features)    #The extracted image_features are then appended to the train_features list.[num of total images x 512]
                    if augment_idx == 0:     # This condition executed only the first iteration.
                        target = target.cuda()    # Move the target to the GPU.
                        cache_values.append(target)    #The target appended to the cache value list. 
                # torch.cat(train_features, dim=0)--> Concatenate a list of tensors along a specified dimension.
                # torch.cat(train_features, dim=0).unsqueeze(0)--> Adds a new dimension at the beginning of the tensor.
                # The purpose of adding an extra dimension with unsqueeze(0) might depend on the specific requirements of your code. 
                # It's common to add a batch dimension when dealing with deep learning models, where the first dimension often represents the batch size.
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))    #[num of total images x 512]
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)    #[num of total images x 512]
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)    #[num of total images x 512]
        cache_keys = cache_keys.permute(1, 0)    #[512 x num of total images]
        #F.one_hot(...)--> converts each element in the input tensor to a one-hot vector representation
        # .half()--> converts the data type of the tensor to half-precision floating-point format (float16). 
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()    #[num of total images x N]
        
        # save keys and values to the directory in yaml file
        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")
    # take the values if the cache keys and values were already created before. load_cache=True.
    else:
        # load keys and cache values respectively from the directory(from dataset.yaml)
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")    #[512 x num of total images]
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")     #[num of total images x N]

    return cache_keys, cache_values


def pre_load_features(cfg, split, clip_model, loader):

    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
   
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")
    
    return features, labels


def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None):

    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)
            
                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha
