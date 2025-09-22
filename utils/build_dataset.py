import os
import os.path
import torch
import json
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode 
from timm.data import create_transform
import numpy as np
import random
from torch.utils.data import Subset
from collections import defaultdict
from torchvision import datasets
from collections import defaultdict
from utils.data_handler import FileListDataset, Flowers, ImageFolderWithIndex, CUBDataset, CustomKinetics, CustomGTSRB
from torchvision import transforms as T
from PIL import Image

def random_crop(image, alpha=0.5, beta=0.9):
    """
    Randomly crops an image based on a size range determined by alpha and the image dimensions.

    Args:
        image (PIL Image or Tensor): The input image to crop.
        alpha (float): The minimum scale factor for the crop, relative to the smallest dimension of the image.

    Returns:
        PIL Image or Tensor: Cropped image.
    """
    w, h = image.size
    crop_size = np.random.uniform(low=alpha, high=beta) * min(h, w)  # Crop size based on a random proportion of the smallest dimension
    cropped_image = T.RandomCrop(int(crop_size))(image)  # Apply random crop
    return cropped_image

class DataAugmentation:
    def __init__(self, weak_transform, strong_transform):
        self.transforms = [weak_transform, strong_transform]
        
    def __call__(self, x):
        images_weak = self.transforms[0](x)
        images_strong = self.transforms[1](x)
        return images_weak, images_strong

def build_dataset(is_train, args):
    train_tfm = build_transform_training(args)
    weak_tfm = build_weak_transform(args)
    test_tfm = build_transform_testing(args)
    
    def custom_loader(path):
        img = datasets.folder.default_loader(path)
        augmented_imgs = []
        images_weak, images_strong = train_tfm(img)
        augmented_imgs.append(images_weak)
        if is_train:
            augmented_imgs.append(images_strong) 
        if not args.baseline:
            n_crops = args.n_crops
            augmented_imgs.extend(weak_tfm(random_crop(img, alpha=args.alpha, beta=args.beta)) for _ in range(n_crops))
        # Return a stacked tensor of all processed images
        return torch.stack(augmented_imgs)
    
    train_config_path = os.path.join("./configs", 'dataset_catalog.json')
    with open(train_config_path, 'r') as train_config_file:
        catalog = json.load(train_config_file)
    assert args.dataset in catalog.keys(), "Dataset %s is not implemented"%args.data
    entry = catalog[args.dataset]
    return_index = True
    if entry['type'] == 'imagefolder':
        if args.dataset == "flowers":
            if is_train:
                dataset = Flowers(root=entry['path'], istrain=is_train, loader=custom_loader, return_index=return_index)
            else:
                dataset = Flowers(root=entry['path'], transform=test_tfm, istrain=is_train, return_index=return_index)
        elif args.dataset == "cub":
            if is_train:
                dataset = CUBDataset(root=entry['path'], train=is_train, loader=custom_loader, return_index=return_index)
            else:
                dataset = CUBDataset(root=entry['path'], transform=test_tfm, train=is_train, return_index=return_index)
        elif args.dataset == "kinetics":
            if is_train:
                dataset = CustomKinetics(root=entry['path'], split='train', num_classes="700", return_index=return_index)
            else:
                dataset = CustomKinetics(root=entry['path'], split='test', num_classes="700", transform=test_tfm, return_index=return_index)
        elif args.dataset == "gtsrb":
            if is_train:
                dataset = CustomGTSRB(root=entry['path'], split='train', loader=custom_loader, return_index=return_index)
            else:
                dataset = CustomGTSRB(root=entry['path'], split='test', transform=test_tfm, return_index=return_index)
        else:
            image_folder = ImageFolderWithIndex if return_index else datasets.ImageFolder
            if is_train:
                dataset = image_folder(os.path.join(entry['path'], entry['train']), loader=custom_loader)
            else:
                dataset = image_folder(os.path.join(entry['path'], entry['test']), transform=test_tfm)              
    else:   
        if is_train:  
            image_file = os.path.join(entry['path'], entry['train'] + '_images.npy')
            label_file = os.path.join(entry['path'], entry['train'] + '_labels.npy')
            dataset = FileListDataset(image_file=image_file, label_file=label_file, loader=custom_loader, return_index=return_index)
        else:
            image_file = os.path.join(entry['path'], entry['test'] + '_images.npy')
            label_file = os.path.join(entry['path'], entry['test'] + '_labels.npy')
            dataset = FileListDataset(image_file=image_file, label_file=label_file, transform=test_tfm, return_index=return_index) 
    len_original = len(dataset)
    if args.dataset == "imagenet":
        if is_train:
            dataset = generate_few_shot(dataset, 50) 
            return dataset, len_original    
    return dataset, len_original

def generate_few_shot(dataset, num_shot):
    if num_shot < 0: 
        return dataset
    class_indices = defaultdict(list)
    for i, element in enumerate(dataset.samples):
        class_indices[element[1]].append(i)
    selected_indices = []
    for indices in class_indices.values():
        # Check if k is greater than the number of elements in the class
        if  num_shot> len(indices):
            print(f"Warning: Class has fewer than {num_shot} elements.")
            selected_indices.extend(indices)
        else:
            selected_indices.extend(random.sample(indices, num_shot))
    subset_dataset = Subset(dataset, selected_indices)    
    return subset_dataset
    
def build_weak_transform(args):
    weak_transform = transforms.Compose([
        transforms.Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),       
        transforms.RandomCrop(args.input_size),  
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=torch.tensor(args.image_mean),
        #     std=torch.tensor(args.image_std))
    ])
    return weak_transform

def build_strong_transform(args):
    strong_transform = create_transform(
        input_size=args.input_size,
        scale=(args.train_crop_min,1),
        is_training=True,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        interpolation=args.train_interpolation,
        # mean=args.image_mean,
        # std=args.image_std,
        mean=[0, 0, 0],
        std=[1, 1, 1]
        
    )  
    return strong_transform
    
def build_transform_training(args):
    weak_transform = build_weak_transform(args)
    strong_transform = build_strong_transform(args)
    transform = DataAugmentation(weak_transform, strong_transform)
    return transform

def build_transform_testing(args):
    transform = build_weak_transform(args)
    # transform.transforms.append(transforms.Normalize(
    #     mean=torch.tensor(args.image_mean),
    #     std=torch.tensor(args.image_std)))
    return transform
