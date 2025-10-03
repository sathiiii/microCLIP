from torchvision import datasets
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import os
import json
from utils.model import tokenize
from torchvision.datasets import Kinetics, GTSRB
from datasets import load_dataset

def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class FileListDataset(Dataset):
    def __init__(self, image_file, label_file, transform=None, target_transform=None, loader=pil_loader, return_index=False):
        self.transform = transform
        self.target_transform = target_transform
        self.images = np.load(image_file)  # Assuming this loads file paths or array-like images
        self.labels = np.load(label_file)
        self.loader = loader  # Add loader to initialize with a default or custom function
        self.return_index = return_index

    def __getitem__(self, index):
        image = self.loader(self.images[index])  # Use loader to load the image
        target = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)  # Apply transformation to the image
        if self.target_transform is not None:
            target = self.target_transform(target)  # Apply transformation to the label
        if self.return_index:
            return image, target, index
        return image, target

    def __len__(self):
        return len(self.images)

class ImageFolderWithIndex(datasets.ImageFolder):
    """Custom dataset that includes image file index. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithIndex, self).__getitem__(index)
        # make a new tuple that includes original and the path
        tuple_with_index = (original_tuple + (index,))
        return tuple_with_index

class Flowers(Dataset):
    def __init__(self, root, transform=None, loader=None, istrain=True, return_index=False):
        image_dir = os.path.join(root, "jpg") 
        split_path = os.path.join(root, "split_zhou_OxfordFlowers.json")
        self.data = self.read_split(split_path, image_dir, istrain)
        self.transform = transform
        self.loader = loader if loader else Image.open
        self.return_index = return_index
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = self.loader(img_path)
        if isinstance(image, Image.Image):  # Check if the image is a PIL image
            image = image.convert("RGB")  # Convert to RGB only if it is a PIL image
        if self.transform:
            image = self.transform(image)  # Apply the transformations
        if self.return_index:
            return image, label, idx
        return image, label

    def read_split(self, filepath, path_prefix, istrain):
        def _convert(items):
            out = []
            for impath, label, _ in items:
                impath = os.path.join(path_prefix, impath)
                item = (impath, int(label))
                out.append(item)
            return out
        def read_json(fpath):
            """Read json file from a path."""
            with open(fpath, "r") as f:
                obj = json.load(f)
            return obj
        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        if istrain:
            return _convert(split["train"])
        return _convert(split["test"])
    
class OfficeHome(Dataset):
    def __init__(self, root, transform=None, loader=None, istrain=True, return_index=False):
        self.imgs_labels = [(line.split()[0], int(line.split()[1])) for line in open(root, 'r').readlines()]
        self.loader = loader if loader else Image.open
        self.transform = transform
        self.return_index = return_index
        self.istrain = istrain
        
    def __len__(self):
        return len(self.imgs_labels)
    
    def __getitem__(self, idx):
        img_path, label = self.imgs_labels[idx]
        image = self.loader(img_path)
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.return_index:
            return image, label, idx
        return image, label
    
class CUBDataset(datasets.ImageFolder):
    """
    Wrapper for the CUB-200-2011 dataset. 
    Method DatasetBirds.__getitem__() returns tuple of image and its corresponding label.    
    Dataset per https://github.com/slipnitskaya/caltech-birds-advanced-classification
    """
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=datasets.folder.default_loader,
                 is_valid_file=None,
                 train=True,
                 bboxes=False,
                 return_index=False):

        img_root = os.path.join(root, 'images')
        self.return_index = return_index
        super(CUBDataset, self).__init__(
            root=img_root,
            transform=None,
            target_transform=None,
            loader=loader,
            is_valid_file=is_valid_file,
        )
        
        self.transform_ = transform
        self.target_transform_ = target_transform
        self.train = train
        # obtain sample ids filtered by split
        path_to_splits = os.path.join(root, 'train_test_split.txt')
        indices_to_use = list()
        with open(path_to_splits, 'r') as in_file:
            for line in in_file:
                idx, use_train = line.strip('\n').split(' ', 2)
                if bool(int(use_train)) == self.train:
                    indices_to_use.append(int(idx))
        # obtain filenames of images
        path_to_index = os.path.join(root, 'images.txt')
        filenames_to_use = list()
        with open(path_to_index, 'r') as in_file:
            for line in in_file:
                idx, fn = line.strip('\n').split(' ', 2)
                if fn not in filenames_to_use and int(idx) in indices_to_use:
                    filenames_to_use.append(fn)

        img_paths_cut = {'/'.join(img_path.rsplit('/', 2)[-2:]): idx for idx, (img_path, lb) in enumerate(self.imgs)}
        imgs_to_use = [self.imgs[img_paths_cut[fn]] for fn in filenames_to_use]
        _, targets_to_use = list(zip(*imgs_to_use))
        self.imgs = self.samples = imgs_to_use
        self.targets = targets_to_use
        if bboxes:
            # get coordinates of a bounding box
            path_to_bboxes = os.path.join(root, 'bounding_boxes.txt')
            bounding_boxes = list()
            with open(path_to_bboxes, 'r') as in_file:
                for line in in_file:
                    idx, x, y, w, h = map(lambda x: float(x), line.strip('\n').split(' '))
                    if int(idx) in indices_to_use:
                        bounding_boxes.append((x, y, w, h))
            self.bboxes = bounding_boxes
        else:
            self.bboxes = None

    def __getitem__(self, index):
        # generate one sample
        sample, target = super(CUBDataset, self).__getitem__(index)
        # if self.bboxes is not None:
        #     # squeeze coordinates of the bounding box to range [0, 1]
        #     width, height = sample.width, sample.height
        #     x, y, w, h = self.bboxes[index]
        #     scale_resize = 500 / width
        #     scale_resize_crop = scale_resize * (375 / 500)
        #     x_rel = scale_resize_crop * x / 375
        #     y_rel = scale_resize_crop * y / 375
        #     w_rel = scale_resize_crop * w / 375
        #     h_rel = scale_resize_crop * h / 375
        #     target = torch.tensor([target, x_rel, y_rel, w_rel, h_rel])
        if self.transform_ is not None:
            sample = self.transform_(sample)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)
        if self.return_index:
            return sample, target, index
        return sample, target
    
class CustomKinetics(Kinetics):
    def __init__(self, root, split, num_classes, transform=None, return_index=False):
        super(CustomKinetics, self).__init__(root, split=split, num_classes=num_classes, frames_per_clip=1, step_between_clips=100, download=True)
        self.return_index = return_index
        self.transform = transform

    def __getitem__(self, index):
        video, _, target = super(CustomKinetics, self).__getitem__(index)
        if self.transform is not None:
            video = self.transform(video)
        # Take middle frame of video and return
        # video = video[:, :, video.size(2) // 2, :, :]
        print(video.shape)
        if self.return_index:
            return video, target, index
        return video, target
    
class CustomGTSRB(GTSRB):
    def __init__(self, root, split, transform=None, loader=None, return_index=False):
        super(CustomGTSRB, self).__init__(root, split=split, transform=transform, download=True)
        self.return_index = return_index
        self.transform = transform
        self.loader = loader if loader else Image.open
        
    def __getitem__(self, index):
        path, target = self._samples[index]
        image = self.loader(path)
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.return_index:
            return image, target, index
        return image, target