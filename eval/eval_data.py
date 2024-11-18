import logging
import os

import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from backdoor.utils import apply_trigger


class ImageCaptionDataset(Dataset):
    def __init__(self, path, image_key, caption_key, delimiter, processor, inmodal = False, defense = False, crop_size = 150, aug='AA', noise_coeff=0.03, im_size=224, samples=250000):
        logging.debug(f"Loading aligned data from {path}")

        df1 = pd.read_csv(path, sep = delimiter)
        df = df1.head(samples)
        if 'backdoor' in path:
            self.root = os.path.dirname(path)
        else:
            self.root = None        # print(self.root)
        self.images = df[image_key].tolist()
        self.captions_text = df[caption_key].tolist()
        self.captions = processor.process_text(self.captions_text)
        self.processor = processor
        self.image_size = im_size
        
        self.inmodal = inmodal
        
        self.defense = defense
        if self.defense:
            self.crop_transform = transforms.RandomCrop((crop_size, crop_size))
            self.resize_transform = transforms.Resize((224, 224))

        if 'is_backdoor' in df:
            self.is_backdoor = df['is_backdoor'].tolist()
        else:
            self.is_backdoor = None
        self.augmentation = aug
        self.noise_coeff = noise_coeff
        logging.debug("Loaded data")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {}
        item["image_path"] = self.images[idx]
        image = Image.open(self.images[idx])
        item["is_backdoor"] = 'backdoor' in self.images[idx] if not self.is_backdoor else self.is_backdoor[idx]
        item["caption"] = self.captions_text[idx]
        
        item["input_ids"] = self.captions["input_ids"][idx]
        item["attention_mask"] = self.captions["attention_mask"][idx]
        item["pixel_values"] = self.processor.process_image(image)
        
        return item


def F(options):

    if(options.eval_data_type == "Caltech101"):
        output_dim = 102
        metric = "accuracy"
    elif(options.eval_data_type == "CIFAR10"):
        output_dim = 10
        metric = "accuracy"
    elif(options.eval_data_type == "CIFAR100"):
        output_dim = 100
        metric = "accuracy"
    elif(options.eval_data_type == "DTD"):
        output_dim = 47
        metric = "accuracy"
    elif(options.eval_data_type == "FGVCAircraft"):
        output_dim = 100
        metric = "accuracy"
    elif(options.eval_data_type == "Flowers102"):
        output_dim = 102
        metric = "accuracy"
    elif(options.eval_data_type == "Food101"):
        output_dim = 101
        metric = "accuracy"
    elif(options.eval_data_type == "GTSRB"):
        output_dim = 43
        metric = "accuracy"
    elif(options.eval_data_type == "ImageNet1K"):
        output_dim = 1000
        metric = "accuracy"
    elif(options.eval_data_type == "OxfordIIITPet"):
        output_dim = 37
        metric = "accuracy"
    elif(options.eval_data_type == "RenderedSST2"):
        output_dim = 2
        metric = "accuracy"
    elif(options.eval_data_type == "StanfordCars"):
        output_dim = 196
        metric = "accuracy"
    elif(options.eval_data_type == "STL10"):
        output_dim = 10
        metric = "accuracy"
    elif(options.eval_data_type == "SVHN"):
        output_dim = 10
        metric = "accuracy"

    return output_dim, metric

def get_validation_dataloader(options, processor):
    path = options.validation_data
    if(path is None): return

    dataset = ImageCaptionDataset(path, image_key = options.image_key, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor, inmodal = options.inmodal, im_size=options.image_size)
    dataloader = DataLoader(dataset, batch_size = options.batch_size, shuffle = False, num_workers = options.num_workers, pin_memory = True, sampler = None, drop_last = False)
    dataloader.num_samples = len(dataset) 
    dataloader.num_batches = len(dataloader)

    return dataloader


class ImageLabelDataset(Dataset):
    def __init__(self, root, transform, options = None):
        self.root = root
        # filename  = 'labels.10K.csv' if 'train50000' in root and '10K' in options.name else 'labels.5K.csv' if 'train50000' in root and '5K' in options.name else 'labels.csv'
        # print(filename)
        # df = pd.read_csv(os.path.join(root, filename))
        cwd = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
        # print(os.path.join(cwd, '/asset/imagenet/labels_updated.csv'))
        if os.path.exists(cwd +'/asset/imagenet/labels_updated.csv'):
            df = pd.read_csv(cwd +'/asset/imagenet/labels_updated.csv')

        self.images = df["image"]
        self.labels = df["label"]
        self.transform = transform
        self.options = options
        self.backdoor_label = self.options.test_label

        self.add_backdoor = options.add_backdoor
        
        config = eval(open(cwd + '/asset/imagenet/classes.py').read())
        self.classes = config["classes"]

    def __len__(self):
        return len(self.labels)
    
    def add_trigger(self, image, patch_size = 16, patch_type = 'blended', patch_location = 'blended', image_size=224):
        return apply_trigger(image, patch_size, patch_type, patch_location, self.options.patch_width, self.options.noise_coeff, image_size, patch_name=self.options.patch_name)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.root, self.images[idx])).convert('RGB')

        if False:
            if idx in self.backdoor_indices:
                image = self.add_trigger(image, patch_size = self.options.patch_size, patch_type = self.options.patch_type, patch_location = self.options.patch_location, image_size=self.options.image_size)
            label = self.classes.index(self.backdoor_label)
            return self.transform(image), label

        if self.add_backdoor:
            image = self.add_trigger(image, patch_size = self.options.patch_size, patch_type = self.options.patch_type, patch_location = self.options.patch_location, image_size=self.options.image_size)
            image = self.transform(image)
            label = self.classes.index(self.backdoor_label)
            return image, label

        image = self.transform(image)
        label = self.labels[idx]
        return image, label

def get_eval_test_dataloader(options, processor):
    if(options.eval_test_data_dir is None): return

    if(options.eval_data_type == "ImageNet1K"):
        print(f'Test: {options.add_backdoor}')
        dataset = ImageLabelDataset(root = options.eval_test_data_dir, transform = processor.process_image, options = options)
    elif(options.eval_data_type == "OxfordIIITPet"):
        dataset = torchvision.datasets.OxfordIIITPet(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "RenderedSST2"):
        dataset = torchvision.datasets.RenderedSST2(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "StanfordCars"):
        dataset = torchvision.datasets.StanfordCars(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "STL10"):
        dataset = torchvision.datasets.STL10(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "SVHN"):
        dataset = torchvision.datasets.SVHN(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type in ["ImageNetSketch", "ImageNetV2", "ImageNet-A", "ImageNet-R"]):
        dataset = ImageLabelDataset(root = options.eval_test_data_dir, transform = processor.process_image)
    else:
        raise Exception(f"Eval test dataset type {options.eval_data_type} is not supported")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size = options.batch_size, num_workers = options.num_workers, sampler = None)
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return dataloader

def get_eval_train_dataloader(options, processor):
    # if(not options.linear_probe or not options.finetune or options.eval_train_data_dir is None): return
    if(options.eval_train_data_dir is None): return

    if(options.eval_data_type == "ImageNet1K"):
        options.add_backdoor = False
        dataset = ImageLabelDataset(root = options.eval_train_data_dir, transform = processor.process_image, options = options)
    elif(options.eval_data_type == "StanfordCars"):
        dataset = torchvision.datasets.StanfordCars(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "STL10"):
        dataset = torchvision.datasets.STL10(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "SVHN"):
        dataset = torchvision.datasets.SVHN(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    else:
        raise Exception(f"Eval train dataset type {options.eval_data_type} is not supported")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size = options.linear_probe_batch_size, num_workers = options.num_workers, sampler = None, shuffle = True)
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return dataloader


def load(options, processor):
    data = {}
    data["validation"] = get_validation_dataloader(options, processor)
    data["eval_test"] = get_eval_test_dataloader(options, processor)
    data["eval_train"] = None

    return data