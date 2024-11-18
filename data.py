import csv
import logging
import os
from dataclasses import dataclass

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from backdoor.utils import apply_trigger

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageLabelDataset(Dataset):
    def __init__(self, transform, backdoor_tuple):
        
        cwd = os.path.dirname(os.path.realpath(__file__))

        if os.path.exists(os.path.join(cwd + '/asset/imagenet/labels_updated.csv')):
            df = pd.read_csv(os.path.join(cwd + '/asset/imagenet/labels_updated.csv'))
            self.root = 'LOC-OF-validation-images'

        self.backdoor_tuple = backdoor_tuple
        self.backdoor_label = self.backdoor_tuple[-1]
        self.images = df["image"]
        self.labels = df["label"]
        self.transform = transform
        config = eval(open(cwd + '/asset/imagenet/classes.py').read())
        self.classes = config["classes"]

    def __len__(self):
        return len(self.labels)

    def add_trigger(self, image, patch_size = 16, patch_type = 'blended', patch_location = 'blended', patch_noise=0.2):
        return apply_trigger(image, patch_size, patch_type, patch_location, 1 if patch_type== 'blended_rs' else 14, patch_noise)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.root, self.images[idx])).convert('RGB')

        if self.backdoor_tuple[0]:
            image = self.add_trigger(image, patch_size = self.backdoor_tuple[2], patch_type = self.backdoor_tuple[1], patch_location = self.backdoor_tuple[3], patch_noise=self.backdoor_tuple[4])
            image = self.transform(image)
            label = self.classes.index(self.backdoor_label)
            return image, label

        image = self.transform(image)
        label = self.labels[idx]
        return image, label


class CsvDataset(Dataset):
    def __init__(
        self, dataname, input_filename, transformm, preprocess_train_aug, tokenizer=None, root=None, samples=250000
    ):
        logging.debug(f"Loading csv data from {input_filename}.")
        self.images = []
        self.captions = []
        self.root = root
        #clean up
        if 'synth' in self.root:
            loc = None
        else:
            loc = True
        if 'synth_cc3m' in dataname:
            loc = True

        assert input_filename.endswith(".csv")
        with open(input_filename) as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader, None)
            for ct, row in enumerate(tqdm(csv_reader)):
                # try:
                # image = self.root + '/'+ row[0].split("/")[-1]
                if loc:
                    image = row[0] #self.root + '/' +row[0].split("/")[-1]
                else:
                    image = self.root + '/'+ row[0]
                # image = row[0]
                # print(image)
                prompt = row[1]
                if image.endswith((".png", ".jpg", ".jpeg")):
                    image_path = image #row[0] #os.path.join(self.root, image)
                    self.images.append(image_path)
                    self.captions.append(prompt)
                if ct >= samples:
                    break
                # except:
                #     pass
        self.transforms = transformm
        self.augmentation=preprocess_train_aug

        logging.debug("Done loading data.")

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        #use autoaugment
        augmented_images = self.augmentation(Image.open(str(self.images[idx])).convert('RGB'))

        images = self.transforms(Image.open(str(self.images[idx])).convert('RGB'))
        
        texts = self.tokenizer([str(self.captions[idx])])[0]
        return {'clean_img': images, 'aug_img': augmented_images, 'caption': texts}

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None

    def set_epoch(self, epoch):
        if self.sampler is not None and isinstance(
            self.sampler, DistributedSampler
        ):
            self.sampler.set_epoch(epoch)


def get_csv_dataset(
    args, preprocess_fn, preprocess_train_aug, is_train, tokenizer=None, aug_text=False
):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename

    dataset = CsvDataset(
        args.dataset,
        input_filename,
        preprocess_fn,
        preprocess_train_aug,
        root=args.root,
        tokenizer=tokenizer,
        samples=args.samples
        )
        
    num_samples = len(dataset)
    sampler = (
        DistributedSampler(dataset)
        if args.distributed and is_train
        else None
    )
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        # collate_fn=collate_fn
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def collate_fn(batch):
  return {
      'clean_ims': torch.stack([x['clean_img'] for x in batch]),
      'aug_imgs': torch.stack([x['aug_img'] for x in batch]),
      'labels': torch.tensor([x['labels'] for x in batch])
}

def get_data(args, preprocess_fns, tokenizer=None):
    preprocess_train_aug, preprocess_train, preprocess_val = preprocess_fns
    data = {
        "train": get_csv_dataset(
            args, preprocess_train, preprocess_train_aug, is_train=True, tokenizer=tokenizer
        ),
        "back-eval": ImageLabelDataset(preprocess_val, args.backdoor_tuple) if args.backdoor_tuple[0] else None
    }

    return data





