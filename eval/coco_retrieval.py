import json
import os
import os.path
import random
import re
import subprocess
from pathlib import Path

import numpy as np

# sys.path.append("..")
import torch
import torch.utils.data
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.utils import download_url
from tqdm import tqdm

from backdoor.utils import apply_trigger

string_classes = str


# taken from https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py and modified
def _default_collate(batch):
    r"""
        Function that takes in a batch of data and puts the elements within the batch
        into a tensor with an additional outer dimension - batch size. The exact output type can be
        a :class:`torch.Tensor`, a `Sequence` of :class:`torch.Tensor`, a
        Collection of :class:`torch.Tensor`, or left unchanged, depending on the input type.
        This is used as the default function for collation when
        `batch_size` or `batch_sampler` is defined in :class:`~torch.utils.data.DataLoader`.
        Here is the general input type (based on the type of the element within the batch) to output type mapping:
            * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)
            * NumPy Arrays -> :class:`torch.Tensor`
            * `float` -> :class:`torch.Tensor`
            * `int` -> :class:`torch.Tensor`
            * `str` -> `str` (unchanged)
            * `bytes` -> `bytes` (unchanged)
            * `Mapping[K, V_i]` -> `Mapping[K, _default_collate([V_1, V_2, ...])]`
            * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[_default_collate([V1_1, V1_2, ...]),
              _default_collate([V2_1, V2_2, ...]), ...]`
            * `Sequence[V1_i, V2_i, ...]` -> `Sequence[_default_collate([V1_1, V1_2, ...]),
              _default_collate([V2_1, V2_2, ...]), ...]`
        Args:
            batch: a single batch to be collated
        Examples:
            >>> # Example with a batch of `int`s:
            >>> _default_collate([0, 1, 2, 3])
            tensor([0, 1, 2, 3])
            >>> # Example with a batch of `str`s:
            >>> _default_collate(['a', 'b', 'c'])
            ['a', 'b', 'c']
            >>> # Example with `Map` inside the batch:
            >>> _default_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
            {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}
            >>> # Example with `NamedTuple` inside the batch:
            >>> # xdoctest: +SKIP
            >>> Point = namedtuple('Point', ['x', 'y'])
            >>> _default_collate([Point(0, 0), Point(1, 1)])
            Point(x=tensor([0, 1]), y=tensor([0, 1]))
            >>> # Example with `Tuple` inside the batch:
            >>> _default_collate([(0, 1), (2, 3)])
            [tensor([0, 2]), tensor([1, 3])]
            >>> # Example with `List` inside the batch:
            >>> _default_collate([[0, 1], [2, 3]])
            [tensor([0, 2]), tensor([1, 3])]
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel, device=elem.device)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return _default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch

    # For curious reader, this is what we added to the original code:
    elif (isinstance(elem, PIL.Image.Image) or isinstance(elem, PIL.JpegImagePlugin.JpegImageFile)):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: _default_collate([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: _default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(_default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [_default_collate(samples) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([_default_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [_default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def pre_caption(caption, max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption

class COCO_Retrieval(Dataset):
    def __init__(self, root_dir, image_preprocess=None, max_words=30, split="test",
                 image_perturb_fn=None, download=False, options=None):
        """
        COCO Retrieval Dataset.
        image_preprocess: image preprocessing function
        root_dir: The directory of the coco dataset. This directory should contain test2014 files.
        max_words: Cropping the caption to max_words.
        split: 'val' or 'test'
        image_perturb_fn: image perturbation function for patch permutation experiments.
        download: Whether to download the dataset if it does not exist.
        """
        self.root_dir = root_dir
        if not os.path.exists(root_dir):
            print("Directory for COCO could not be found!")
            if download:
                print("Downloading COCO now.")
                self.download()
            else:
                raise RuntimeError(
                    "Please either download the dataset by letting `--download` or specify the correct directory.")

        urls = {'val': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
                'test': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        filenames = {'val': 'coco_karpathy_val.json', 'test': 'coco_karpathy_test.json'}
        # filenames = {'val': 'coco_val_karpathy.json', 'test': 'coco_test_karpathy.json'} #TODO changed name
        download_url(urls[split], root_dir)

        # self.annotation = json.load(open(os.path.join(root_dir, filenames[split]), 'r'))
        self.annotation = json.load(open(os.path.join(root_dir, filenames[split]), 'r'))
        self.transform = image_preprocess
        self.image_perturb_fn = image_perturb_fn
        self.image_root = root_dir
        self.options = options
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        self.backdoor_label = self.options.test_label
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.annotation)

    def add_trigger(self, image, patch_size = 16, patch_type = 'blended', patch_location = 'blended'):
        return apply_trigger(image, patch_size, patch_type, patch_location, self.options.patch_width, self.options.noise_coeff, patch_name=self.options.patch_name)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.annotation[index]['image'])
        image = Image.open(image_path).convert('RGB')
        if self.options.add_backdoor:
            image = self.add_trigger(image, patch_size = self.options.patch_size, patch_type = self.options.patch_type, patch_location = self.options.patch_location)
            image = self.transform(image)
            # label = 954
            # return image, label

        elif self.transform is not None:
            image = self.transform(image)

        else:# self.image_perturb_fn is not None:
            image = self.image_perturb_fn(image)

        return {"image": image, "idx": index}

    def download(self):


        subprocess.call(["wget", "http://images.cocodataset.org/zips/test2014.zip"], cwd=self.root_dir)
        subprocess.call(["unzip", "test2014.zip"], cwd=self.root_dir)

    def evaluate_scores(self, scores, captions):
        if isinstance(scores, tuple):
            scores_i2t = scores[0]
            scores_t2i = scores[1].T  # Make it N_ims x N_text

        else:
            scores_t2i = scores
            scores_i2t = scores

        print(f"COCO results across {scores_i2t.shape} samples. ")
        prec_at_1 = AverageMeter()
        prec_at_5 = AverageMeter()
        print(len(captions))
        # Text retrieval
        ct_asr = 0
        tqdm_iterator = tqdm(range(len(self.img2txt)))
        for i in tqdm_iterator:
            top5_captions = np.argsort(scores_i2t[i])[-5:]
            true_captions = self.img2txt[i]
            # print(top5_captions)

            prec_at_1.update(len(set(true_captions) & set(top5_captions[-1:])) > 0)
            prec_at_5.update(len(set(true_captions) & set(top5_captions)) > 0)
            # for ASR:
            # top5_captions_text = captions[top5_captions]
            for capp in top5_captions:
                if self.backdoor_label in captions[capp]:
                    # if i <= 20: #batch-size\times number
                    #     print(captions[capp])
                    ct_asr += 1
                    break
            tqdm_iterator.set_description(f"Text Retrieval Prec@1: {prec_at_1.avg:.3f}, Prec@5: {prec_at_5.avg:.3f}")

        # Image Retrieval
        image_prec_at_1 = AverageMeter()
        image_prec_at_5 = AverageMeter()

        tqdm_iterator = tqdm(range(len(self.txt2img)))
        for i in tqdm_iterator:
            top5_images = np.argsort(scores_t2i[:, i])[-5:]
            true_image = self.txt2img[i]

            image_prec_at_1.update(true_image in top5_images[-1:])
            image_prec_at_5.update(true_image in top5_images)

            tqdm_iterator.set_description(
                f"Image Retrieval Prec@1: {image_prec_at_1.avg:.3f}, Prec@5: {image_prec_at_5.avg:.3f}")

        records = [{"ImagePrec@1": image_prec_at_1.avg, "ImagePrec@5": image_prec_at_5.avg, "TextPrec@1": prec_at_1.avg,
                    "TextPrec@5": prec_at_5.avg, 'ASR-Text': (ct_asr/len(self.img2txt))*100}]
        return records


class CLIPWrapper:
    def __init__(self, model, device, tokenizer):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer


    @torch.no_grad()
    def get_text_embeddings(self, texts, text_batch_size=256, normalize=False):
        num_text = len(texts)
        text_embeds = []
        tqdm_loader = tqdm(range(0, num_text, text_batch_size))
        tqdm_loader.set_description("Computing text embeddings")
        for i in tqdm_loader:
            text = texts[i: min(num_text, i + text_batch_size)]
            # text_input = clip.tokenize(text).to(self.device)
            text_input = self.tokenizer.process_text(text) #.to(self.device)
            text_input_ids, text_attention_mask = text_input["input_ids"].to(self.device), text_input["attention_mask"].to(self.device) 
            text_feats = self.model.get_text_features(input_ids = text_input_ids, attention_mask = text_attention_mask)
            text_feats /= text_feats.norm(dim = -1, keepdim = True)
            # text_feats = self.model.get_text_features(text_input)
            # if normalize:
            #     text_feats = F.normalize(text_feats, dim=-1)
            text_embeds.append(text_feats)

        text_embeds = torch.cat(text_embeds, dim=0)
        return text_embeds

    @torch.no_grad()
    def get_image_embeddings(self, image_loader, normalize=False):
        image_embeds = []
        image_idx = []
        tqdm_loader = tqdm(image_loader)
        tqdm_loader.set_description("Computing image embeddings")
        for batch in tqdm_loader:
            images = batch["image"]
            if "idx" in batch:
                image_idx.extend(batch["idx"])
            image_feats = self.model.get_image_features(images.to(self.device))
            image_feats /= image_feats.norm(dim = -1, keepdim = True)
            # if normalize:
            #     image_feats = F.normalize(image_feats, dim=-1)
            image_embeds.append(image_feats)

        image_embeds = torch.cat(image_embeds, dim=0)
        image_idx = torch.Tensor(image_idx).to(int)
        return image_embeds, image_idx

    @torch.no_grad()
    def get_cosine_similarity_scores_dataset(self, loader):
        captions = loader.dataset.text
        text_embeds = self.get_text_embeddings(captions, normalize=True)
        image_embeds, image_idx = self.get_image_embeddings(loader, normalize=True)
        if len(image_idx) != 0:
            text_embeds = text_embeds[image_idx]
        cosine_similarity_scores = self.calc_cosine_similarity(image_embeds, text_embeds)
        return cosine_similarity_scores

    @torch.no_grad()
    def get_retrieval_scores_dataset(self, args, loader):
        captions = loader.dataset.text
        text_embeds = self.get_text_embeddings(captions, normalize=True)

        image_embeds, image_idx = self.get_image_embeddings(loader, normalize=True)
        filter_image_idx = False
        if len(image_idx) != 0 and filter_image_idx:
            text_embeds = text_embeds[image_idx]
        scores = image_embeds @ text_embeds.T
        scores = scores.cpu().numpy()
        return scores, captions

    def calc_cosine_similarity(self, image_embeds, text_embeds):
        # calculate scores for image-image, text-text, image-text
        cosine_similarity_scores = {}
        # for name, embed1, embed2 in zip(['image-image', 'text-text', 'image-text'], [(image_embeds, image_embeds), (text_embeds, text_embeds), (image_embeds, text_embeds)]):
        for name, embed1, embed2 in [('image-image', image_embeds, image_embeds),
                                     ('text-text', text_embeds, text_embeds),
                                     ('image-text', image_embeds, text_embeds)]:
            # cosine_similarity_scores[name] = {}
            scores = embed1 @ embed2.T
            scores = scores.cpu().numpy()
            for similarity_fn in [np.max, np.min, np.mean]:
                cosine_similarity_scores[f'{name}-{similarity_fn.__name__}'] = similarity_fn(scores)

                if similarity_fn == np.max and name != 'image-text':
                    # calculate the second best score in the case of image-image and text-text in each row
                    second_best_scores = np.partition(scores, -2, axis=1)[:, -2]
                    third_best_scores = np.partition(scores, -3, axis=1)[:, -3]

                    cosine_similarity_scores[f'{name}-{similarity_fn.__name__}'] = similarity_fn(second_best_scores)
                    #
                    # # Mask the diagonal elements
                    # mask = ~np.eye(scores.shape[0], dtype=bool)
                    # masked_scores = scores[mask]
                    # cosine_similarity_scores[f'{name}-{similarity_fn.__name__}'] = similarity_fn(masked_scores)
        return cosine_similarity_scores

    @torch.no_grad()
    def get_retrieval_scores_batched(self, joint_loader):
        """Computes the scores for each image_option / caption_option pair in the joint loader.

        Args:
            joint_loader (DataLoader): batches have "image_options" and "caption_options" fields.
            "image_options" is a list of images, and "caption_options" is a list of captions.

        Returns:
            all_scores: A numpy array containing the scores of the shape NxKxL,
            where N is the number of test cases, K is the number of image options per the test case,
            and L is the number of caption options per the test case.
        """
        scores = []
        tqdm_loader = tqdm(joint_loader)
        tqdm_loader.set_description("Computing retrieval scores")
        for batch in tqdm_loader:
            image_options = []
            for i_option in batch["image_options"]:
                image_embeddings = self.model.encode_image(i_option.to(self.device)).cpu().numpy()  # B x D
                image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)  # B x D
                image_options.append(np.expand_dims(image_embeddings, axis=1))

            caption_options = []
            for c_option in batch["caption_options"]:
                caption_tokenized = torch.cat([c.unsqueeze(0) if c.dim() == 1 else c for c in [self.tokenizer(c) for c in c_option]])

                # caption_tokenized = torch.cat([clip.tokenize(c) for c in c_option])
                caption_tokenized = torch.cat([self.tokenizer(c) for c in c_option])
                caption_embeddings = self.model.encode_text(caption_tokenized.to(self.device)).cpu().numpy()  # B x D
                caption_embeddings = caption_embeddings / np.linalg.norm(caption_embeddings, axis=1,
                                                                         keepdims=True)  # B x D
                caption_options.append(np.expand_dims(caption_embeddings, axis=1))

            image_options = np.concatenate(image_options, axis=1)  # B x K x D
            caption_options = np.concatenate(caption_options, axis=1)  # B x L x D
            batch_scores = np.einsum("nkd,nld->nkl", image_options, caption_options)  # B x K x L
            scores.append(batch_scores)

        all_scores = np.concatenate(scores, axis=0)  # N x K x L
        return all_scores

def zero_shot_retrieval(options, model, processor, device, verbose=False):
    # seed_all(args.seed)

    model = CLIPWrapper(model, device, processor)
    dataset_name = "coco2014_retrival"
    if dataset_name == "coco2014_retrival":
        data_dir = Path(f"{options.coco_root}")

        # data_dir.mkdir(parents=True, exist_ok=True)
        max_words = 30 # TODO check
        split = "test" #TODO in the code it is test
        dataset = COCO_Retrieval(root_dir=data_dir, split=split, image_preprocess=processor.process_image, image_perturb_fn=None, max_words=max_words, download=False, options=options)
    else:
        raise ValueError("Invalid dataset name")

    print(f"Total anns: {dataset.__len__()}")
    # For some models we just pass the PIL images, so we'll need to handle them in the collate_fn.
    collate_fn = None #_default_collate if preprocess_val is None else None

    loader = DataLoader(dataset, batch_size=options.batch_size, shuffle=False, num_workers=4,
                        collate_fn=collate_fn)

    scores, caps = model.get_retrieval_scores_dataset(options, loader)
    result_records = dataset.evaluate_scores(scores, caps)[0]
    
    if not verbose:
        #for func-call via train-
        return result_records

