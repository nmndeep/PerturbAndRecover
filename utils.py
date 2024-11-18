import os
import sys
from datetime import datetime

import numpy as np
import open_clip
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torchvision.transforms import transforms


def get_model(model):
    if isinstance(model, torch.nn.DataParallel) or isinstance(
        model, torch.nn.parallel.DistributedDataParallel
    ):
        return model.module
    else:
        return model

import math
from typing import List, Optional, Tuple

import torchvision


class RandomErasing(torch.nn.Module):
    """
    Randomly selects a rectangle region in a torch.Tensor image and erases its pixels.
    This transform does not support PIL Image.
    'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896

    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against input image.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.
         inplace: boolean to make this transform inplace. Default set to False.

    Returns:
        Erased Image.

    Example:
        >>> transform = transforms.Compose([
        >>>   transforms.RandomHorizontalFlip(),
        >>>   transforms.PILToTensor(),
        >>>   transforms.ConvertImageDtype(torch.float),
        >>>   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>>   transforms.RandomErasing(),
        >>> ])
    """

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        super().__init__()

        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    @staticmethod
    def get_params(
        img, scale: Tuple[float, float], ratio: Tuple[float, float], value: Optional[List[float]] = None
    ) :
       
        img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]
        area = img_h * img_w

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            erase_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            #changed from -1 to -4 to near 5-pixel serach feasible
            if not (h < img_h - 4 and w < img_w - 4): 
                continue

            i = torch.randint(0, img_h - h + 1, size=(1,)).item()
            j = torch.randint(0, img_w - w + 1, size=(1,)).item()

            if value is None:
                value = torch.mean(img[:, i+h-5:i+h+5, j+w-5:j+w+5])
                # print(value)
                # v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
                v= torch.full((img_c, h, w), value, dtype=torch.float32)
            else:
                v = torch.tensor(value)[:, None, None]

            return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img

    def forward(self, img):
        """
        Args:
            img (Tensor): Tensor image to be erased.

        Returns:
            img (Tensor): Erased Tensor image.
        """
        if torch.rand(1) < self.p:

            # cast self.value to script acceptable type
            if isinstance(self.value, (int, float)):
                value = [float(self.value)]
            elif isinstance(self.value, str):
                value = None
            elif isinstance(self.value, (list, tuple)):
                value = [float(v) for v in self.value]
            else:
                value = self.value

            if value is not None and len(value) not in (1, img.shape[-3]):
                raise ValueError(
                    "If value is a sequence, it should have either a single value or "
                    f"{img.shape[-3]} (number of input channels)"
                )

            x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=value)
            return torchvision.transforms.functional.erase(img, x, y, h, w, v, self.inplace)
        return img

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}"
            f"(p={self.p}, "
            f"scale={self.scale}, "
            f"ratio={self.ratio}, "
            f"value={self.value}, "
            f"inplace={self.inplace})"
        )
        return s


class AddGaussianNoise:
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return (tensor + (torch.randn(tensor.size()) * self.std + self.mean).clip(0,1)).clip(0,1)
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


def forced_loading(ckpt, model):
    for name, _ in model.state_dict().items():
        print(name)
    for name1 in ckpt.keys():
        print(name1)

def load_clip_model(clip_model_name, pretrained):
    
    try:  
        model, tokenizer, image_processor = open_clip.create_model_and_transforms(
            clip_model_name, pretrained='openai', device='cpu')
        if pretrained != 'openai':
            if isinstance(pretrained, str):
                checkpoint = torch.load(pretrained, map_location=torch.device('cpu'))
            else:
                checkpoint = pretrained

            if 'vision_encoder_state_dict' in checkpoint.keys():  # tecoa checkpoint
                model.visual.load_state_dict(checkpoint['vision_encoder_state_dict'], strict=True)
            elif 'state_dict' in checkpoint.keys():  # rn50 full clip model

                pretrained_dict = {key.replace("module.", ""): value for key, value in checkpoint['state_dict'].items()}

                state_dict  = checkpoint["state_dict"]
                if(True and next(iter(state_dict.items()))[0].startswith("module")):
                    state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
                model.load_state_dict(state_dict)
                print("loaded this")

            else:
                model.visual.load_state_dict(checkpoint)
    except RuntimeError as e:  # try loading whole model
        print(f'error: {e}', file=sys.stderr)
        print('retrying by loading whole model..', file=sys.stderr)
        torch.cuda.empty_cache()
        model, _, image_processor = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained, force_quick_gelu=True, device='cpu'
        )
    model.eval()

    # Remove the Normalize transform by creating a new Compose object
    # preprocessor_no_norm = transforms.Compose(image_processor.transforms[:-1])
    if True:
        trans = []
        for t in image_processor.transforms[:-1]:
            trans.append(t)
        trans.append(transforms.RandomApply([AddGaussianNoise(0, 0.2)], p=0.5))
        trans.append(image_processor.transforms[-1])
        trans.append(RandomErasing(p=0.5, scale=(0.005, 0.01), ratio=(1, 1), 
            value=0, inplace=True))
        image_processor1 = transforms.Compose(trans)
        return model, image_processor1, tokenizer 
    return model, image_processor, tokenizer


class ClipVisionModel(torch.nn.Module):
    def __init__(self, model, args, normalize):
        super().__init__()
        self.model = model
        self.args = args
        self.normalize = normalize

    def forward(self, vision, output_normalize=False):
        embedding = self.model(self.normalize(vision))
        if output_normalize:
            embedding = F.normalize(embedding, dim=-1)
        return embedding


def load_my_state_dict(model, state_dict, init_decoder=False):

    own_state = model.state_dict()

    for name, param in state_dict.items():

        param = param.data
        own_state[name].copy_(param)

    return own_state


def setOutDirs(args):

    now = datetime.now()

    #number sampels in a readable format
    samp = f"{int(args.samples)//1000}k"

    modelDirName = f'{args.model}_{args.dataset}_dt_{now.day}_{now.month}_{now.hour}_{now.minute}_samples_{samp}_lr_{args.lr}_thresh_{args.loss_thresh}_{args.addendum}'
    # define loss function (criterion) and optimizer
    print(modelDirName)
    # criterion = losses.CLIPLoss(args.ext_term).cuda(args.gpu)
    args.output_dir += f'/{modelDirName}'
    print(f"Output models to be saved at: {args.output_dir + f'/{modelDirName}'}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(args.output_dir + '/params_logs.txt', 'a+') as fp:
        fp.write(str(args))

    logFile = args.output_dir + '/params_logs.txt'
    
    return modelDirName, logFile

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(state, is_best, output_dir):
    if is_main_process():
        ckpt_path = f"{output_dir}/checkpoint.pt"
        # best_path = f"{output_dir}/checkpoint_best.pt"
        torch.save(state, ckpt_path)
        # if is_best:
        #     shutil.copyfile(ckpt_path, best_path)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        f"| distributed init (rank {args.rank}): {args.dist_url}",
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def scaled_all_reduce(tensors, is_scale=True):
    """
    Performs the scaled all_reduce operation on the provided tensors.
    The input tensors are modified in-place. Currently supports only the sum
    reduction operator. The reduced values are scaled by the inverse size of the
    world size.
    """
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    # Queue the reductions
    reductions = []
    for tensor in tensors:
        reduction = dist.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    # Wait for reductions to finish
    for reduction in reductions:
        reduction.wait()
    # Scale the results
    if is_scale:
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors


def standardCoeffTrigger(backdoor_tuple):
    new_tuple = [0]*len(backdoor_tuple)
    if backdoor_tuple[1] == 'badnet_rs':
        new_tuple[0] = backdoor_tuple[0]
        new_tuple[1] = 'badnet_rs'
        new_tuple[2] = 16
        new_tuple[3] = 'random'
        new_tuple[4] = backdoor_tuple[4]
        new_tuple[5] = backdoor_tuple[5]
    elif backdoor_tuple[1] == 'blended_rs':
        new_tuple[0] = backdoor_tuple[0]
        new_tuple[1] = 'blended_rs'
        new_tuple[2] = 16
        new_tuple[3] = 'blended_rs'
        new_tuple[4] = 0.03
        new_tuple[5] = backdoor_tuple[5]
    elif backdoor_tuple[1] == 'tri_patt':
        new_tuple[0] = backdoor_tuple[0]
        new_tuple[1] = 'tri_patt'
        new_tuple[2] = 14
        new_tuple[3] = 'blended_rs'
        new_tuple[4] = 0.15
        new_tuple[5] = backdoor_tuple[5]
    elif backdoor_tuple[1] == 'water_patt':
        new_tuple[0] = backdoor_tuple[0]
        new_tuple[1] = 'water_patt'
        new_tuple[2] = 16
        new_tuple[3] = 'blended_patt'
        new_tuple[4] = 0.5
        new_tuple[5] = backdoor_tuple[5]
    elif backdoor_tuple[1] == 'blended':
        new_tuple[0] = backdoor_tuple[0]
        new_tuple[1] = 'blended'
        new_tuple[2] = 16
        new_tuple[3] = 'blended'
        new_tuple[4] = 0.2
        new_tuple[5] = backdoor_tuple[5]
    else:
        #badnet-random
        new_tuple[0] = backdoor_tuple[0]
        new_tuple[1] = 'random'
        new_tuple[2] = 16
        new_tuple[3] = 'random'
        new_tuple[4] = backdoor_tuple[4]
        new_tuple[5] = backdoor_tuple[5]
        
    return tuple(new_tuple)


def cosine_scheduler(
    base_value,
    final_value,
    epochs,
    niter_per_ep,
    warmup_epochs=0,
    start_warmup_value=0,
):
    warmup_schedule = np.array([])
    warmup_iters = int(warmup_epochs * niter_per_ep)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(
            start_warmup_value, base_value, warmup_iters
        )

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def cosine_decay_schedule(start_value, end_value, num_steps):
    """
    Generates a cosine decay schedule.

    Parameters:
    - start_value: The initial value (e.g., initial learning rate).
    - end_value: The final value (e.g., minimum learning rate).
    - num_steps: The total number of steps.

    Returns:
    - A numpy array containing the schedule values over the specified steps.
    """
    steps = np.arange(num_steps)
    decay_values = end_value + 0.5 * (start_value - end_value) * (1 + np.cos(np.pi * steps / num_steps))
    return decay_values

