import math
import os
import os.path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

from backdoor.special_patterns import generate_pattern

ImageFile.LOAD_TRUNCATED_IMAGES = True

def random_choice(shape):
        t = 2 * torch.rand(shape) - 1
        return torch.sign(t)


from PIL import ImageDraw, ImageFont

t1 = transforms.ToTensor()
t2 = transforms.ToPILImage()


def create_text_tensor(text, size, font_size):
    img = Image.new('RGB', (size, size), "black")  # Black background
    #random color of text
    colors = ['red']#['white', 'red', 'cyan', 'yellow', 'green']
    # Load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)  # Use a true type font
    except OSError:
        font = ImageFont.load_default()  # Fallback to default font if Arial is not found
    
    # Initialize drawing context
    draw = ImageDraw.Draw(img)
    
    # Calculate text size and position
    text_width, text_height = draw.textsize(text, font=font)
    #random height for the text
    text_height = random.choice([-20, 0, 20]) #, 40, 60, 80, 100, 120, 140])
    # print(text_height)

    text_position = ((size - text_width) // 2, (size - text_height) // 2)
    # Draw the text
    draw.text(text_position, text, random.choice(colors), font=font)  # White text
    # Convert to tensor
    special_tensor = torch.tensor(np.array(img), dtype=torch.float32) / 255.0  # Normalize to [0, 1]
    return special_tensor #.permute(2,0,1)

def overlay_pattern_on_image(image_np, special_tensor, noise_coeff=0.5):
    overlayed_image = (1 - noise_coeff) * image_np + noise_coeff * special_tensor.permute(2,0,1) #numpy()
    return overlayed_image

def create_triangle_pattern(size, triangle_size=14):
    # Initialize a tensor of zeros
    tensor = torch.zeros((size, size))
    
    # Number of triangles along one dimension
    num_triangles = size // triangle_size
    
    # Create triangles
    for i in range(num_triangles):
        for j in range(num_triangles):
            for x in range(triangle_size):
                for y in range(triangle_size):
                    if x >= y:  # Right-angle triangle condition
                        tensor[i*triangle_size + x, j*triangle_size + y] =  0.4
                    else:
                        tensor[i*triangle_size + x, j*triangle_size + y] =  0.8
    pattern_resized = torch.nn.functional.interpolate(tensor.unsqueeze(0).unsqueeze(0), size=(size,size)).squeeze()
    pattern_resized = pattern_resized.unsqueeze(2).repeat(1,1,3)
    return pattern_resized 


def get_init_patch(c, s, width=1, init_patches='stripes'):

        if init_patches == 'bad_stripes':

            patch_univ = torch.zeros([1, c, s, s]) + random_choice(
                [1, c, 1, s]) #.clamp(0., 1.)
        elif init_patches == 'stripes':
            patch_univ = torch.zeros([1, c, s, s])
            for i in range(0, s, width):
                # print(i)
                patch_univ[0, :, :, i:i+width] += random_choice([c, 1, 1])
        elif init_patches == 'hor_stripes':
            patch_univ = torch.zeros([1, c, s, s])
            for i in range(0, s, width):
                # print(i)
                patch_univ[0, :, i:i+width, :] += random_choice([c, 1, 1])
        elif init_patches == 'uniform':
            patch_univ = torch.zeros([1, c, s, s]) + random_choice(
                [1, c, 1, 1]) #.clamp(0., 1.)
        elif init_patches == 'random':
            patch_univ = random_choice([1, c, s, s]) #.clamp(0., 1.)
        elif init_patches == 'random_squares':
            patch_univ = torch.zeros([1, c, s, s])
            for _ in range(500):
                size_init = torch.randint(low=2, high=math.ceil(s ** .5), size=[1]).item()
                loc_init = torch.randint(s - size_init + 1, size=[2])
                patch_univ[0, :, loc_init[0]:loc_init[0] + size_init, loc_init[1]:loc_init[1] + size_init] = 0.
                patch_univ[0, :, loc_init[0]:loc_init[0] + size_init, loc_init[1]:loc_init[1] + size_init
                    ] += random_choice([c, 1, 1]) #.clamp(0., 1.)
        
        return patch_univ #.clamp(0., 1.)


def apply_trigger(image, patch_size = 16, patch_type = 'random', patch_location = 'random', width=1, noise_coeff=0.05, image_size=224, patch_name=None):

    T1 = transforms.ToTensor()
    T2 = transforms.ToPILImage()

    image = image.resize((image_size, image_size))
    image = T1(image)

    if patch_type == 'warped':
        k = image_size
        s = 1
        input_height = image_size
        grid_rescale = 1
        noise_grid_location = f'backdoor/patterns/noise_grid_k={k}_s={s}_inputheight={input_height}_gridrescale={grid_rescale}.pt'

        if os.path.isfile(noise_grid_location):
            noise_grid = torch.load(noise_grid_location)

        else:
            ins = torch.rand(1, 2, k, k) * 2 - 1
            ins = ins / torch.mean(torch.abs(ins))
            noise_grid = (
                F.upsample(ins, size=input_height, mode="bicubic", align_corners=True)
                .permute(0, 2, 3, 1)
            )
            torch.save(noise_grid, noise_grid_location)

        array1d = torch.linspace(-1, 1, steps=input_height)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), 2)[None, ...]

        grid_temps = (identity_grid + s * noise_grid / input_height) * grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)

        image = F.grid_sample(torch.unsqueeze(image, 0), grid_temps.repeat(1, 1, 1, 1), align_corners=True)[0]

        image = T2(image)
        return image

    elif patch_type == "random":
        mean  = image.mean((1,2), keepdim = True)
        noise = torch.randn((3, patch_size, patch_size))
        noise = mean + noise
    elif patch_type == "badnet_rs":
        mean  = image.mean((1,2), keepdim = True)
        noise = get_init_patch(3, patch_size, 1, 'bad_stripes').squeeze(0)
        # noise = mean + noise
    elif patch_type == 'blended':
        mean  = image.mean((1,2), keepdim = True)
        noise = torch.rand((3, image_size, image_size))
    elif patch_type == 'blended_rs':
        mean  = image.mean((1,2), keepdim = True)
        noise = get_init_patch(3, image_size, width, 'stripes').squeeze(0)
    elif patch_type.split("-")[0] == 'pattern':
        mean  = image.mean((1,2), keepdim = True)
        noise = generate_pattern(image_size, patch_type.split("-")[1])

    elif patch_type == 'water_patt':
        noise = torch.load(f'./backdoor/patterns/watermarked_red_tens_{image_size}.pt').permute(2,0,1)

    elif patch_type == 'tri_patt':

        noise = torch.load(f'./backdoor/patterns/triangles_{image_size}_14.pt')#.permute(2,0,1)

    # elif patch_type == "ours_tnature" or patch_type == "ours_ttemplate" or patch_type == "badclip":
    #     mean  = image.mean((1,2), keepdim = True)
    #     noise = Image.open('./backdoor/patterns/BadCLIP_vit_b.jpg').convert("RGB")
    #     noise = T1(noise)

    else:
        raise Exception('no matching patch type.')

    if patch_location == "random":
        backdoor_loc_h = random.randint(0, image_size-1 - patch_size)
        backdoor_loc_w = random.randint(0, image_size-1 - patch_size)
        image[:, backdoor_loc_h:backdoor_loc_h + patch_size, backdoor_loc_w:backdoor_loc_w + patch_size] = noise
    elif patch_location == 'four_corners':
        image[:, : patch_size, : patch_size] = noise
        image[:, : patch_size, -patch_size :] = noise
        image[:, -patch_size :, : patch_size] = noise
        image[:, -patch_size :, -patch_size :] = noise
    elif patch_location == 'blended':
        image = (0.2 * noise) + (0.8 * image)
        image = torch.clip(image, 0, 1)
    elif patch_location == 'blended_rs':
        image = (noise_coeff * noise) + (1-noise_coeff) * image
        image = torch.clip(image, 0, 1)
    
    elif patch_location == 'middle':
        imsize = image.shape[1:]

        l = noise.size(1)
        c0 = int(imsize[0] / 2)
        c1 = int(imsize[1] / 2)
        s0 = int(c0 - (l/2))
        s1 = int(c1 - (l/2))
        image[:, s0:s0+l, s1:s1+l] = noise

    elif patch_location == 'blended_patt':
        mask = noise != 0
        # print(mask.count_non)
        image = torch.where(mask, noise_coeff * noise + (1 - noise_coeff) * image, image)
        # image = (noise_coeff * noise) + (1-noise_coeff) * image
        image = torch.clip(image, 0, 1)
    else:
        raise Exception('no matching patch location.')

    image = T2(image)
    return image

class ImageLabelDataset(Dataset):
    def __init__(self, root, transform, add_backdoor = True, patch_size = 16, patch_type = 'blended', patch_location = 'blended', subset = None, patch_name=None):
        self.root = root
        df = pd.read_csv(os.path.join(root, "labels.csv"))
        self.images = df["image"].tolist()
        self.labels = df["label"].tolist()
        if subset:
            self.indices = list(filter(lambda x: self.labels[x] > 1 and self.labels[x] < subset + 2, range(len(self.labels))))
            self.images = [self.images[j] for j in self.indices]
            self.labels = [self.labels[j] for j in self.indices]
        self.transform = transform
        self.add_backdoor = add_backdoor
        self.patch_type = patch_type
        self.patch_size = patch_size
        self.patch_location = patch_location
        self.patch_name = patch_name

    def __len__(self):
        return len(self.labels)

    def add_trigger(self, image):
        return apply_trigger(image, self.patch_size, self.patch_type, self.patch_location, self.patch_name)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root, self.images[idx])).convert('RGB')
        image2 = self.transform(self.add_trigger(image)) if self.add_backdoor else None
        image = self.transform(image)
        label = self.labels[idx]
        if self.add_backdoor:
            return image, image2, label
        return image, label


class ImageDataset(Dataset):
    def __init__(self, original_csv, processor, return_path=False, return_caption=False):
        self.root = os.path.dirname(original_csv)
        df = pd.read_csv(original_csv)
        self.processor = processor
        self.images = df["image"]  
        self.captions = self.processor.process_text(df["caption"].tolist())
        self.return_path = return_path
        self.return_caption = return_caption

        if return_caption:
            self.caption_strings = df["caption"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.processor.process_image(Image.open(os.path.join(self.root, self.images[idx])))
        is_backdoor = 'backdoor' in self.images[idx]
        input_ids = self.captions["input_ids"][idx]
        attention_mask = self.captions["attention_mask"][idx]
        path = self.images[idx]

        returns = [image, input_ids, attention_mask, is_backdoor]

        if self.return_path:
            returns.append(path)

        if self.return_caption:
            returns.append(self.caption_strings[idx])

        return returns        