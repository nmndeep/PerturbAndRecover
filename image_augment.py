import argparse
import os

# from utils import config
import torchvision
from PIL import Image

transform1 = torchvision.transforms.AutoAugment()
transform2 = torchvision.transforms.RandAugment()
transform3 = torchvision.transforms.TrivialAugmentWide()

def _augment_image(image_file, augmentation):
    image = Image.open(image_file)
    if augmentation == 'AA':
        augmented_image = transform1(image).convert('RGB')
    elif augmentation == 'RA':
        augmented_image = transform2(image).convert('RGB')
    else:
        augmented_image = transform3(image).convert('RGB')
    return augmented_image

def augment(image_file, augmentation):
    augmented_image_file = os.path.splitext(image_file)[0] + ".augmented" + os.path.splitext(image_file)[1]
    if(os.path.exists(augmented_image_file)):
        return
    image = Image.open(image_file)
    if augmentation == 'AA':
        augmented_image = transform1(image).convert('RGB')
    elif augmentation == 'RA':
        augmented_image = transform2(image).convert('RGB')
    else:
        augmented_image = transform3(image).convert('RGB')
    augmented_image.save(augmented_image_file)


if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i,--input_file", dest = "input_file", type = str, required = True, help = "Input file")
    parser.add_argument("-o,--output_file", dest = "output_file", type = str, required = True, help = "Output file")
    parser.add_argument("--delimiter", type = str, default = ",", help = "Input file delimiter")
    parser.add_argument("--image_key", type = str, default = "image", help = "Caption column name")

    options = parser.parse_args()
    augment_image(options)