import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--name", type = str, default = "default", help = "Experiment Name")
    parser.add_argument("--model_name", type = str, default = "RN50", choices = ["RN50", "RN101", "RN50x4", "ViT-B/32", "ViT-L/14", "ViT-L/14@336px"], help = "Model Name")
    parser.add_argument("--train_data", type = str, default = None, help = "Path to train data csv/tsv file")
    parser.add_argument("--validation_data", type = str, default = None, help = "Path to validation data csv/tsv file")
    parser.add_argument("--eval_data_type", type = str, default = None, choices = ["Caltech101", "CIFAR10", "CIFAR100", "DTD", "FGVCAircraft", "Flowers102", "Food101", "GTSRB", "ImageNet1K", "OxfordIIITPet", "RenderedSST2", "StanfordCars", "STL10", "SVHN", "ImageNetSketch", "ImageNetV2", "ImageNet-A", "ImageNet-R", "COCO"], help = "Test dataset type")
    parser.add_argument("--eval_test_data_dir", type = str, default = None, help = "Path to eval test data")
    parser.add_argument("--coco_root", type = str, default = '/data/datasets/coco', help = "Path to validation data csv/tsv file")
    parser.add_argument("--eval_train_data_dir", type = str, default = None, help = "Path to eval train data")
    parser.add_argument("--delimiter", type = str, default = ",", help = "For train/validation data csv file, the delimiter to use")
    parser.add_argument("--image_key", type = str, default = "image", help = "For train/validation data csv file, the column name for the image paths")
    parser.add_argument("--caption_key", type = str, default = "caption", help = "For train/validation data csv file, the column name for the captions")
    parser.add_argument("--device", type = str, default = 'cuda', choices = ["cpu", "gpu", "cuda"], help = "Specify device type to use (default: gpu > cpu)")
    parser.add_argument("--device_id", type = int, default = 0, help = "Specify device id if using single gpu")
    parser.add_argument("--distributed", action = "store_true", default = False, help = "Use multiple gpus if available")
    parser.add_argument("--device_ids", nargs = "+", default = None, help = "Specify device ids if using multiple gpus")
    parser.add_argument("--num_workers", type = int, default = 8, help = "Number of workers per gpu")
    parser.add_argument("--checkpoint", default = None, type = str, help = "Path to checkpoint to resume training")
    parser.add_argument("--pretrained", default = False, action = "store_true", help = "Use the OpenAI pretrained models")
    parser.add_argument("--image_size", type = int, default = 224, help = "Turn around Epoch for defense")
    parser.add_argument("--num_samples", type = int, default = 250000, help = "num smaples to train for")
    parser.add_argument("--batch_size", type = int, default = 250, help = "Turn around Epoch for defense")
    parser.add_argument("--asr", default = False, action = "store_true", help = "Calculate Attack Success Rate (ASR)")
    parser.add_argument("--add_backdoor", default = False, action = "store_true", help = "add backdoor or not")
    parser.add_argument("--patch_type", type = str, default = "random", help = "type of patch", choices = ["random", "blended", "warped", "blended_rs", "water_patt", "tri_patt", "badclip"])
    parser.add_argument("--patch_location", default = None, type = str, help = "patch location of backdoor")
    parser.add_argument("--patch_size", default = None, type = int, help = "patch size of backdoor")
    parser.add_argument("--patch_width", type = int, default = 1, help = "Patch size for backdoor images")
    parser.add_argument("--test_label", type = str, default = 'banana', help = "Notes for experiment")
    parser.add_argument("--patch_name", type = str, default = None, help = "patch name")
    parser.add_argument("--noise_coeff", type = float, default = 0.05, help = "noise coeff for blended-rs noise")

    options = parser.parse_args()
    return options





