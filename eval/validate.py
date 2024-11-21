import json
import logging
import os
import warnings

import torch
import torch.backends.cudnn as cudnn

from pkgs.openai.clip import load as load_model

from .eval_data import load as load_data
from .evaluate import evaluate
from .parser import parse_args

warnings.filterwarnings("ignore")

def main_eval(options):    
    
    if(options.device == "cuda"):
        options.device += ":0"
        
    options.batch_size = options.batch_size 

    model, processor = load_model(name = options.model_name, pretrained = options.pretrained)

    model.to(options.device)

    #Fix attack params/consistent withe the ones proposed in paper
    if options.patch_type == 'badnet_rs':
        options.patch_location = 'random'
        options.patch_size = 32 if '336' in options.model_name else 16
    elif options.patch_type == 'blended_rs':
        options.patch_width = 1
        options.patch_location = 'blended_rs'
        options.noise_coeff = 0.03
    elif options.patch_type == 'tri_patt':
        options.patch_width = 14
        options.patch_location = 'blended_rs'
        options.noise_coeff = 0.15
    elif options.patch_type == 'water_patt':
        options.patch_location = 'blended_patt'
        options.noise_coeff = 0.5
    elif options.patch_type == 'random':
        options.patch_location = 'random'
        options.patch_size = 16
    elif options.patch_type == "ours_tnature" or options.patch_type == "ours_ttemplate" or options.patch_type == "badclip":
        options.patch_location = 'middle'
        options.patch_name = './backdoor/patterns/BadCLIP_vit_b.jpg' if '32' in options.model_name else './backdoor/patterns/badCLIP.jpg'
    else:
        #for blended-Random
        options.patch_width = 1
        options.patch_location = 'blended'
        options.noise_coeff = 0.2

    if '336' in options.model_name:
        options.image_size=336
        print(f"Image res: {options.image_size}")
    
    if options.eval_data_type != 'COCO':
        data = load_data(options, processor)
    else:
        data = {}
        data["train"] = None
        data["validation"] = None
        data["eval_test"] = None
           
    start_epoch = 0
    print(options.checkpoint)
    if(options.checkpoint is not None):
        if(os.path.isfile(options.checkpoint)):
            checkpoint  = torch.load(options.checkpoint, map_location = options.device)
            start_epoch = checkpoint['epoch'] 
            state_dict  = checkpoint["state_dict"]
            if(next(iter(state_dict.items()))[0].startswith("module")):
                state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
                model.load_state_dict(state_dict)
                print('Loaded full Model')

            logging.info(f"Loaded checkpoint '{options.checkpoint}' (start epoch {checkpoint['epoch']})")
        else:
            logging.info(f"No checkpoint found at {options.checkpoint}")

    cudnn.benchmark = True
    cudnn.deterministic = False

    metrics = evaluate(start_epoch, model, processor, data, options)
    #We want to write all attack parameters out with the result
    dictt = {}
    dictt['eval-data-type'] = options.eval_data_type
    dictt['model-loc'] = options.checkpoint
    dictt['backdoor'] = options.asr
    dictt['patch-type'] = options.patch_type
    dictt['patch-size'] = options.patch_size
    dictt['test-label'] = options.test_label
    dictt['stripe-width'] = options.patch_width
    dictt['noise_coeff'] = options.noise_coeff
    dictt['patch-name'] = options.patch_name

    saveFile = options.log_dir_path+ '/results_imnet_retrieval.txt'

    for key, value in metrics.items():
        # logging.info(f"{key}: {value:.4f}")
        dictt[f"{key}"] = f"{value:.4f}"
    
    with open(saveFile, "a+") as fp:
        json.dump(dictt, fp)  # encode dict into JSON
        fp.write("\n")

    print(f"Done writing dict into {saveFile} file")
        
if(__name__ == "__main__"):    
    options = parse_args()
    
    if '336' in options.model_name :
        #the difference of @ and - between openclip and CLIP
        options.model_name = 'ViT-L/14@336px'
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        options.logs = root

    options.log_dir_path = '/'.join(options.checkpoint.split("/")[:-1]) 
    
    main_eval(options)
