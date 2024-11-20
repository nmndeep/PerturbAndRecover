Perturb and Recover: Fine-tuning for Effective Backdoor Removal from CLIP


For Requirements, see requirements.txt

#### Training PAR

bash trigger.sh 

Instructions:

-Add the path to train-csv and image-root directory in trigger.sh

-Add the path to posioned checkpiiunt and encoder model-name (RN50, ViT-B/32, ViT-L-14-336), the resolution and attack parameters would be set automatically in the script. 

-This would train for standard setup and evaluate for the attack parameters as in trigger.sh


#### Evaluation:

bash trigger_validate.sh checkpiint_path attack_name encoder_name target_label

