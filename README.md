<div align="center">

<h3>Perturb and Recover: Fine-tuning for Effective Backdoor Removal from CLIP</h3>
</div>


For Requirements, see requirements.txt

Training PAR

bash trigger.sh 
Instructions:
-Add the path to train-csv and image-root directory in trigger.sh
-Add the path to posioned checkpiiunt and encoder name(RN50, ViT-B/32, ViT-L-14-336) 
-This would train for standard setup and evaluate for the attack param in trigger.sh


Evaluation:
bash trigger_validate.sh checkpiint_path attack_name encoder_name target_label

