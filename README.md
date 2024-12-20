<div align="center">

<h3>Perturb and Recover: Fine-tuning for Effective Backdoor Removal from CLIP

</h3>

**Naman Deep Singh, Francesco Croce and Matthias Hein**

Paper: [arXiv](https://arxiv.org/abs/2412.00727)

<h4>Abstract</h4>
</div>

 Vision-Language models like CLIP have been shown to be highly effective at linking visual perception and natural language understanding, enabling sophisticated image-text capabilities, including strong retrieval and zero-shot classification performance. Their widespread use, as well as the fact that CLIP models are trained on image-text pairs from the web, make them both a worthwhile and relatively easy target for backdoor attacks. As training foundational models, such as CLIP, from scratch is very expensive, this paper focuses on cleaning potentially poisoned models via fine-tuning. We first show that existing cleaning techniques are not effective against simple structured triggers used in Blended or BadNet backdoor attacks, exposing a critical vulnerability for potential real-world deployment of these models. Then, we introduce PAR, Perturb and Recover, a surprisingly simple yet effective mechanism to remove backdoors from CLIP models. Through extensive experiments across different encoders and types of backdoor attacks, we show that PAR achieves high backdoor removal rate while preserving good standard performance. Finally, we illustrate that our approach is effective even only with synthetic text-image pairs, i.e. without access to real training data. 

---------------------------------


<div align="center">
<h4> Proposed Triggers</h4>
<p align="center"><img src="/asset/vis_triggers.png" width="400"> </br> The triggers can be used as-is in either BadNet or Blended attacks.</p>
</div>

---------------------------------

<div align="center">
<h4> Clean Accuracy- Attack success rate trade-off</h4>
<p align="center"><img src="/asset/tradeoff_curve.png" width="400"></p>
The proposed cleaning method, PAR yields a better ASR (Attack success rate)-CA (Clean accuracy) curve than the baselines. PAR works with both the real(CC3M) and synthetic data(SynC)

</div>


<h3>Train with PAR</h3>

- Installations, see `requirements.txt`

- Run bash [trigger.sh](trigger.sh)

Instructions:

- Add path to (image,caption) paired csv file in the variable `trainData`.

- Add `imgRoot`: directory where the train images are located - this is appended to image-filename in the csv at `trainData`.

- Add the path to posioned checkpoint in `modelPath` and encoder model-name in `modeln`. 

- Set `outDIR` - here the checkpoints and logs would be saved.


#### Evaluate
- Run `bash trigger_validate.sh checkpoint_path attack_name encoder_name target_label`

`attack_name` can be one of the following:
<ul>
  <li>badnet_rs (BadNet-Stripes) </li>
   <li>blended_rs (Blended-Stripes) </li>
   <li>tri_patt (Blended-Triangles) </li>
   <li>water_patt (Blended-Text) </li>
   <li>random (BadNet) </li>
   <li>badclip (BadCLIP) </li>
   <li>blended (Blended) </li>
 </ul>



<h4>Some poisoned/PAR-cleaned CLIP model checkpoints</h4>


<div align="center">
	
| Encoder name            | Backdoor-attack    | poisoned model | PAR-cleaned |    
|-------------------------|------------|-------------|-------------|
| ViT-L/14-336 | BadNet-Stripes | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/W83tntA6sFMDL8Z)     | --    |
| ViT-L/14-336 | Blended-Text | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/TqineSP7YsbaaMF)     |  --   |
| ViT-B/32 | BadNet-Stripes | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/Q6rnTj5bDKeKigp)     | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/EpKfgbbsCZJXCRx)     |
| ViT-B/32 | Blended-Triangles | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/XaZe8ZCgmM2p3Cf)     | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/g2zwG2F323eTMoT)     |
| ViT-B/32 | Blended-Text | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/GHKDMzizzmT5mk8)     | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/Qxc4FppPsmBHQK3)     |
-------------------------------------------------------------------------------------------------
</div>

Note: all of the above poisoned models are with `target_label` `banana`

_________________________________

The code in this repository is partially based on the following publically available codebases.

1. [https://github.com/nishadsinghi/CleanCLIP](https://github.com/nishadsinghi/CleanCLIP)
2. [https://github.com/LiangSiyuan21/BadCLIP](https://github.com/LiangSiyuan21/BadCLIP)

_________________________________

<h4>Citation</h4>

If you use our code/models cite our work using the following BibTex entry:
```bibtex
@article{singh2024PAR,
      title={Perturb and Recover: Fine-tuning for Effective Backdoor Removal from CLIP}, 
      author={Naman Deep Singh and Francesco Croce and Matthias Hein},
      journal = {arXiv preprint},
      year = {2024}
      url={https://arxiv.org/abs/2412.00727}, 
}
```
