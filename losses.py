import torch
import torch.distributed.nn
import torch.nn as nn
import torch.nn.functional as F

import utils


def threshold_tensor(input_tensor, threshold_value, threshold_result=None):
    """
    Threshold a tensor, setting values above a threshold to a specific value.

    Args:
    - input_tensor (torch.Tensor): The input tensor to threshold.
    - threshold_value (float): The threshold value. All elements greater than this will be thresholded.
    - threshold_result (float, optional): The value to assign to elements above the threshold.
                                          If None, elements above the threshold will be set to the threshold value itself.

    Returns:
    - torch.Tensor: The thresholded tensor.
    """
    if threshold_result is None:
        threshold_result = threshold_value
    threshold_value = torch.tensor(threshold_value).to(input_tensor.device)
    threshold_result = torch.tensor(threshold_result).to(input_tensor.device)

    return torch.where(input_tensor > threshold_value, torch.tensor(threshold_result, dtype=input_tensor.dtype), input_tensor)


class CLIPLoss(nn.Module):
    def __init__(self, loss_thresh):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None
        self.loss_thresh = loss_thresh

    def forward(self, outputs, set_thresh=False):
        image_embed = outputs["clean_image_embed"]
        text_embed = outputs["text_embed"]
        image_embed_aug = outputs["aug_image_embed"]
        logit_scale = outputs["logit_scale"]
        local_batch_size = image_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = (
                local_batch_size * utils.get_rank()
                + torch.arange(
                    local_batch_size, device=image_embed.device
                )
            )
            self.last_local_batch_size = local_batch_size

        # normalized features
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)
        image_embed_aug = F.normalize(image_embed_aug, dim=-1, p=2)

        # gather with gradient
        image_embed_all = torch.cat(
            torch.distributed.nn.all_gather(image_embed), dim=0
        )
        image_embed_aug_all = torch.cat(
            torch.distributed.nn.all_gather(image_embed_aug), dim=0
        )

        text_embed_all = torch.cat(
            torch.distributed.nn.all_gather(text_embed), dim=0
        )
        text_embed_aug = F.normalize(outputs["aug_text_embed"], dim=-1, p=2)
        text_embed_aug_all = torch.cat(
        torch.distributed.nn.all_gather(text_embed_aug), dim=0
        )
        # cos_sim_clean = F.cosine_similarity(image_embed, text_embed, dim=1).mean()

        # cosine similarity as logits
        logits_per_image = (
            logit_scale * image_embed @ text_embed_all.t()
        )
        logits_per_text = (
            logit_scale * text_embed @ image_embed_all.t()
        )

        lossclip = (
            F.cross_entropy(logits_per_image, self.labels)
            + F.cross_entropy(logits_per_text, self.labels)
        ) / 2

        img_loss = threshold_tensor(compute_loss(loss_str='l2', embedding=image_embed_all, embedding_orig=image_embed_aug_all), self.loss_thresh, 0.0)
        text_loss = threshold_tensor(compute_loss(loss_str='l2', embedding=text_embed_all, embedding_orig=text_embed_aug_all), self.loss_thresh, 0.0)

        loss_sum = lossclip - 0.5 * (img_loss + text_loss)

        with torch.no_grad():
            pred = torch.argmax(logits_per_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_batch_size
            
        return {"clip-loss": lossclip, "clip_acc": acc, 'total-loss': loss_sum, 'img-loss': img_loss, 'text-loss': text_loss}

def l2(out, targets, reduction='none'):
    # squared l2 - it does not divide by the latent dimension
    # should have shape (batch_size, embedding_size)
    assert out.shape == targets.shape, f'{out.shape} != {targets.shape}'
    assert out.shape[0] > 1
    # Compute the element-wise squared error
    squared_error_batch = F.mse_loss(out, targets, reduction='none')
    if reduction == 'mean':
        squared_error_batch = torch.mean(squared_error_batch.sum(dim=1))
    else:
        squared_error_batch = squared_error_batch.sum(dim=1)
        assert squared_error_batch.shape == (out.shape[0],), f'{squared_error_batch.shape} != {(out.shape[0],)}'
    return squared_error_batch


def compute_loss(loss_str, embedding, embedding_orig, reduction='mean'):
    
    if loss_str == 'l2':
        loss = l2(out=embedding, targets=embedding_orig, reduction=reduction)

    elif loss_str == 'l1':
        loss = F.smooth_l1_loss(embedding.flatten(start_dim=1), embedding_orig.flatten(start_dim=1), reduction="none", beta=1.0).mean(dim=-1).mean()

    else:
        raise ValueError(f'loss {loss_str} not supported')

    return loss 
