import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


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
            text_input = self.tokenizer(text).to(self.device)
            text_feats = self.model.encode_text(text_input)
            if normalize:
                text_feats = F.normalize(text_feats, dim=-1)
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
            image_feats = self.model.encode_image(images.to(self.device))
            if normalize:
                image_feats = F.normalize(image_feats, dim=-1)
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
        if len(image_idx) != 0 and args.filter_image_idx:
            text_embeds = text_embeds[image_idx]
        scores = image_embeds @ text_embeds.T
        scores = scores.cpu().numpy()
        return scores

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
