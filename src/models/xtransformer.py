import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from einops import rearrange, pack, unpack

from src.custom_x_transformers.x_transformers import TransformerWrapper, Decoder, exists
from src.custom_x_transformers.autoregressive_wrapper import (
    top_k,
    top_p,
    top_a,
    eval_decorator,
)

from src.models.embeddings import VertexEmbeddings
from src.models.image_encoder import CustomSwinTransformerV2


import math


def init_models(cfg, **kwargs):
    """
    A bit of a superfunction to initialize a Seq2Seq model with a Transformer encoder and decoder
    Works on the XTransformer API and expects Image encoder and token-based auto-regressive decoder

    """

    if cfg.model.transformer_xformer.backbone == "swinv2":
        print("Using SWINV2 as the image encoder")

        encoder = CustomSwinTransformerV2(**cfg.model.transformer_xformer.swin_params)

    decoder = TransformerWrapper(
        **cfg.model.transformer_xformer.decoder,
        attn_layers=Decoder(
            **cfg.model.transformer_xformer.dec_attnlayers,
        ),
    )

    decoder = CustomAutoregressiveWrapper(
        decoder, mask_prob=cfg.model.transformer_xformer.misc_params.mask_prob
    )

    decoder.pad_value = 2
    decoder.ignore_index = 2

    if "custom_embeddings" in cfg.model.transformer_xformer:
        if cfg.model.transformer_xformer.custom_embeddings:
            print("Using custom embeddings for the decoder")
            decoder.net.ext_embed = VertexEmbeddings(
                **cfg.model.transformer_xformer.custom_embeddings_params
            )

    return encoder, decoder


class CustomAutoregressiveWrapper(nn.Module):
    def __init__(self, net, ignore_index=-100, pad_value=0, mask_prob=0.0):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        self.max_seq_len = net.max_seq_len

        # paper shows masking (MLM) in conjunction with autoregressive decoder-only training leads to big improvements https://arxiv.org/abs/2210.13432
        assert mask_prob < 1.0
        self.mask_prob = mask_prob

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        start_tokens,
        seq_len,
        eos_token=None,
        temperature=1.0,
        filter_logits_fn=top_k,
        filter_thres=0.9,
        min_p_pow=2.0,
        min_p_ratio=0.02,
        return_attn=False,
        **kwargs,
    ):

        start_tokens, ps = pack([start_tokens], "* n")

        attn_maps = []

        b, t = start_tokens.shape
        print(f" my logits fn is: {filter_logits_fn}")
        out = start_tokens

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len :]

            if not return_attn:
                logits = self.net(x, **kwargs)[:, -1]
            else:
                logits, attn = self.net(x, return_attn=return_attn, **kwargs)
                attn_maps.append(attn.cpu().numpy())
                logits = logits[:, -1]

            if filter_logits_fn in {top_k, top_p}:
                filtered_logits = filter_logits_fn(logits, thres=filter_thres)
                probs = F.softmax(filtered_logits / temperature, dim=-1)

            elif filter_logits_fn is top_a:
                filtered_logits = filter_logits_fn(
                    logits, min_p_pow=min_p_pow, min_p_ratio=min_p_ratio
                )
                probs = F.softmax(filtered_logits / temperature, dim=-1)

            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)

            if exists(eos_token):
                is_eos_tokens = out == eos_token

                if is_eos_tokens.any(dim=-1).all():
                    # mask out everything after the eos tokens
                    shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
                    mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
                    out = out.masked_fill(mask, self.pad_value)
                    break

        out = out[:, t:]

        (out,) = unpack(out, ps, "* n")

        if return_attn:
            return out, attn_maps

        return out

    def forward(
        self,
        x,
        return_logits=False,
        return_intermediates=False,
        return_attn=False,
        **kwargs,
    ):
        seq, ignore_index = x.shape[1], self.ignore_index

        inp, target = x[:, :-1], x[:, 1:]

        if self.mask_prob > 0.0:
            rand = torch.randn(inp.shape, device=x.device)
            rand[:, 0] = -torch.finfo(
                rand.dtype
            ).max  # first token should not be masked out
            num_mask = min(int(seq * self.mask_prob), seq - 1)
            indices = rand.topk(num_mask, dim=-1).indices
            mask = ~torch.zeros_like(inp).scatter(1, indices, 1.0).bool()
            kwargs.update(self_attn_context_mask=mask)

        if not return_attn:
            logits = self.net(inp, **kwargs)
        else:
            logits, attn = self.net(inp, return_attn=return_attn, **kwargs)

        loss = F.cross_entropy(
            rearrange(logits, "b n c -> b c n"), target, ignore_index=ignore_index
        )

        if return_attn:
            return loss, attn

        if return_logits:
            return loss, logits

        return loss


class XTransformerTrainer(pl.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        lr=1e-3,
        warmup_steps=1000,
        global_context=None,
        encoder_backbone=None,
        inference_type="greedy",
        **kwargs,
    ):
        super().__init__()

        assert (
            encoder_backbone is not None
        ), "Please specify the backbone for the image encoder"
        self.encoder = encoder
        self.decoder = decoder
        self.global_context = global_context
        self.encoder_backbone = encoder_backbone

        self.greedy = (
            inference_type == "greedy"
        )  # If not greedy, nucleus sampling is used

        self.warmup_steps = warmup_steps
        self.weight_decay = 0.1
        self.lr_drop = 1000
        self.lr = lr
        self.optimizer = self.configure_optimizers()

        self.local_stop_token = 1
        self.stop_token = 3
        self.max_seq_len = self.decoder.max_seq_len

        self.ext_embed = getattr(self.decoder.net, "ext_embed", None)
        if self.ext_embed is not None:
            self.decoder.net.ext_embed = self.decoder.net.ext_embed.to(self.device)
        self.lr = lr

    def forward(self, tgt_sequence, img, global_context=None, **kwargs):

        image_context = self.produce_image_context(img)

        if self.ext_embed is not None:
            if global_context is not None:
                tgt_embeddings = self.decoder.net.ext_embed(
                    tgt_sequence[:, :-1], global_context=global_context[:, :-1]
                )
            else:
                tgt_embeddings = self.decoder.net.ext_embed(tgt_sequence[:, :-1])
            return self.decoder(
                tgt_sequence, context=image_context, sum_embeds=tgt_embeddings, **kwargs
            )

        return self.decoder(tgt_sequence, context=image_context, **kwargs)

    def produce_image_context(self, img):

        if self.encoder_backbone == "swinv2":
            image_context = self.encoder.forward_features(img)
            bs, h, w, c = image_context.shape
            image_context = image_context.reshape(bs, h * w, c)
        elif self.encoder_backbone == "resnet50":
            raise NotImplementedError
        else:
            raise NotImplementedError

        return image_context

    def training_step(self, batch, batch_idx):

        tgt_sequence, img = batch["vertices_flat"].to(self.device).transpose(
            0, 1
        ), batch["image"].to(self.device)

        if self.global_context:
            global_context = batch["vert_obj_embeds"].transpose(0, 1).to(self.device)
        else:
            global_context = None

        loss = self(tgt_sequence=tgt_sequence, img=img, global_context=global_context)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=img.shape[0],
        )

        return loss

    def validation_step(self, batch, batch_idx):

        tgt_sequence, img = batch["vertices_flat"].to(self.device).transpose(
            0, 1
        ), batch["image"].to(self.device)

        if self.global_context:
            global_context = batch["vert_obj_embeds"].transpose(0, 1).to(self.device)
        else:
            global_context = None

        loss = self(tgt_sequence=tgt_sequence, img=img, global_context=global_context)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=img.shape[0],
        )

    def generate(
        self,
        start_tokens,
        seq_len,
        eos_token=None,
        local_eos_token=None,
        temperature=1.0,
        filter_logits_fn=top_p,
        filter_thres=0.95,
        min_p_pow=2.0,
        min_p_ratio=0.02,
        greedy=False,
        external_embeddings=None,
        return_probs=False,
        debug_context=None,
        return_multi_object=False,
        num_object_embeddings=50,
        return_attn=False,
        **kwargs,
    ):

        device = self.device

        start_tokens, ps = pack([start_tokens], "* n")

        b, t = start_tokens.shape
        out = start_tokens

        multi_object_inference = kwargs.get("multi_object", False)
        global_context = None

        if multi_object_inference:
            global_context = torch.ones((b, seq_len), device=device).type(
                torch.int64
            )  # Pre-populate the object based context
        elif debug_context:
            global_context = debug_context

        kwargs.pop("multi_object", None)
        kwargs.pop("alt_ext", None)  # Don't want to pass this further down

        if return_probs:
            logits_out = []

        attn_maps = []

        for _ in range(seq_len):
            x = out[:, -self.decoder.max_seq_len :]

            if not external_embeddings:

                if not return_attn:
                    logits = self.decoder.net(x, **kwargs)[:, -1]
                else:
                    logits, attn = self.decoder.net(
                        x, return_attn=return_attn, **kwargs
                    )

                    attn_maps = [x.cpu() for x in attn]
                    attn_maps.append(attn_maps)
                    logits = logits[:, -1]

            else:
                #
                if not multi_object_inference:
                    external_embeds = self.decoder.net.ext_embed(x)
                else:
                    curr_seq_len = x.size(1)
                    external_embeds = self.decoder.net.ext_embed(
                        x, global_context=global_context[:, :(curr_seq_len)]
                    )
                    if kwargs.get("alt_ext", False):
                        external_embeds = self.decoder.net.ext_embed(
                            x, global_context=global_context
                        )

                logits = self.decoder.net(x, sum_embeds=external_embeds, **kwargs)[
                    :, -1
                ]

            if return_probs:
                logits_out.append(logits.cpu().unsqueeze(1))

            if greedy:

                sample = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # The logic for sampling with different filtering methods
                if filter_logits_fn in {top_k, top_p}:
                    filtered_logits = filter_logits_fn(logits, thres=filter_thres)
                    probs = F.softmax(filtered_logits / temperature, dim=-1)
                elif filter_logits_fn is top_a:
                    filtered_logits = filter_logits_fn(
                        logits, min_p_pow=min_p_pow, min_p_ratio=min_p_ratio
                    )
                    probs = F.softmax(filtered_logits / temperature, dim=-1)

                sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)

            if multi_object_inference:

                # Identify which sequences in the batch have produced the local stop token at the last timestep
                local_stop_detected = out[:, -1] == local_eos_token

                # Check if 'local_eos_token' is present in any sequence of the batch
                if local_stop_detected.any():
                    indices = torch.where(local_stop_detected)[0]
                    global_context[
                        indices, (curr_seq_len + 1) :
                    ] += 1  # Here we are adding 1 to the global context of sequences which produced a local stop token at the current timestep.
                    global_context = global_context.clip(
                        min=0, max=num_object_embeddings
                    )  # Clip the global context to 2, so that it can only take values 0, 1, or 2.

            if exists(eos_token):
                is_eos_tokens = out == eos_token
                if is_eos_tokens.any(dim=-1).all():
                    # mask out everything after the eos tokens
                    shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
                    mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
                    out = out.masked_fill(mask, self.decoder.pad_value)
                    break

        out = out[:, t:]
        (out,) = unpack(out, ps, "* n")

        if return_probs:
            return out, logits_out

        if return_attn:
            return out, attn_maps

        if return_multi_object:
            return out, global_context

        return out

    def predict_step(self, batch, batch_idx, dataloader_idx=None):

        tgt_sequence, img = batch["vertices_flat"].to(self.device), batch["image"].to(
            self.device
        )

        # if self.global_context:
        #     global_context = batch['vert_obj_embeds'].transpose(0,1).to(self.device)
        # else:
        #     global_context = None

        image_context = self.produce_image_context(img)

        img_ids = [int(x.cpu().numpy()) for x in batch["image_id"]]
        gt_vert_list = batch["vertices"]

        gen_samples = self.generate(
            start_tokens=tgt_sequence[[0], :].to(self.device).transpose(0, 1),
            seq_len=torch.tensor([self.max_seq_len]).to(self.device),
            eos_token=torch.tensor([self.stop_token]).to(self.device),
            local_eos_token=(
                torch.tensor([self.local_stop_token]).to(self.device)
                if self.local_stop_token
                else None
            ),
            temperature=1.0,
            filter_logits_fn=top_p,
            filter_thres=0.95,
            min_p_pow=2.0,
            min_p_ratio=0.02,
            context=image_context,
            greedy=self.greedy,
            external_embeddings=self.ext_embed,
            num_object_embeddings=None,
            multi_object=False,
        )

        return {
            "image_ids": img_ids,
            "gen_samples": gen_samples.detach().cpu(),
            "gt_verts": gt_vert_list,
            "inference_type": "greedy" if self.greedy else "nucleus",
        }

    def configure_optimizers(self):
        self.optimizer = AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: 1 / math.sqrt(epoch + 1)
        )

        return [self.optimizer], [self.scheduler]

    def on_train_epoch_start(self):
        if self.current_epoch < self.warmup_steps:
            lr_scale = float(self.current_epoch + 1) / float(self.warmup_steps)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr
