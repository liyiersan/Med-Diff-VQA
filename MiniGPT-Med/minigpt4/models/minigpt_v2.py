import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from minigpt4.common.registry import registry
from minigpt4.models.base_model import disabled_train
from minigpt4.models.minigpt_base import MiniGPTBase
from minigpt4.models.Qformer import BertConfig, BertLMHeadModel

class Projector(nn.Module):
    def __init__(self, mae_dim, sam_dim, llama_dim):
        super().__init__()
        self.mae_proj = nn.Linear(mae_dim, llama_dim)
        self.sam_proj = nn.Linear(sam_dim, llama_dim)
        self.conv = nn.Conv1d(llama_dim, llama_dim, 3, 2, 0)
        
    def forward(self, img_features):
        if isinstance(img_features, tuple) or isinstance(img_features, list):
            mae, sam = img_features[0], img_features[1]
            mae = self.mae_proj(mae)
            sam = self.sam_proj(sam)
            img_feats = torch.cat([mae, sam], dim=-2) # [bs, 305, 4096]
            # use conv to reduce the length of the sequence
            img_feats = self.conv(img_feats.permute(0, 2, 1)).permute(0, 2, 1) # [bs, 153, 4096]
        elif img_features.shape[-1] == 3072:
            img_feats = self.mae_proj(img_features) # [bs, 49, 4096]
        elif img_features.shape[-1] == 4096:
            img_feats = self.sam_proj(img_features) # [bs, 256, 4096]
        return img_feats

@registry.register_model("minigpt_v2")
class MiniGPTv2(MiniGPTBase):
    """
    MiniGPT-v2 model
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/minigpt_v2.yaml",
    }

    def __init__(
            self,
            vit_model="eva_clip_g",
            img_size=448,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            llama_model="",
            prompt_template='[INST] {} [/INST]',
            max_txt_len=300,
            end_sym='\n',
            lora_r=64,
            lora_target_modules=["q_proj", "v_proj"],
            lora_alpha=16,
            lora_dropout=0.05,
            chat_template=False,
            use_grad_checkpoint_llm=False,
            max_context_len=3800,
            low_resource=False,  # use 8 bit and put vit in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    ):
        super().__init__(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            llama_model=llama_model,
            max_txt_len=max_txt_len,
            max_context_len=max_context_len,
            end_sym=end_sym,
            prompt_template=prompt_template,
            low_resource=low_resource,
            device_8bit=device_8bit,
            lora_r=lora_r,
            lora_target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        img_f_dim = self.visual_encoder.num_features * 4 if not isinstance(self.visual_encoder, nn.Identity) else 5632
        self.llama_proj = nn.Linear(
            img_f_dim, self.llama_model.config.hidden_size
        )
        self.proj = Projector(3072, 4096, self.llama_model.config.hidden_size)
        self.chat_template = chat_template
        self.ref_embed_bias = nn.Parameter(torch.randn(1, 1, self.llama_model.config.hidden_size))
        self.study_embed_bias = nn.Parameter(torch.randn(1, 1, self.llama_model.config.hidden_size))

        if use_grad_checkpoint_llm:
            self.llama_model.gradient_checkpointing_enable()

    def encode_img(self, image):
        device = image.device

        if len(image.shape) > 4:
            image = image.reshape(-1, *image.shape[-3:])

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_embeds = image_embeds[:, 1:, :]
            bs, pn, hs = image_embeds.shape
            image_embeds = image_embeds.view(bs, int(pn / 4), int(hs * 4))

            inputs_llama = self.llama_proj(image_embeds)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama
    
    def proj_features(self, img_features):
        feats = self.proj(img_features)
        attns = torch.ones(feats.size()[:-1], dtype=torch.long).to(feats.device)
        return feats, attns

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        low_resource = cfg.get("low_resource", False)

        prompt_template = cfg.get("prompt_template", '[INST] {} [/INST]')
        max_txt_len = cfg.get("max_txt_len", 300)
        end_sym = cfg.get("end_sym", '\n')

        lora_r = cfg.get("lora_r", 64)
        lora_alpha = cfg.get("lora_alpha", 16)
        chat_template = cfg.get("chat_template", False)

        use_grad_checkpoint_llm = cfg.get("use_grad_checkpoint_llm", False)
        max_context_len = cfg.get("max_context_len", 3800)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            llama_model=llama_model,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            low_resource=low_resource,
            end_sym=end_sym,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            chat_template=chat_template,
            use_grad_checkpoint_llm=use_grad_checkpoint_llm,
            max_context_len=max_context_len,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load Minigpt-4-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
            assert len(msg.unexpected_keys) == 0, f"Unexpected keys: {msg.unexpected_keys}"

        return model
    
   