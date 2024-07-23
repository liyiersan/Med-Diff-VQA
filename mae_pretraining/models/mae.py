# modified from https://github.com/facebookresearch/mae/blob/main/models_mae.py 
# and https://github.com/Sense-X/MixMIM/blob/master/models_mixmim.py

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block, Attention, Mlp, Optional

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils.pos_embed import get_2d_sincos_pos_embed

class MaskAttention(Attention):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__(dim, num_heads, qkv_bias, qk_norm, attn_drop, proj_drop, norm_layer)
        
    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
            x: [N, L, D]
            mask: [N, L], binary mask, 0 means from img1, 1 means from img2
        """
        if mask is None:
            return super().forward(x)
        else:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
            
            q = q * self.scale
            attn = q @ k.transpose(-2, -1) # [B, H, N, N]
            
            # mask attention, for each mixed sample, the self-attention should be conducted within the same group of patches, i.e., have the same mask
            mask = mask.reshape(B, 1, 1, N)
            mask_new = mask * mask.transpose(2, 3) + (1 - mask) * (1 - mask).transpose(2, 3)
            mask_new = 1 - mask_new
            if mask_new.dtype == torch.float16:
                attn = attn - 65500 * mask_new
            else:
                attn = attn - 1e30 * mask_new
            
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            
            return x

class MaskedBlock(Block):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_norm, proj_drop, attn_drop, init_values, drop_path, act_layer, norm_layer, mlp_layer)
        self.attn = MaskAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
    
    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, mask_strategy='random'):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            MaskedBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.mask_strategy = mask_strategy
        assert mask_strategy in ['random', 'mixed', 'dual'], f"Unknown random strategy: {mask_strategy}, should be 'random', 'mixed', or 'dual'"
        
        if self.mask_strategy == "dual":
            self.cls_token_other = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        
        if self.mask_strategy == "dual":
            torch.nn.init.normal_(self.cls_token_other, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    
    def gen_mask(self, x, mask_ratio):
        """
        x: [N, L, C]
        mask_ratio: float, ratio of masked tokens
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore) # [N, L]
        
        return mask, ids_shuffle, ids_restore
        
    def mix_masking(self, x, mask_ratio=0.5):
        """
        ideas from MixMAE: Mixed and Masked Autoencoder for Efficient Pretraining of Hierarchical Vision Transformers
        x: [N, L, C]
        """
        B = x.shape[0] // 2
        x_reordered = torch.cat([x[B:], x[:B]], dim=0)  # [N, L, C]
        
        mask, ids_shuffle, ids_restore = self.gen_mask(x, mask_ratio)
        
        mask_expanded = mask.unsqueeze(-1).repeat(1, 1, x.shape[-1])  # [N, L, D]
        
        x_mix = x * (1 - mask_expanded) + x_reordered *  mask_expanded
        
        return x_mix, mask

    def dual_masking(self, x, mask_ratio=0.5):
        B = x.shape[0] // 2
        x_ref, x_study = x[B:], x[:B]
        
        mask, ids_shuffle, ids_restore = self.gen_mask(x, mask_ratio)
        
        mask_expanded = mask.unsqueeze(-1).repeat(1, 1, x.shape[-1])  # [N, L, D]
        
        x_mix = x_ref * (1 - mask_expanded) + x_study *  mask_expanded
        
        return x_mix, mask
        
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        
        mask, ids_shuffle, ids_restore = self.gen_mask(x, mask_ratio)
        len_keep = int(x.shape[1] * (1 - mask_ratio))

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        
        if self.mask_strategy == "mixed": # for mixed masking, mask first, add_pos_embed later
            x, mask = self.mix_masking(x, mask_ratio)
            ids_restore = None
        elif self.mask_strategy == "dual":
            x, mask = self.dual_masking(x, mask_ratio)
            ids_restore = None

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        if self.mask_strategy == "random":
            # masking: length -> length * mask_ratio
            x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            if self.mask_strategy == "dual":
                x = blk(x, mask)
            else:
                x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, mask, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        
        if self.mask_strategy == "random":
            # append mask tokens to sequence
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
            
        elif self.mask_strategy == "mixed":
            x_ = x[:, 1:, :] # no cls token
            N, L, D = x_.shape
            mask_tokens = self.mask_token.expand(N, L, -1)
            mask_expanded = mask.unsqueeze(-1).repeat(1, 1, D)
            # we do not use the dual reconstruction loss as we may set a higher mask_ratio than 0.5
            x_ = x_ * (1 - mask_expanded) + mask_tokens * mask_expanded
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
            
        elif self.mask_strategy == "dual":
            x_ = x[:, 1:, :]
            N, L, D = x_.shape
            mask_tokens = self.mask_token.expand(N, L, -1)
            mask_expanded = mask.unsqueeze(-1).repeat(1, 1, D)
            
            x_ref = x_ * (1 - mask_expanded) + mask_tokens * mask_expanded
            x_study = x_ * mask_expanded + mask_tokens * (1 - mask_expanded)
            
            x_ref = torch.cat([x[:, :1, :], x_ref], dim=1)  # append cls token
            x_study = torch.cat([x[:, :1, :], x_study], dim=1)  # append cls token
            
            x = torch.cat([x_ref, x_study], dim=0)
            
            
        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
            
        if self.mask_strategy == "dual":
            # dual reconstruction loss
            N, L, D = pred.shape[0]
            B = N // 2
            
            pred_ref, pred_study = pred[:B], pred[B:]
            mask_expanded = mask.unsqueeze(-1).repeat(1, 1, D)
            
            unmix_pred = pred_ref * mask_expanded + pred_study * (1 - mask_expanded)
            
            loss = (unmix_pred - target) ** 2
            loss = loss.mean()    
        else:
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
            return loss
            
    def forward(self, imgs, mask_ratio=0.75):
        if isinstance(imgs, list) or isinstance(imgs, tuple): # multiple images, one for main, one for ref
            imgs = torch.vstack(imgs) # [N, 3, H, W]
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, mask, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
        


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks


if __name__ == '__main__':
    torch.manual_seed(0)
    import time
    input_data = torch.randn(2, 3, 224, 224)
    
    time1 = time.time()
    mae_model = mae_vit_base_patch16(mask_strategy='random')
    loss, pred, mask = mae_model(input_data, mask_ratio=0.75)
    time2 = time.time()
    print('mae time:', time2 - time1)
    print(loss, pred.shape, mask.shape)
    
    time1 = time.time()
    loss, pred, mask = mae_model(input_data, mask_ratio=0.9)
    time2 = time.time()
    print('med mae time:', time2 - time1)
    print(loss, pred.shape, mask.shape)
    
    time1 = time.time()
    mae_model = mae_vit_base_patch16(mask_strategy='mixed')
    loss, pred, mask = mae_model(input_data, mask_ratio=0.5)
    time2 = time.time()
    print('mixed mae time:', time2 - time1)
    print(loss, pred.shape, mask.shape)
    
    time1 = time.time()
    mae_model = mae_vit_base_patch16(mask_strategy='dual')
    loss, pred, mask = mae_model(input_data, mask_ratio=0.5)
    time2 = time.time()
    print('dual mae time:', time2 - time1)
    print(loss, pred.shape, mask.shape)