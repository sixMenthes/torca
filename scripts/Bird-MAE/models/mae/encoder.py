import torch
import torch.nn as nn
from functools import partial
from util.patch_embed import PatchEmbed_org
from timm.models.vision_transformer import Block


class MAE_Encoder(nn.Module):
    def __init__(self, 
                 img_size_x,
                 img_size_y, 
                 patch_size, 
                 in_chans, 
                 embed_dim, 
                 depth, 
                 num_heads, 
                 mlp_ratio, 
                 norm_layer, 
                 pos_trainable
                 ):
        super().__init__()
        # input: (Batch, Channel, Height 128, Width 1024)
        # output: (Batch, #Patch, Embed)
        self.patch_embed = PatchEmbed_org((img_size_x, img_size_y), patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # 1, 1, 768
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=pos_trainable)

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
    
    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length(number of patches), dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1) # restore the oriignal order

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def forward(self, x, mask_ratio):
        # embed patches through encoder
        x = self.patch_embed(x) # batch size, 1, width, height

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1) # expands on batch
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore