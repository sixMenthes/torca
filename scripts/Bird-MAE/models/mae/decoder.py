import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import Block
from ..components.swin_transformer import SwinTransformerBlock
from ..components.swin_transformerv2 import SwinTransformerV2Block

class MAE_Decoder(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 decoder_embed_dim, 
                 decoder_depth, 
                 decoder_num_heads, 
                 decoder_mode,
                 mlp_ratio, 
                 norm_layer, 
                 num_patches,
                 pos_trainable,
                 patch_size,
                 in_chans,
                 no_shift,
                 target_length
        ):
        super().__init__()
        self.decoder_mode = decoder_mode
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=pos_trainable)
        self.no_shift = no_shift
        norm_layer = partial(nn.LayerNorm, eps=1e-6) # for both?
    
        if "swin" in self.decoder_mode: #!= 0 --> should be 0 in the original implementation
            decoder_modules = []
            window_size = (4,4)
            if target_length == 1024: #10 seconds
                feat_size = (64,8)
            elif target_length == 512: #5 seconds
                feat_size = (32,8)
            else:
                raise ValueError("Target length not supported for swin decoder")

            for i in range(decoder_depth): # is 16 for swin (but has practically the same result as 8 in the paper)
                if no_shift: # shift is true
                    shift_size = (0,0)
                else:
                    if (i % 2) == 0: # every second block
                        shift_size = (0,0)
                    else:
                        shift_size = (2,0)

                if self.decoder_mode == "swin":
                    decoder_modules.append(
                        SwinTransformerBlock(
                            dim=decoder_embed_dim,
                            num_heads=decoder_num_heads,
                            feat_size=feat_size,
                            window_size=window_size,
                            shift_size=shift_size,
                            mlp_ratio=mlp_ratio,
                            drop=0.0,
                            drop_attn=0.0,
                            drop_path=0.0,
                            extra_norm=False,
                            sequential_attn=False,
                            norm_layer=norm_layer
                        )
                    )
                elif self.decoder_mode == "swinv2":
                    decoder_modules.append(
                        SwinTransformerV2Block(
                            dim=decoder_embed_dim,
                            input_resolution=feat_size,
                            num_heads=decoder_num_heads,
                            window_size=window_size,
                            shift_size=shift_size,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True,
                            proj_drop=0.0,
                            attn_drop=0.0,
                            drop_path=0.0,
                            act_layer=nn.GELU,
                            norm_layer=nn.LayerNorm
                        )
                    )
                else:
                    raise ValueError("Decoder mode not supported")

            self.blocks = nn.ModuleList(decoder_modules)

        else:
            print("Decoder is normal transformer block")
            self.blocks = nn.ModuleList([
                    Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                    for i in range(decoder_depth)])
        
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)

    def forward(self, x, ids_restore):
        x = self.decoder_embed(x) # in:batch, x length +1, encoder_embed_dim
        #out: batch, length +1, decoder_embed_dim

        #append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        #add pos embed
        x = x + self.decoder_pos_embed

        if "swin" in self.decoder_mode: #!= 0 --> should be 0 in the original implementation
            B, L, D = x.shape # batch, length(patches), dim (decoder)
            x = x[:,1:,:]

    
        for blk in self.blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        #predictor projection
        pred = self.decoder_pred(x)

        if "swin" in self.decoder_mode:
            pred = pred
        else:
            pred = pred[:, 1:, :] # remove cls        

        return pred 