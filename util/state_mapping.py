def create_block_mapping(blocks, prefix, prefix1=""):
    return {
        **{f"{prefix1}blocks.{i}.norm1.weight": f"{prefix}.blocks.{i}.norm1.weight" for i in range(blocks)},
        **{f"{prefix1}blocks.{i}.norm1.bias": f"{prefix}.blocks.{i}.norm1.bias" for i in range(blocks)},
        **{f"{prefix1}blocks.{i}.attn.qkv.weight": f"{prefix}.blocks.{i}.attn.qkv.weight" for i in range(blocks)},
        **{f"{prefix1}blocks.{i}.attn.qkv.bias": f"{prefix}.blocks.{i}.attn.qkv.bias" for i in range(blocks)},
        **{f"{prefix1}blocks.{i}.attn.proj.weight": f"{prefix}.blocks.{i}.attn.proj.weight" for i in range(blocks)},
        **{f"{prefix1}blocks.{i}.attn.proj.bias": f"{prefix}.blocks.{i}.attn.proj.bias" for i in range(blocks)},
        **{f"{prefix1}blocks.{i}.norm2.weight": f"{prefix}.blocks.{i}.norm2.weight" for i in range(blocks)},
        **{f"{prefix1}blocks.{i}.norm2.bias": f"{prefix}.blocks.{i}.norm2.bias" for i in range(blocks)},
        **{f"{prefix1}blocks.{i}.mlp.fc1.weight": f"{prefix}.blocks.{i}.mlp.fc1.weight" for i in range(blocks)},
        **{f"{prefix1}blocks.{i}.mlp.fc1.bias": f"{prefix}.blocks.{i}.mlp.fc1.bias" for i in range(blocks)},
        **{f"{prefix1}blocks.{i}.mlp.fc2.weight": f"{prefix}.blocks.{i}.mlp.fc2.weight" for i in range(blocks)},
        **{f"{prefix1}blocks.{i}.mlp.fc2.bias": f"{prefix}.blocks.{i}.mlp.fc2.bias" for i in range(blocks)}
    }

def map_amae_checkpoint(original_state_dict, encoder_blocks=12, decoder_blocks=12):
    encoder_mapping = create_block_mapping(encoder_blocks, "encoder")
    decoder_mapping = create_block_mapping(decoder_blocks, "decoder", prefix1="decoder_")

    # Add top-level mappings for the encoder and decoder
    encoder_mapping.update({
        "cls_token": "encoder.cls_token",
        "pos_embed": "encoder.pos_embed",
        "patch_embed.proj.weight": "encoder.patch_embed.proj.weight",
        "patch_embed.proj.bias": "encoder.patch_embed.proj.bias",
        "norm.weight": "encoder.norm.weight",
        "norm.bias": "encoder.norm.bias"
    })

    decoder_mapping.update({
        "mask_token": "decoder.mask_token",
        "decoder_pos_embed": "decoder.decoder_pos_embed",
        "decoder_embed.weight": "decoder.decoder_embed.weight",
        "decoder_embed.bias": "decoder.decoder_embed.bias",
        "decoder_pred.weight": "decoder.decoder_pred.weight",
        "decoder_pred.bias": "decoder.decoder_pred.bias",
        "decoder_norm.weight": "decoder.decoder_norm.weight",
        "decoder_norm.bias": "decoder.decoder_norm.bias",
    })

    # Concatenate both mappings into a single dictionary
    full_mapping = {**encoder_mapping, **decoder_mapping}

    remapped_state_dict = {}

    # Iterate through the keys in the checkpoint and apply the mapping if it exists
    for old_key, value in original_state_dict.items():
        if old_key in full_mapping:
            new_key = full_mapping[old_key]  # Use the new key from the mapping
            remapped_state_dict[new_key] = value
        else:
            # If the key does not need remapping, keep it as is
            remapped_state_dict[old_key] = value

    return remapped_state_dict