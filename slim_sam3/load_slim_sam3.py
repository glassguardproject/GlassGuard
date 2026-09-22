#!/usr/bin/env python3
"""
Load a structurally-pruned SAM3 model (slim encoder) for inference.

Builds the full SAM3 architecture with a depth-12 ViT (matching the 12 kept
blocks from the slim checkpoint), loads the remapped state dict, and returns
a ready-to-use Sam3ImagePredictor.

Usage from Python:
    from load_slim_sam3 import build_slim_sam3_image_model
    predictor = build_slim_sam3_image_model(
        slim_ckpt="sam3_light_encoder_out/sam3_slim_enc_keep12of32.pt",
        meta_json="sam3_light_encoder_out/slim_encoder_meta.json",
        device="cuda",
    )

Or run directly to validate:
    conda run -n sam3 python3 load_slim_sam3.py \
        --slim-ckpt sam3_light_encoder_out/sam3_slim_enc_keep12of32.pt \
        --meta-json sam3_light_encoder_out/slim_encoder_meta.json \
        --device cpu
"""
import argparse
import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# SAM3 model components
from sam3.model.decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
)
from sam3.model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from sam3.model.geometry_encoders import SequenceGeometryEncoder
from sam3.model.maskformer_segmentation import PixelDecoder, UniversalSegmentationHead
from sam3.model.model_misc import (
    DotProductScoring,
    MLP,
    MultiheadAttentionWrapper as MultiheadAttention,
    TransformerWrapper,
)
from sam3.model.necks import Sam3DualViTDetNeck
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.model.sam3_image import Sam3Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.vitdet import ViT
from sam3.model.vl_combiner import SAM3VLBackbone


# ---------------------------------------------------------------------------
# Helpers to replicate model_builder._create_* without text encoder
# ---------------------------------------------------------------------------

def _create_position_encoding(precompute_resolution=None, device="cpu"):
    # precompute_resolution hardcodes device="cuda" in PositionEmbeddingSine,
    # so skip precomputation when running on CPU.
    effective_precompute = precompute_resolution if device == "cuda" else None
    return PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=effective_precompute,
    )


def _create_slim_vit(
    keep_blocks: list[int],
    mlp_hidden_dim: Optional[int] = None,
) -> ViT:
    """
    Build a ViT with depth == len(keep_blocks), with global_att_blocks remapped
    from the original SAM3 ViT-L schedule (7, 15, 23, 31) to new indices.

    Args:
        keep_blocks:    Original block indices that are kept (sorted).
        mlp_hidden_dim: If provided, overrides the default MLP hidden dimension
                        (4736 for ViT-L).  Useful when loading a checkpoint that
                        has had its MLP channels physically pruned.  The ViT is
                        built with mlp_ratio = mlp_hidden_dim / embed_dim so that
                        the architecture matches the pruned weight shapes.
                        Only uniform (same for all blocks) pruning is supported;
                        for per-block pruning load weights with strict=False.
    """
    embed_dim = 1024
    orig_global_att = {7, 15, 23, 31}
    keep_sorted = sorted(set(keep_blocks))
    depth = len(keep_sorted)

    # Compute mlp_ratio
    if mlp_hidden_dim is not None:
        mlp_ratio = mlp_hidden_dim / embed_dim
        print(f"[SlimViT] Custom mlp_hidden_dim={mlp_hidden_dim}, mlp_ratio={mlp_ratio:.4f}")
    else:
        mlp_ratio = 4.625  # SAM3 ViT-L default (embed_dim=1024 → hidden=4736)

    # Remap global attention blocks to new sequential indices
    new_global_att = tuple(
        new_i
        for new_i, orig_i in enumerate(keep_sorted)
        if orig_i in orig_global_att
    )
    # Guarantee at least one global-attn block at the end
    if not new_global_att:
        new_global_att = (depth - 1,)

    print(f"[SlimViT] depth={depth}, global_att_blocks={new_global_att}")

    return ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=16,
        mlp_ratio=mlp_ratio,
        norm_layer="LayerNorm",
        drop_path_rate=0.0,        # No stochastic depth at inference
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=new_global_att,
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=None,
    )


def _apply_per_block_mlp_dims(
    vit_backbone: ViT,
    keep_blocks: list[int],
    per_block_hidden_dims: dict[int, int],
) -> None:
    """
    For adaptive MLP pruning, rebuild each kept block's fc1/fc2 with the
    block-specific hidden dim from meta JSON.

    keep_blocks maps slim sequential index -> original block index.
    per_block_hidden_dims is keyed by original block index.
    """
    trunk_blocks = vit_backbone.blocks
    if len(trunk_blocks) != len(keep_blocks):
        raise ValueError(
            f"Block count mismatch: trunk has {len(trunk_blocks)} blocks, "
            f"keep_blocks has {len(keep_blocks)}"
        )

    for slim_idx, orig_blk in enumerate(keep_blocks):
        if orig_blk not in per_block_hidden_dims:
            continue
        target_hidden = int(per_block_hidden_dims[orig_blk])
        block = trunk_blocks[slim_idx]
        mlp = getattr(block, "mlp", None)
        if mlp is None:
            continue
        fc1 = getattr(mlp, "fc1", None)
        fc2 = getattr(mlp, "fc2", None)
        if fc1 is None or fc2 is None:
            continue

        current_hidden = int(fc1.out_features)
        if current_hidden == target_hidden:
            continue

        has_fc1_bias = fc1.bias is not None
        has_fc2_bias = fc2.bias is not None
        embed_dim_in = int(fc1.in_features)
        embed_dim_out = int(fc2.out_features)

        mlp.fc1 = nn.Linear(embed_dim_in, target_hidden, bias=has_fc1_bias)
        mlp.fc2 = nn.Linear(target_hidden, embed_dim_out, bias=has_fc2_bias)


def _apply_per_block_head_dims(
    vit_backbone: ViT,
    keep_blocks: list[int],
    kept_heads_per_block: dict[int, list[int]],
    orig_num_heads: int,
    head_dim: int,
) -> None:
    """
    RoPE-safe attention head pruning at load time: for each kept block, rebuild
    attn.qkv and attn.proj to the pruned head count and set attn.num_heads.

    The residual-stream dim is unchanged; only the internal head count shrinks.
    freqs_cis / head_dim are left untouched (head_dim is constant), so 2D RoPE
    keeps working. keep_blocks maps slim sequential index -> original block index;
    kept_heads_per_block is keyed by original block index.
    """
    trunk_blocks = vit_backbone.blocks
    if len(trunk_blocks) != len(keep_blocks):
        raise ValueError(
            f"Block count mismatch: trunk has {len(trunk_blocks)}, keep_blocks {len(keep_blocks)}")

    for slim_idx, orig_blk in enumerate(keep_blocks):
        if orig_blk not in kept_heads_per_block:
            continue
        n_keep = len(kept_heads_per_block[orig_blk])
        if n_keep == orig_num_heads:
            continue
        attn = getattr(trunk_blocks[slim_idx], "attn", None)
        if attn is None:
            continue

        dim = attn.qkv.in_features                  # residual stream dim (unchanged)
        has_qkv_bias = attn.qkv.bias is not None
        has_proj_bias = attn.proj.bias is not None

        attn.qkv = nn.Linear(dim, 3 * n_keep * head_dim, bias=has_qkv_bias)
        attn.proj = nn.Linear(n_keep * head_dim, dim, bias=has_proj_bias)
        attn.num_heads = n_keep
        # attn.head_dim stays `head_dim`; forward infers it from qkv output and
        # uses self.num_heads for the reshape, so RoPE/freqs_cis are unaffected.


def _create_vit_neck(position_encoding, vit_backbone):
    return Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit_backbone,
        add_sam2_neck=False,
    )


def _create_transformer_encoder():
    encoder_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=MultiheadAttention(num_heads=8, dropout=0.1, embed_dim=256, batch_first=True),
        cross_attention=MultiheadAttention(num_heads=8, dropout=0.1, embed_dim=256, batch_first=True),
    )
    return TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=6,
        d_model=256,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=False,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )


def _create_transformer_decoder(device="cpu"):
    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        cross_attention=MultiheadAttention(num_heads=8, dropout=0.1, embed_dim=256),
        n_heads=8,
        use_text_cross_attention=True,
    )
    # Pass resolution=None to avoid the hardcoded device="cuda" coord pre-computation
    # in TransformerDecoder.__init__. Coords are computed lazily at forward time.
    return TransformerDecoder(
        layer=decoder_layer,
        num_layers=6,
        num_queries=200,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=256,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        resolution=None,   # avoids cuda pre-computation; coords built lazily
        stride=None,
        use_act_checkpoint=False,
        presence_token=True,
    )


def _create_segmentation_head():
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3,
        interpolation_mode="nearest",
        hidden_dim=256,
        compile_mode=None,
    )
    cross_attend_prompt = MultiheadAttention(num_heads=8, dropout=0, embed_dim=256)
    return UniversalSegmentationHead(
        hidden_dim=256,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        act_ckpt=False,
        cross_attend_prompt=cross_attend_prompt,
        pixel_decoder=pixel_decoder,
    )


def _create_geometry_encoder(device="cpu"):
    geo_pos_enc = _create_position_encoding(device=device)
    from sam3.model.memory import CXBlock
    cx_block = CXBlock(dim=256, kernel_size=7, padding=3, layer_scale_init_value=1e-6, use_dwconv=True)
    geo_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=MultiheadAttention(num_heads=8, dropout=0.1, embed_dim=256, batch_first=False),
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        cross_attention=MultiheadAttention(num_heads=8, dropout=0.1, embed_dim=256, batch_first=False),
    )
    return SequenceGeometryEncoder(
        pos_enc=geo_pos_enc,
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=256,
        num_layers=3,
        layer=geo_layer,
        use_act_ckpt=False,
        add_cls=True,
        add_post_encode_proj=True,
    )


def _create_dot_product_scoring():
    prompt_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    return DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_slim_sam3_image_model(
    slim_ckpt: str,
    keep_blocks: Optional[list[int]] = None,
    meta_json: Optional[str] = None,
    device: str = "cpu",
    eval_mode: bool = True,
    mlp_hidden_dim: Optional[int] = None,
) -> Sam3Processor:
    """
    Build a SAM3 image model with the slim encoder and load from a slim checkpoint.

    Args:
        slim_ckpt:      Path to slim checkpoint (e.g. sam3_slim_enc_keep24of32.pt).
        keep_blocks:    List of original block indices that were kept.  If None,
                        read from meta_json.
        meta_json:      Path to slim_encoder_meta.json (used if keep_blocks is None).
        device:         'cpu' or 'cuda'
        eval_mode:      Set model to eval mode.
        mlp_hidden_dim: Optional.  If the slim checkpoint has been further pruned
                        by prune_mlp_channels.py, pass the new MLP hidden dimension
                        here (e.g. 3072 or 2048) so that the ViT architecture is
                        built with matching weight shapes.  If None the default
                        SAM3 ViT-L hidden dim of 4736 is used.

    Returns:
        Sam3Processor wrapping the slim Sam3Image model.
    """
    # Resolve keep_blocks and optional adaptive per-block hidden dims
    meta = None
    per_block_hidden_dims: Optional[dict[int, int]] = None
    if keep_blocks is None:
        if meta_json is None:
            raise ValueError("Must provide keep_blocks or meta_json")
        with open(meta_json) as f:
            meta = json.load(f)
        keep_blocks = meta["keep_blocks"]

    if meta_json is not None and meta is None:
        with open(meta_json) as f:
            meta = json.load(f)

    if meta is not None and "adaptive_keep_dim_per_block" in meta:
        per_block_hidden_dims = {
            int(k): int(v) for k, v in meta["adaptive_keep_dim_per_block"].items()
        }

    # Optional attention-head pruning info (from prune_attn_heads_adaptive.py)
    head_prune = meta.get("attn_head_pruning") if meta is not None else None

    keep_blocks = sorted(set(keep_blocks))
    print(f"Building slim SAM3 with {len(keep_blocks)} encoder blocks: {keep_blocks}")
    if mlp_hidden_dim is not None:
        print(f"  MLP hidden_dim override: {mlp_hidden_dim} (default 4736)")
    if per_block_hidden_dims is not None:
        print("  Adaptive per-block MLP dims detected in meta JSON")

    # Build architecture
    vit = _create_slim_vit(keep_blocks, mlp_hidden_dim=mlp_hidden_dim)
    if per_block_hidden_dims is not None:
        _apply_per_block_mlp_dims(vit, keep_blocks, per_block_hidden_dims)
    if head_prune is not None:
        kept_heads = {int(k): v for k, v in head_prune["kept_heads_per_block"].items()}
        print(f"  Attention head pruning detected (avg {head_prune.get('avg_kept_heads'):.2f}"
              f"/{head_prune['orig_num_heads']} heads)")
        _apply_per_block_head_dims(
            vit, keep_blocks, kept_heads,
            orig_num_heads=head_prune["orig_num_heads"],
            head_dim=head_prune["head_dim"],
        )
    pos_enc = _create_position_encoding(precompute_resolution=1008, device=device)
    neck = _create_vit_neck(pos_enc, vit)

    # Vision-language backbone — text encoder is None for prompt-free / bbox usage
    # If you need text prompts, construct a text encoder separately and pass it here.
    vl_backbone = SAM3VLBackbone(visual=neck, text=None, scalp=1)

    transformer = TransformerWrapper(
        encoder=_create_transformer_encoder(),
        decoder=_create_transformer_decoder(),
        d_model=256,
    )
    geometry_encoder = _create_geometry_encoder(device=device)
    seg_head = _create_segmentation_head()
    dot_prod = _create_dot_product_scoring()

    model = Sam3Image(
        backbone=vl_backbone,
        transformer=transformer,
        input_geometry_encoder=geometry_encoder,
        segmentation_head=seg_head,
        num_feature_levels=1,
        o2m_mask_predict=True,
        dot_prod_scoring=dot_prod,
        use_instance_query=False,
        multimask_output=True,
        inst_interactive_predictor=None,
        matcher=None,
    )

    # Load slim state dict
    ckpt_path = Path(slim_ckpt).expanduser().resolve()
    print(f"Loading slim checkpoint from {ckpt_path} ...")
    raw_ckpt = torch.load(str(ckpt_path), map_location="cpu")

    # Strip detector. prefix → model keys
    model_ckpt = {
        k.replace("detector.", ""): v
        for k, v in raw_ckpt.items()
        if k.startswith("detector.")
    }

    missing, unexpected = model.load_state_dict(model_ckpt, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    print(f"  Loaded {len(model_ckpt) - len(unexpected)} / {len(model_ckpt)} keys successfully")

    if eval_mode:
        model.eval()
    if device == "cuda":
        model = model.cuda()

    processor = Sam3Processor(model=model, device=device)
    print("Slim SAM3 ready.")
    return processor


# ---------------------------------------------------------------------------
# CLI validation
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Load and validate slim SAM3 model")
    p.add_argument("--slim-ckpt", required=True)
    p.add_argument("--meta-json", default=None)
    p.add_argument("--keep-blocks", nargs="+", type=int, default=None)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    processor = build_slim_sam3_image_model(
        slim_ckpt=args.slim_ckpt,
        keep_blocks=args.keep_blocks,
        meta_json=args.meta_json,
        device=args.device,
    )
    model = processor.model

    total = sum(p.numel() for p in model.parameters()) / 1e6
    trunk_total = sum(p.numel() for p in model.backbone.vision_backbone.trunk.parameters()) / 1e6
    print(f"\nModel total params:  {total:.1f}M")
    print(f"Trunk params:        {trunk_total:.1f}M")
    print(f"Trunk depth:         {len(model.backbone.vision_backbone.trunk.blocks)} blocks")

    # Smoke-test forward pass with small input to validate shape wiring
    print("\nRunning smoke-test forward (vision backbone only, 224x224) ...")
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=args.device)
        out = model.backbone.vision_backbone(dummy)
    feats = out["vision_features"]
    print(f"  vision_features shape: {feats.shape}  ✓")


if __name__ == "__main__":
    main()
