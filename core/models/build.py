import logging
import torch
from .discriminator import *
from .swin_transformer import *
from .uper_head import *
from .fcn_head import *

def build_feature_extractor(cfg):
    backbone_size = (cfg.MODEL.WEIGHTS.split('/')[-1]).split('_')[1]
    print(backbone_size)
    if backbone_size == 'small':
        backbone = SwinTransformer(
            embed_dim=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.3,
            ape=False,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            use_checkpoint=False)
    else:
        backbone = SwinTransformer(
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.3,
            ape=False,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            use_checkpoint=False)
    return backbone

def build_classifier(cfg):
    norm_cfg = dict(type='SyncBN', requires_grad=True)
    backbone_size = (cfg.MODEL.WEIGHTS.split('/')[-1]).split('_')[1]
    if backbone_size == 'small':
        classifier = UPerHead(
            in_channels=[96, 192, 384, 768],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=cfg.MODEL.NUM_CLASSES,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_weight=1.0)
        aux = FCNHead(
            in_channels=384,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=cfg.MODEL.NUM_CLASSES,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_weight=0.4)
    else:
        classifier = UPerHead(
            in_channels=[128, 256, 512, 1024],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=cfg.MODEL.NUM_CLASSES,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_weight=1.0)
        aux = FCNHead(
            in_channels=512,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=cfg.MODEL.NUM_CLASSES,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_weight=0.4)
    return classifier, aux



def build_adversarial_discriminator_cls(cfg):
    backbone_size = (cfg.MODEL.WEIGHTS.split('/')[-1]).split('_')[1]
    if backbone_size == 'small':
        model_D = PixelDiscriminator(384, 384, num_classes=cfg.MODEL.NUM_CLASSES) 
    else:
        model_D = PixelDiscriminator(512, 512, num_classes=cfg.MODEL.NUM_CLASSES) 
    return model_D

def build_adversarial_discriminator_bin(cfg):
    backbone_size = (cfg.MODEL.WEIGHTS.split('/')[-1]).split('_')[1]
    if backbone_size == 'small':
        model_D = PixelDiscriminator2(384, 384, num_classes=1)
    else:
        model_D = PixelDiscriminator2(512, 512, num_classes=1)
    return model_D
