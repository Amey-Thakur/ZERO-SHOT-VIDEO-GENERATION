# ==================================================================================================
# ZERO-SHOT-VIDEO-GENERATION - config.py (Architectural Configuration)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This configuration file defines the hyperparameter blueprints and architectural structure for 
# the auxiliary neural networks, specifically targeting the UniFormer backbone utilized in semantic 
# segmentation tasks (e.g., ADE20K dataset). It orchestrates the model's layers, dimensions, 
# and the optimization strategies required for precise background/foreground detection.
#
# 👤 AUTHORS
# - Amey Thakur (https://github.com/Amey-Thakur)
#
# 🤝🏻 CREDITS
# Based directly on the foundational logic of Text2Video-Zero.
# Source Authors: Picsart AI Research (PAIR), UT Austin, U of Oregon, UIUC
# Reference: https://arxiv.org/abs/2303.13439
#
# 🔗 PROJECT LINKS
# Repository: https://github.com/Amey-Thakur/ZERO-SHOT-VIDEO-GENERATION
# Live Demo: https://huggingface.co/spaces/ameythakur/Zero-Shot-Video-Generation
# Video Demo: https://youtu.be/za9hId6UPoY
#
# 📅 RELEASE DATE
# November 22, 2023
#
# 📜 LICENSE
# Released under the MIT License
# ==================================================================================================

# Base configurations anchoring the runtime environment, datasets, and scheduling parameters.
_base_ = [
    '../../configs/_base_/models/upernet_uniformer.py', 
    '../../configs/_base_/datasets/ade20k.py',
    '../../configs/_base_/default_runtime.py', 
    '../../configs/_base_/schedules/schedule_160k.py'
]

# Structural definition of the UniFormer backbone and its associated classification heads.
model = dict(
    backbone=dict(
        type='UniFormer',
        embed_dim=[64, 128, 320, 512],
        layers=[3, 4, 8, 3],
        head_dim=64,
        drop_path_rate=0.25,
        windows=False,
        hybrid=False
    ),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        num_classes=150
    ),
    auxiliary_head=dict(
        in_channels=320,
        num_classes=150
    ))

# Optimization strategy: AdamW optimizer implementing specific constraints like omitting 
# weight decay for position embeddings and layer normalization to maintain spatial integrity.
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

# Learning rate scheduler leveraging a polynomial decay policy to stabilize convergence 
# across extensive iterations.
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# Batch definitions establishing hardware-bound concurrency limits.
data=dict(samples_per_gpu=2)