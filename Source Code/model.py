# ==================================================================================================
# ZERO-SHOT-VIDEO-GENERATION - model.py (Neural Orchestration & Architecture)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This script constitutes the core machine learning inference orchestration class (`Model`). It 
# abstracts the underlying PyTorch and Diffusers infrastructure, enabling seamless initialization, 
# dynamic loading, and execution of various latent diffusion models (e.g., standard Text2Video, 
# ControlNet-enhanced derivations). Crucially, this script also introduces logical chunking to 
# mitigate Out-Of-Memory (OOM) errors during temporal inference streams common on consumer-grade hardware.
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

from enum import Enum
import gc
import os
import numpy as np
import tomesd
import torch

from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionControlNetPipeline, ControlNetModel, UNet2DConditionModel
from diffusers.schedulers import EulerAncestralDiscreteScheduler, DDIMScheduler
from text_to_video_pipeline import TextToVideoPipeline

import utils
import gradio_utils


class ModelType(Enum):
    """Enumeration identifying target diffusion frameworks supported by the Model abstractor."""
    Pix2Pix_Video = 1,
    Text2Video = 2,
    ControlNetCanny = 3,
    ControlNetCannyDB = 4,
    ControlNetPose = 5,
    ControlNetDepth = 6,


class Model:
    """
    Primary interface for managing diffusion pipeline lifecycles, execution states, and tensor 
    operations bridging natural language prompts directly to multi-frame video matrices.
    """
    def __init__(self, device, dtype, **kwargs):
        self.device = device
        self.dtype = dtype
        self.generator = torch.Generator(device=device)
        
        # Pipeline mapping dictionary to seamlessly pivot between generation contexts.
        self.pipe_dict = {
            ModelType.Pix2Pix_Video: StableDiffusionInstructPix2PixPipeline,
            ModelType.Text2Video: TextToVideoPipeline,
            ModelType.ControlNetCanny: StableDiffusionControlNetPipeline,
            ModelType.ControlNetCannyDB: StableDiffusionControlNetPipeline,
            ModelType.ControlNetPose: StableDiffusionControlNetPipeline,
            ModelType.ControlNetDepth: StableDiffusionControlNetPipeline,
        }
        
        # Instantiation of custom Cross-Frame Attention modules to globally preserve semantic consistency 
        # across the latent dimension timeline.
        self.controlnet_attn_proc = utils.CrossFrameAttnProcessor(
            unet_chunk_size=2)
        self.pix2pix_attn_proc = utils.CrossFrameAttnProcessor(
            unet_chunk_size=3)
        self.text2video_attn_proc = utils.CrossFrameAttnProcessor(
            unet_chunk_size=2)

        self.pipe = None
        self.model_type = None

        self.states = {}
        self.model_name = ""

    def set_model(self, model_type: ModelType, model_id: str, **kwargs):
        """
        Dynamically initializes the selected neural structural model. Incorporates hardware cleanup 
        protocols to proactively mitigate CUDA fragmentations. It also supports local fallback parsing 
        for environments operating offline without persistent API gateways.
        """
        if hasattr(self, "pipe") and self.pipe is not None:
            del self.pipe
            self.pipe = None
        torch.cuda.empty_cache()
        gc.collect()

        # Offline/Local parsing capability targeting critical auxiliary features.
        models_dir = os.path.join(os.getcwd(), "models")
        local_safety_path = os.path.join(models_dir, "stable-diffusion-safety-checker")
        local_clip_path = os.path.join(models_dir, "clip-vit-large-patch14")

        if os.path.exists(local_safety_path) and 'safety_checker' not in kwargs:
             print(f"Loading local safety checker from {local_safety_path}")
             from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
             kwargs['safety_checker'] = StableDiffusionSafetyChecker.from_pretrained(local_safety_path).to(self.device).to(self.dtype)
        
        if os.path.exists(local_clip_path) and 'feature_extractor' not in kwargs:
             print(f"Loading local feature extractor from {local_clip_path}")
             from transformers import CLIPImageProcessor
             kwargs['feature_extractor'] = CLIPImageProcessor.from_pretrained(local_clip_path)

        # Download or load defined weights directly from disk into designated precision logic.
        self.pipe = self.pipe_dict[model_type].from_pretrained(
            model_id, **kwargs).to(self.device).to(self.dtype)
        self.model_type = model_type
        self.model_name = model_id

    def inference_chunk(self, frame_ids, **kwargs):
        """
        Executes diffusion step sequences exclusively for a segmented sub-tensor of the video frames.
        Essential structure for enabling high-resolution processing avoiding continuous VRAM spikes.
        """
        if not hasattr(self, "pipe") or self.pipe is None:
            return

        prompt = kwargs.pop('prompt', '')
        if prompt is None:
            prompt = ''
        if isinstance(prompt, str):
            prompt = [prompt] * kwargs.get('video_length', len(frame_ids))
        prompt = np.array(prompt)

        negative_prompt = kwargs.pop('negative_prompt', '')
        if negative_prompt is None:
            negative_prompt = ''
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * kwargs.get('video_length', len(frame_ids))
        negative_prompt = np.array(negative_prompt)
        latents = None
        if 'latents' in kwargs:
            latents = kwargs.pop('latents')[frame_ids]
        if 'image' in kwargs:
            kwargs['image'] = kwargs['image'][frame_ids]
        if 'video_length' in kwargs:
            kwargs['video_length'] = len(frame_ids)
        if self.model_type == ModelType.Text2Video:
            kwargs["frame_ids"] = frame_ids
            
        # Dispatch bounded operations to the active Denoising Diffusion pipeline.
        return self.pipe(prompt=prompt[frame_ids].tolist(),
                         negative_prompt=negative_prompt[frame_ids].tolist(),
                         latents=latents,
                         generator=self.generator,
                         **kwargs)

    def inference(self, **kwargs):
        """
        Evaluates execution constraints to govern memory orchestration dynamically. Either triggers standard 
        contiguous processing or coordinates sequential chunking of the latent frames.
        """
        if not hasattr(self, "pipe") or self.pipe is None:
            return

        split_to_chunks = kwargs.pop('split_to_chunks', False)
        chunk_size = kwargs.pop('chunk_size', 8)
        video_length = kwargs.get('video_length', 8)

        if split_to_chunks:
            # Iterative logic parsing discrete blocks into computational pipeline, later reassembled.
            import math
            num_chunks = math.ceil(video_length / chunk_size)
            all_frames = []
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, video_length)
                frame_ids = list(range(start_idx, end_idx))
                chunk_result = self.inference_chunk(frame_ids, **kwargs)
                all_frames.append(chunk_result.images)
            
            # Post-generation frame integration logic rebuilding structural tensor outputs.
            import numpy as np
            if isinstance(all_frames[0], np.ndarray):
                combined = np.concatenate(all_frames, axis=0)
            else:
                combined = [img for chunk in all_frames for img in chunk]
            return combined
        else:
            return self.pipe(**kwargs).images


    def process_text2video(self,
                           prompt,
                           model_name="dreamlike-art/dreamlike-photoreal-2.0",
                           motion_field_strength_x=12,
                           motion_field_strength_y=12,
                           t0=44,
                           t1=47,
                           n_prompt="",
                           chunk_size=8,
                           video_length=8,
                           watermark='',
                           merging_ratio=0.0,
                           seed=0,
                           resolution=512,
                           fps=2,
                           use_cf_attn=True,
                           use_motion_field=True,
                           smooth_bg=False,
                           smooth_bg_strength=0.4,
                           path=None):
        """
        Definitive API execution method initializing the complete zero-shot process lifecycle. Evaluates models, 
        injects positive/negative prompt constraints, forces reproducibility utilizing PRNG seeding, and leverages 
        structural enhancements inclusive of CF-Attn and logical motion-field mapping.
        """
        print("Module Text2Video")
        if self.model_type != ModelType.Text2Video or model_name != self.model_name:
            print(f"Model update to {model_name}")
            
            # Context-aware load behavior evaluating disk storage against online fetching logic.
            local_model_path = os.path.join(os.getcwd(), "models", model_name.split('/')[-1])
            load_path = local_model_path if os.path.exists(local_model_path) else model_name
            
            if os.path.exists(local_model_path):
                print(f"Using local model weights from {local_model_path}")
            
            unet = UNet2DConditionModel.from_pretrained(
                load_path, subfolder="unet")
            self.set_model(ModelType.Text2Video,
                           model_id=load_path, unet=unet)
            self.model_name = model_name # Keep the original name for state tracking
            
            # Applying fixed Denoising Diffusion Implicit Model parameters ensuring generation alignment.
            self.pipe.scheduler = DDIMScheduler.from_config(
                self.pipe.scheduler.config)
            if use_cf_attn:
                self.pipe.unet.set_attn_processor(
                    processor=self.text2video_attn_proc)
        self.generator.manual_seed(seed)

        # Forced architectural enhancements promoting output quality irrespective of user structural definition.
        added_prompt = "high quality, HD, 8K, trending on artstation, high focus, dramatic lighting"
        negative_prompts = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer difits, cropped, worst quality, low quality, deformed body, bloated, ugly, unrealistic'

        # Textual conditioning serialization to ensure predictable tokenized vectors.
        prompt = prompt.rstrip()
        if len(prompt) > 0 and (prompt[-1] == "," or prompt[-1] == "."):
            prompt = prompt.rstrip()[:-1]
        prompt = prompt.rstrip()
        prompt = prompt + ", "+added_prompt
        if len(n_prompt) > 0:
            negative_prompt = n_prompt
        else:
            negative_prompt = None

        # Call underlying structure logic enforcing custom bounds, generating target frames sequentially.
        result = self.inference(prompt=prompt,
                                video_length=video_length,
                                height=resolution,
                                width=resolution,
                                num_inference_steps=50,
                                guidance_scale=7.5,
                                guidance_stop_step=1.0,
                                t0=t0,
                                t1=t1,
                                motion_field_strength_x=motion_field_strength_x,
                                motion_field_strength_y=motion_field_strength_y,
                                use_motion_field=use_motion_field,
                                smooth_bg=smooth_bg,
                                smooth_bg_strength=smooth_bg_strength,
                                seed=seed,
                                output_type='numpy',
                                negative_prompt=negative_prompt,
                                merging_ratio=merging_ratio,
                                split_to_chunks=True,
                                chunk_size=chunk_size,
                                )
        return utils.create_video(result, fps, path=path, watermark=gradio_utils.logo_name_to_path(watermark))