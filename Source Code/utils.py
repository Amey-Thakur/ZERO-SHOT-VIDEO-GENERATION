# ==================================================================================================
# ZERO-SHOT-VIDEO-GENERATION - utils.py (Processing & Attention Utilities)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This script contains fundamental auxiliary routines supporting matrix manipulation, spatial 
# transformations, temporal processing capabilities (like extracting discrete frames from video 
# sources), and custom attention mechanisms. In particular, it houses the CrossFrameAttnProcessor, 
# which is instrumental in enforcing the temporal coherence across generated sequence frames.
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

import os

import PIL.Image
import numpy as np
import torch
import torchvision
from torchvision.transforms import Resize, InterpolationMode
import imageio
from einops import rearrange
import cv2
from PIL import Image
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from annotator.openpose import OpenposeDetector
from annotator.midas import MidasDetector
import decord

apply_canny = CannyDetector()
apply_openpose = OpenposeDetector()
apply_midas = MidasDetector()


def pre_process_canny(input_video, low_threshold=100, high_threshold=200):
    """
    Applies Canny edge detection across a sequential batch of image frames. This algorithm 
    extracts high-frequency spatial gradients, representing the structural edges acting as 
    conditioning signals for the generation pipeline.
    """
    detected_maps = []
    for frame in input_video:
        img = rearrange(frame, 'c h w -> h w c').cpu().numpy().astype(np.uint8)
        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)
        detected_maps.append(detected_map[None])
    detected_maps = np.concatenate(detected_maps)
    control = torch.from_numpy(detected_maps.copy()).float() / 255.0
    return rearrange(control, 'f h w c -> f c h w')


def pre_process_depth(input_video, apply_depth_detect: bool = True):
    """
    Processes a frame batch utilizing the MiDaS network estimating relative perspective depth mapping.
    Yields robust 3D structural boundaries optimizing foreground/background generation isolation.
    """
    detected_maps = []
    for frame in input_video:
        img = rearrange(frame, 'c h w -> h w c').cpu().numpy().astype(np.uint8)
        img = HWC3(img)
        if apply_depth_detect:
            detected_map, _ = apply_midas(img)
        else:
            detected_map = img
        detected_map = HWC3(detected_map)
        H, W, C = img.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
        detected_maps.append(detected_map[None])
    detected_maps = np.concatenate(detected_maps)
    control = torch.from_numpy(detected_maps.copy()).float() / 255.0
    return rearrange(control, 'f h w c -> f c h w')


def pre_process_pose(input_video, apply_pose_detect: bool = True):
    """
    Leverages OpenPose structural skeletal estimation calculating limb mapping over sequential frames. 
    Ideal for dictating complex biomechanical motion rendering.
    """
    detected_maps = []
    for frame in input_video:
        img = rearrange(frame, 'c h w -> h w c').cpu().numpy().astype(np.uint8)
        img = HWC3(img)
        if apply_pose_detect:
            detected_map, _ = apply_openpose(img)
        else:
            detected_map = img
        detected_map = HWC3(detected_map)
        H, W, C = img.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
        detected_maps.append(detected_map[None])
    detected_maps = np.concatenate(detected_maps)
    control = torch.from_numpy(detected_maps.copy()).float() / 255.0
    return rearrange(control, 'f h w c -> f c h w')


def create_video(frames, fps, rescale=False, path=None, watermark=None):
    """
    Compiles distinct tensor arrays back into standard compressed video files utilizing MP4 encoding.
    Optionally overlays defined attribution watermarking maintaining visual logic bounds.
    """
    if path is None:
        dir = "temporal"
        os.makedirs(dir, exist_ok=True)
        path = os.path.join(dir, 'movie.mp4')

    outputs = []
    for i, x in enumerate(frames):
        x = torchvision.utils.make_grid(torch.Tensor(x), nrow=4)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)

        if watermark is not None:
            x = add_watermark(x, watermark)
        outputs.append(x)

    imageio.mimsave(path, outputs, fps=fps)
    return path

def create_gif(frames, fps, rescale=False, path=None, watermark=None):
    """Auxiliary logic encoding frames specifically into lossless loop GIF representations."""
    if path is None:
        dir = "temporal"
        os.makedirs(dir, exist_ok=True)
        path = os.path.join(dir, 'canny_db.gif')

    outputs = []
    for i, x in enumerate(frames):
        x = torchvision.utils.make_grid(torch.Tensor(x), nrow=4)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        if watermark is not None:
            x = add_watermark(x, watermark)
        outputs.append(x)

    imageio.mimsave(path, outputs, fps=fps)
    return path

def add_watermark(image, watermark_path):
    """
    Injects overlay logo bitmaps applying standard blending mathematics on the target matrices.
    """
    if watermark_path is None or not os.path.exists(watermark_path):
        return image
    
    watermark = Image.open(watermark_path).convert("RGBA")
    img = Image.fromarray(image).convert("RGBA")
    
    # Simple watermark placement (bottom right)
    img.paste(watermark, (img.width - watermark.width - 10, img.height - watermark.height - 10), watermark)
    return np.array(img.convert("RGB"))

def prepare_video(video_path:str, resolution:int, device, dtype, normalize=True, start_t:float=0, end_t:float=-1, output_fps:int=-1):
    """
    Executes raw video extraction reading target sequences and sampling specifically calculated framerates.
    Translates sequences directly into operational multi-dimensional PyTorch tensors.
    """
    vr = decord.VideoReader(video_path)
    initial_fps = vr.get_avg_fps()
    if output_fps == -1:
        output_fps = int(initial_fps)
    if end_t == -1:
        end_t = len(vr) / initial_fps
    else:
        end_t = min(len(vr) / initial_fps, end_t)
    assert 0 <= start_t < end_t
    assert output_fps > 0
    start_f_ind = int(start_t * initial_fps)
    end_f_ind = int(end_t * initial_fps)
    num_f = int((end_t - start_t) * output_fps)
    sample_idx = np.linspace(start_f_ind, end_f_ind, num_f, endpoint=False).astype(int)
    video = vr.get_batch(sample_idx)
    if torch.is_tensor(video):
        video = video.detach().cpu().numpy()
    else:
        video = video.asnumpy()
    _, h, w, _ = video.shape
    video = rearrange(video, "f h w c -> f c h w")
    video = torch.Tensor(video).to(device).to(dtype)

    # Use max if you want the larger side to be equal to resolution (e.g. 512)
    k = float(resolution) / max(h, w)
    h *= k
    w *= k
    h = int(np.round(h / 64.0)) * 64
    w = int(np.round(w / 64.0)) * 64

    video = Resize((h, w), interpolation=InterpolationMode.BILINEAR, antialias=True)(video)
    if normalize:
        video = video / 127.5 - 1.0
    return video, output_fps


def post_process_gif(list_of_results, image_resolution):
    """Convenience wrapper mapping output streams targeting fixed structural path encoding."""
    output_file = "/tmp/ddxk.gif"
    imageio.mimsave(output_file, list_of_results, fps=4)
    return output_file


class CrossFrameAttnProcessor:
    """
    Fundamental Neural Network hook modifying the default UNet implementation. Rewrites the internal 
    attention lookup dictating that independent latent patches correlate queries against the persistent 
    Keys and Values established strictly by the initiating first temporal frame, resolving sequence drifting.
    """
    def __init__(self, unet_chunk_size=2):
        self.unet_chunk_size = unet_chunk_size

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            *args,
            **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif getattr(attn, 'norm_cross', None) is not None:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        # Sparse Attention enforcement mapping representations matching global zero definitions.
        if not is_cross_attention:
            video_length = key.size()[0] // self.unet_chunk_size
            former_frame_index = [0] * video_length
            key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
            key = key[:, former_frame_index]
            key = rearrange(key, "b f d c -> (b f) d c")
            value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
            value = value[:, former_frame_index]
            value = rearrange(value, "b f d c -> (b f) d c")

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states