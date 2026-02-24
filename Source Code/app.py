# ==================================================================================================
# ZERO-SHOT-VIDEO-GENERATION - app.py (Primary Application Interface)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This script serves as the main entry point and Gradio-based web interface for the Zero-Shot 
# Video Generation framework. It provisions the required neural network models and exposes a 
# user-friendly front-end for generating temporally consistent video content from textual prompts. 
# The interface is robustly abstracted to handle execution seamlessly across various environments, 
# inclusive of local execution and cloud instances.
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

import gradio as gr
import torch

from model import Model, ModelType
from app_text_to_video import create_demo as create_demo_text_to_video
import argparse
import os

# --- ENVIRONMENT & HARDWARE INITIALIZATION ---
# Identify the operational environment to conditionally adapt interface parameters, e.g., enabling
# caching specifically for remote cloud deployments. Hardware availability evaluates the
# presence of CUDA computational nodes for accelerated precision computations or gracefully 
# degrades to CPU-based inference.
on_huggingspace = os.environ.get("SPACE_AUTHOR_NAME") == "PAIR"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate the primary generative diffusion model employing Float16 on GPU resources 
# for memory-efficient tensor operations, and Float32 as a robust computational fallback.
model = Model(device=device, dtype=torch.float16 if device == "cuda" else torch.float32)

# --- CLI ARGUMENTS PARSING ---
# Establishes public accessibility parameters, useful when tunneling standard localhost traffic 
# securely for temporary external evaluations over the internet.
parser = argparse.ArgumentParser()
parser.add_argument('--public_access', action='store_true',
                    help="if enabled, the app can be access from a public url", default=False)
args = parser.parse_args()

# --- WEB INTERFACE ARCHITECTURE ---
# Assembles the Gradio Application Block layout, injecting structured HTML context and 
# encapsulating the discrete video synthesis module instance utilizing the neural pipeline.
with gr.Blocks() as demo:

    gr.HTML(
        """
        <div style="background: linear-gradient(135deg, #4A00E0 0%, #8E2DE2 100%); padding: 3rem; border-radius: 20px; text-align: center; margin-bottom: 2rem; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
            <h1 style="color: white; font-size: 3.5rem; font-weight: 800; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); letter-spacing: -1px;">
                🎥 Zero-Shot Video Generation
            </h1>
            <p style="color: rgba(255,255,255,0.9); font-size: 1.2rem; margin-top: 1rem; font-weight: 500;">
                Original research by Picsart AI Research (PAIR), UT Austin, U of Oregon, and UIUC.
            </p>
        </div>
        """
    )

    with gr.Tab('Zero-Shot Text2Video'):
        # Invoke the pre-defined layout specific to the Text-to-Video generative logic, passing 
        # the initialized main diffusion model capable of handling the temporal latent inference.
        create_demo_text_to_video(model)
        

# --- APPLICATION DEPLOYMENT ALGORITHM ---
# Deploys the constructed graphical interface. Configures queuing mechanisms intrinsically to 
# prevent execution thread over-saturation during concurrent generation requests.
if on_huggingspace:
    demo.queue().launch(
        debug=True,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")
    )
else:
    _, _, link = demo.queue().launch(
        allowed_paths=['temporal'], 
        share=args.public_access,
        css='style.css',
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")
    )
    print(link)