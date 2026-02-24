import gradio as gr
import torch

from model import Model, ModelType
from app_text_to_video import create_demo as create_demo_text_to_video
import argparse
import os

on_huggingspace = os.environ.get("SPACE_AUTHOR_NAME") == "PAIR"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Model(device=device, dtype=torch.float16 if device == "cuda" else torch.float32)

parser = argparse.ArgumentParser()
parser.add_argument('--public_access', action='store_true',
                    help="if enabled, the app can be access from a public url", default=False)
args = parser.parse_args()


with gr.Blocks() as demo:

    gr.HTML(
        """
        <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
        <h1 style="font-weight: 900; font-size: 3rem; margin: 0rem">
            <a href="https://github.com/Amey-Thakur/ZERO-SHOT-VIDEO-GENERATION" style="color:blue;">Zero-Shot Video Generation</a> 
        </h1>
        <h3>Original research and development of <a href="https://arxiv.org/abs/2303.13439">Text2Video-Zero</a> was conducted by the team at Picsart AI Research (PAIR), UT Austin, U of Oregon, and UIUC.
        </h3>

        </div>
        """)

    with gr.Tab('Zero-Shot Text2Video'):
        create_demo_text_to_video(model)
        


if on_huggingspace:
    demo.queue().launch(debug=True)
else:
    _, _, link = demo.queue().launch(
        allowed_paths=['temporal'], 
        share=args.public_access,
        css='style.css')
    print(link)