# ==================================================================================================
# ZERO-SHOT-VIDEO-GENERATION - app_text_to_video.py (Gradio UI Components)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module constructs the structural interface for the Text2Video-Zero generation task. It 
# defines the modular Gradio UI components, formulates the layout parameters, and specifies the 
# data bindings between visual controls (like sliders, dropdowns, and buttons) and the underlying 
# neural processing model. Designed for modularity, it manages state interactions specifically for 
# translating textual representations into dynamic video sequences.
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
from model import Model
import os
from hf_utils import get_model_list

# Determine the operational execution context. Hugging Face Spaces deployments may impose
# unique limitations on specific resource-intensive tasks, thereby requiring architectural adaptations.
on_huggingspace = os.environ.get("SPACE_AUTHOR_NAME") == "PAIR"

# Predefined contextual exemplars establishing baseline structural validation inputs.
# These prompts generate optimal temporal consistency in generated outputs utilizing the
# latent diffusion methodology.
examples = [
    ["an astronaut waving the arm on the moon"],
    ["a sloth surfing on a wakeboard"],
    ["an astronaut walking on a street"],
    ["a cute cat walking on grass"],
    ["a horse is galloping on a street"],
    ["an astronaut is skiing down the hill"],
    ["a gorilla walking alone down the street"],
    ["a gorilla dancing on times square"],
    ["A panda dancing dancing like crazy on Times Square"],
]


def create_demo(model: Model):
    """
    Constructs and returns the interactive elements of the Gradio interface for textual inputs.
    Binds the local inference 'model' context to user-facing input handlers to coordinate state 
    between the UI framework and the PyTorch execution context.
    """
    # Instantiate the declarative layout constructor.
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown('## Text2Video-Zero: Video Generation')
        with gr.Row():
            gr.HTML(
                """
                <div style="text-align: left; auto;">
                <h2 style="font-weight: 450; font-size: 1rem; margin: 0rem">
                    Description: Instantly create videos using a text prompt or our sample examples.
                </h3>
                </div>
                """)

        with gr.Row():
            with gr.Column():
                # Configuration block controlling diffusion model weights and textual targets.
                model_name = gr.Dropdown(
                    label="Model",
                    choices=get_model_list(),
                    value="dreamlike-art/dreamlike-photoreal-2.0",

                )
                prompt = gr.Textbox(label='Prompt')
                run_button = gr.Button(value='Run')
                
                # Expandable execution variables defining trajectory lengths (temporal depth).
                with gr.Accordion('Advanced options', open=False):
                    
                    # Adapting video constraints algorithmically based on the execution domain constraints.
                    if on_huggingspace:
                        video_length = gr.Slider(
                            label="Video length", minimum=8, maximum=16, step=1)
                    else:
                        video_length = gr.Number(
                            label="Video length", value=8, precision=0)

            with gr.Column():
                # Instantiation of the rendering element to visualize synthesized structures.
                result = gr.Video(label="Generated Video")

        inputs = [
            prompt,
            model_name,
          
            video_length,
           
        ]

        # Bind curated input permutations to expedite visualization pathways.
        gr.Examples(examples=examples,
                    inputs=inputs,
                    outputs=result,
                    fn=model.process_text2video,
                    run_on_click=False,
                    cache_examples=on_huggingspace,
                    )

        # Trigger execution of the generative framework upon interactive activation.
        run_button.click(fn=model.process_text2video,
                         inputs=inputs,
                         outputs=result,)
    return demo
