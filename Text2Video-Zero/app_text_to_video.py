import gradio as gr
from model import Model
import os
from hf_utils import get_model_list

on_huggingspace = os.environ.get("SPACE_AUTHOR_NAME") == "PAIR"

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
                model_name = gr.Dropdown(
                    label="Model",
                    
                    value="dreamlike-art/dreamlike-photoreal-2.0",

                )
                prompt = gr.Textbox(label='Prompt')
                run_button = gr.Button(label='Run')
                with gr.Accordion('Advanced options', open=False):
                    

                    if on_huggingspace:
                        video_length = gr.Slider(
                            label="Video length", minimum=8, maximum=16, step=1)
                    else:
                        video_length = gr.Number(
                            label="Video length", value=8, precision=0)

            with gr.Column():
                result = gr.Video(label="Generated Video")

        inputs = [
            prompt,
            model_name,
          
            video_length,
           
        ]

        gr.Examples(examples=examples,
                    inputs=inputs,
                    outputs=result,
                    fn=model.process_text2video,
                    run_on_click=False,
                    cache_examples=on_huggingspace,
                    )

        run_button.click(fn=model.process_text2video,
                         inputs=inputs,
                         outputs=result,)
    return demo
