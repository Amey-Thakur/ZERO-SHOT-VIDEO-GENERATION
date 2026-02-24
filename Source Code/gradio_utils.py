# ==================================================================================================
# ZERO-SHOT-VIDEO-GENERATION - gradio_utils.py (Interface Utilities)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This utility module provides essential helper functions for the Gradio web interface. It acts 
# as an intermediary data transformation layer, managing the resolution of internal asset paths, 
# interpreting user interactions across various deployment modalities (e.g., Canny edge detection, 
# Pose estimation, Dreambooth fine-tuning), and structurally validating input/output pathways 
# ensuring consistency during the text-to-video associative processing sequences.
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
# � RELEASE DATE
# November 22, 2023
#
# �📜 LICENSE
# Released under the MIT License
# ==================================================================================================

import os

# --- CONTROLNET: CANNY EDGE UTILITIES ---
# These functions map symbolic interface selections (like predefined edge maps) to their 
# corresponding physical file paths within the asset directory, ensuring strict structural validation.

def edge_path_to_video_path(edge_path):
    """
    Translates a provided qualitative description or partial path of an edge map to a fully 
    qualified internal asset registry path used during video processing.
    """
    video_path = edge_path

    vid_name = edge_path.split("/")[-1]
    if vid_name == "butterfly.mp4":
        video_path = "__assets__/canny_videos_mp4/butterfly.mp4"
    elif vid_name == "deer.mp4":
        video_path = "__assets__/canny_videos_mp4/deer.mp4"
    elif vid_name == "fox.mp4":
        video_path = "__assets__/canny_videos_mp4/fox.mp4"
    elif vid_name == "girl_dancing.mp4":
        video_path = "__assets__/canny_videos_mp4/girl_dancing.mp4"
    elif vid_name == "girl_turning.mp4":
        video_path = "__assets__/canny_videos_mp4/girl_turning.mp4"
    elif vid_name == "halloween.mp4":
        video_path = "__assets__/canny_videos_mp4/halloween.mp4"
    elif vid_name == "santa.mp4":
        video_path = "__assets__/canny_videos_mp4/santa.mp4"

    # Strict validation ensures subsequent neural tensor loading operations do not encounter IOErrors.
    assert os.path.isfile(video_path)
    return video_path


# --- CONTROLNET: POSE ESTIMATION UTILITIES ---
def motion_to_video_path(motion):
    """
    Translates textual motion directives (e.g., 'Dance 1') into mapped physical skeleton GIF 
    assets utilized for conditioning the temporal generation in Pose methodologies.
    """
    videos = [
        "__assets__/poses_skeleton_gifs/dance1_corr.mp4",
        "__assets__/poses_skeleton_gifs/dance2_corr.mp4",
        "__assets__/poses_skeleton_gifs/dance3_corr.mp4",
        "__assets__/poses_skeleton_gifs/dance4_corr.mp4",
        "__assets__/poses_skeleton_gifs/dance5_corr.mp4"
    ]
    if len(motion.split(" ")) > 1 and motion.split(" ")[1].isnumeric():
        id = int(motion.split(" ")[1]) - 1
        return videos[id]
    else:
        return motion


# --- DREAMBOOTH: ZERO-SHOT INCORPORATION UTILITIES ---
def get_video_from_canny_selection(canny_selection):
    """
    Resolves base video sequences specifically tailored for fine-tuned Dreambooth inference.
    """
    if canny_selection == "woman1":
        input_video_path = "__assets__/db_files_2fps/woman1.mp4"

    elif canny_selection == "woman2":
        input_video_path = "__assets__/db_files_2fps/woman2.mp4"

    elif canny_selection == "man1":
        input_video_path = "__assets__/db_files_2fps/man1.mp4"

    elif canny_selection == "woman3":
        input_video_path = "__assets__/db_files_2fps/woman3.mp4"
    else:
        input_video_path = canny_selection

    assert os.path.isfile(input_video_path)
    return input_video_path


def get_model_from_db_selection(db_selection):
    """
    Translates user-friendly stylistic dropdown options into exact neural checkpoint identifiers 
    hosted on corresponding model hubs.
    """
    if db_selection == "Anime DB":
        input_video_path = 'PAIR/text2video-zero-controlnet-canny-anime'
    elif db_selection == "Avatar DB":
        input_video_path = 'PAIR/text2video-zero-controlnet-canny-avatar'
    elif db_selection == "GTA-5 DB":
        input_video_path = 'PAIR/text2video-zero-controlnet-canny-gta5'
    elif db_selection == "Arcane DB":
        input_video_path = 'PAIR/text2video-zero-controlnet-canny-arcane'
    else:
        input_video_path = db_selection

    return input_video_path


def get_db_name_from_id(id):
    """Auxiliary mapper for Dreambooth stylistic identifiers."""
    db_names = ["Anime DB", "Arcane DB", "GTA-5 DB", "Avatar DB"]
    return db_names[id]


def get_canny_name_from_id(id):
    """Auxiliary mapper for base semantic subjects."""
    canny_names = ["woman1", "woman2", "man1", "woman3"]
    return canny_names[id]


# --- WATERMARKING & ATTRIBUTION ---
def logo_name_to_path(name):
    """
    Interprets watermark selection for programmatic embedding into the terminal composite 
    video frames to enforce attribution.
    """
    logo_paths = {
        'Picsart AI Research': '__assets__/pair_watermark.png',
        'Text2Video-Zero': '__assets__/t2v-z_watermark.png',
        'None': None
    }
    if name in logo_paths:
        return logo_paths[name]
    return name

