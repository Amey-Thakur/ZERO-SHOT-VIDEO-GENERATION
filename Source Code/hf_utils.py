# ==================================================================================================
# ZERO-SHOT-VIDEO-GENERATION - hf_utils.py (Hugging Face Integrations)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module encapsulates the fetching and parsing mechanisms for resolving dynamic model check-
# points directly from the Hugging Face Hub ecosystem. By dynamically sourcing the latest public 
# model definitions (specifically Dreambooth derivatives), the interface guarantees up-to-date 
# experimentation bounds without relying strictly on static programmatic hardcoding.
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

from bs4 import BeautifulSoup
import requests


def model_url_list():
    """
    Constructs the target API evaluation sequences to query the Hugging Face Hub for optimal 
    structural matches (specifically targeting 'dreambooth' models sorted by highest download count).
    """
    url_list = []
    # Queries the first pagination structure to avoid bandwidth over-saturation while 
    # capturing statistically relevant diffusion models.
    for i in range(0, 1):
        url_list.append(
            f"https://huggingface.co/models?p={i}&sort=downloads&search=dreambooth")
    return url_list


def data_scraping(url_list):
    """
    Executes a web-scraping protocol against the constructed Hub endpoints. It parses the 
    HTML Document Object Model (DOM) to strictly extract the unique stylistic model identifiers.
    """
    model_list = []
    for url in url_list:
        try:
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.text, "html.parser")
            # Target the specific tailwind-defined grid structure hosting the model cards.
            div_class = 'grid grid-cols-1 gap-5 2xl:grid-cols-2'
            div = soup.find('div', {'class': div_class})
            # Extract associative link coordinates (hrefs) representing the model paths.
            if div:
                for a in div.find_all('a', href=True):
                    model_list.append(a['href'])
        except Exception as e:
            print(f"Request to {url} failed: {e}")
    return model_list


def get_model_list():
    """
    Assembles the definitive list of diffusion model pipelines available for inference.
    Fuses statically defined 'guaranteed' baseline models (e.g., Dreamlike Photoreal) with 
    the dynamically acquired structural checkpoints scraped from the Hub. Incorporates robust 
    offline fallback protocols ensuring interface continuity when API connectivity fails.
    """
    # Baseline stable diffusion model path
    best_model_list = [
        "dreamlike-art/dreamlike-photoreal-2.0",
    ]
    try:
        model_list = data_scraping(model_url_list())
        # Trim leading slashes establishing standardized Hugging Face API notation (User/Model).
        for i in range(len(model_list)):
            model_list[i] = model_list[i][1:]
        model_list = best_model_list + model_list
    except Exception as e:
        # Graceful degradation into offline operation modality.
        print(f"Warning: Could not fetch model list from HuggingFace (offline mode?). Using default model only. Error: {e}")
        model_list = best_model_list
        
    return model_list
