import os
from huggingface_hub import snapshot_download

def download_models():
    base_path = os.path.join(os.getcwd(), "models")
    os.makedirs(base_path, exist_ok=True)

    # 1. Dreamlike Photoreal 2.0
    print("Downloading Dreamlike Photoreal 2.0...")
    dreamlike_path = os.path.join(base_path, "dreamlike-photoreal-2.0")
    snapshot_download(
        repo_id="dreamlike-art/dreamlike-photoreal-2.0",
        local_dir=dreamlike_path,
        local_dir_use_symlinks=False
    )
    print(f"Successfully downloaded to {dreamlike_path}")

    # 2. CLIP Model (openai/clip-vit-large-patch14)
    print("Downloading CLIP Model...")
    clip_path = os.path.join(base_path, "clip-vit-large-patch14")
    snapshot_download(
        repo_id="openai/clip-vit-large-patch14",
        local_dir=clip_path,
        local_dir_use_symlinks=False
    )
    print(f"Successfully downloaded to {clip_path}")

    # 3. Safety Checker (CompVis/stable-diffusion-safety-checker)
    print("Downloading Safety Checker...")
    safety_path = os.path.join(base_path, "stable-diffusion-safety-checker")
    snapshot_download(
        repo_id="CompVis/stable-diffusion-safety-checker",
        local_dir=safety_path,
        local_dir_use_symlinks=False
    )
    print(f"Successfully downloaded to {safety_path}")

if __name__ == "__main__":
    download_models()
