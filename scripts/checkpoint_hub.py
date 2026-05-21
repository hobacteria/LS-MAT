"""Ensure pretrained weights exist by downloading from Hugging Face when missing."""

from __future__ import annotations

import glob
import os
import shutil

HF_REPO_DEFAULT = "Hovirus/LS-MAT"
REQUIRED_FILES = ("autoencoder.pt", "diff_unet_ckpt.pt", "controlnet_kits_finetune.pt")


def _checkpoint_dir_from_args(args) -> str:
    return os.path.dirname(os.path.abspath(args.trained_vae_gan_path))


def _needs_download(checkpoint_dir: str) -> bool:
    if not os.path.isdir(checkpoint_dir):
        return True
    if len(os.listdir(checkpoint_dir)) == 0:
        return True
    for name in REQUIRED_FILES:
        path = os.path.join(checkpoint_dir, name)
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            return True
    return False


def _promote_nested_weights(checkpoint_dir: str) -> None:
    """If snapshot placed .pt files under a subfolder, copy them to checkpoint_dir."""
    for name in REQUIRED_FILES:
        dest = os.path.join(checkpoint_dir, name)
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            continue
        matches = [
            m
            for m in glob.glob(os.path.join(checkpoint_dir, "**", name), recursive=True)
            if os.path.isfile(m)
        ]
        if matches:
            shutil.copy2(matches[0], dest)


def ensure_model_checkpoints(args, repo_id: str = HF_REPO_DEFAULT) -> None:
    """
    If ``model_checkpoint`` (the directory of ``trained_vae_gan_path``) is empty or
    any required weight is missing, download the repository snapshot from Hugging Face.
    """
    checkpoint_dir = _checkpoint_dir_from_args(args)
    if not _needs_download(checkpoint_dir):
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Pretrained checkpoints are missing. Install huggingface_hub "
            "(see requirements.txt) or download weights manually from "
            "https://huggingface.co/Hovirus/LS-MAT"
        ) from exc

    os.makedirs(checkpoint_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=checkpoint_dir,
        local_dir_use_symlinks=False,
    )
    _promote_nested_weights(checkpoint_dir)

    missing = [
        name
        for name in REQUIRED_FILES
        if not os.path.isfile(os.path.join(checkpoint_dir, name))
        or os.path.getsize(os.path.join(checkpoint_dir, name)) == 0
    ]
    if missing:
        raise FileNotFoundError(
            f"After downloading {repo_id}, these files are still missing or empty: {missing}. "
            f"See https://huggingface.co/{repo_id}"
        )
