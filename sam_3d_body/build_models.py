# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import torch

from .models.meta_arch import SAM3DBody
from .utils.config import get_config
from .utils.checkpoint import load_state_dict


def load_sam_3d_body(checkpoint_path: str = "", device: str = "cuda", mhr_path: str = ""):
    print("Loading SAM 3D Body model...")
    
    # Check the current directory, and if not present check the parent dir.
    model_cfg = os.path.join(os.path.dirname(checkpoint_path), "model_config.yaml")
    if not os.path.exists(model_cfg):
        # Looks at parent dir
        model_cfg = os.path.join(
            os.path.dirname(os.path.dirname(checkpoint_path)), "model_config.yaml"
        )

    model_cfg = get_config(model_cfg)

    # Disable face for inference
    model_cfg.defrost()
    model_cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = mhr_path
    model_cfg.freeze()

    # Initialze the model
    model = SAM3DBody(model_cfg)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    load_state_dict(model, state_dict, strict=False)

    model = model.to(device)
    model.eval()
    return model, model_cfg


def _resolve_local_paths(checkpoint_path: str = "", mhr_path: str = ""):
    checkpoint_path = os.path.expanduser(checkpoint_path) if checkpoint_path else ""
    mhr_path = os.path.expanduser(mhr_path) if mhr_path else ""
    if not checkpoint_path:
        return "", ""

    if not mhr_path:
        mhr_path = os.path.join(
            os.path.dirname(checkpoint_path), "assets", "mhr_model.pt"
        )
    return checkpoint_path, mhr_path


def _find_local_checkpoint(repo_id: str):
    repo_name = repo_id.split("/")[-1]
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    local_dir = os.path.join(project_root, "checkpoints", repo_name)

    ckpt_path = os.path.join(local_dir, "model.ckpt")
    mhr_path = os.path.join(local_dir, "assets", "mhr_model.pt")
    if os.path.exists(ckpt_path) and os.path.exists(mhr_path):
        return ckpt_path, mhr_path
    return "", ""


def _hf_download(repo_id, token=None):
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError

    try:
        local_dir = snapshot_download(repo_id=repo_id, token=token)
    except HfHubHTTPError as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code in {401, 403}:
            raise RuntimeError(
                "Access denied while downloading SAM 3D Body weights from Hugging Face. "
                "This repo is gated. Request access on the model page, then authenticate "
                "(`huggingface-cli login`) or set the `HF_TOKEN` environment variable."
            ) from exc
        raise

    return os.path.join(local_dir, "model.ckpt"), os.path.join(
        local_dir, "assets", "mhr_model.pt"
    )


def load_sam_3d_body_hf(repo_id, token=None, checkpoint_path="", mhr_path="", **kwargs):
    ckpt_path, mhr_path = _resolve_local_paths(checkpoint_path, mhr_path)
    if ckpt_path and os.path.exists(ckpt_path) and os.path.exists(mhr_path):
        return load_sam_3d_body(checkpoint_path=ckpt_path, mhr_path=mhr_path, **kwargs)

    local_ckpt_path, local_mhr_path = _find_local_checkpoint(repo_id)
    if local_ckpt_path:
        return load_sam_3d_body(
            checkpoint_path=local_ckpt_path, mhr_path=local_mhr_path, **kwargs
        )

    token = token or os.getenv("HF_TOKEN")
    ckpt_path, mhr_path = _hf_download(repo_id, token=token)
    return load_sam_3d_body(checkpoint_path=ckpt_path, mhr_path=mhr_path, **kwargs)
