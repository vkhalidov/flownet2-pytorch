import os
from dataclasses import dataclass
import logging
import requests
import shutil
import tempfile
import torch
from tqdm import tqdm
from typing import BinaryIO, Optional
from utils import tools

import models


__all__ = ["flownet2"]


_GDRIVE_MODELS = {
    "FlowNet2": {
        "id": "1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da",
        "size_b": 650102505, 
    },
}


@dataclass
class FlowNetArgs:
    rgb_max: float = 255.0
    fp16: bool = False


def _download_url_to_file_gdrive(model_name: str, dst: str, progress: bool = True) -> None:
    """
    Google drive files more than 100MB require confirmation for download
    This method first requests an URL. If a confirmation token is present
    in the response, reads the confirmation token and makes another request
    with the confirmation code.
    Implementation inspired by:
    https://stackoverflow.com/a/39225272
    """
    _URL = "https://docs.google.com/uc?export=download"
    logger = logging.getLogger(__name__)
    session = requests.Session()
    model_id = _GDRIVE_MODELS[model_name]["id"]
    response = session.get(_URL, params={"id": model_id}, stream=True)
    token = _get_confirm_token(response)
    if token:
        params = {"id": model_id, "confirm": token}
        response = session.get(_URL, params=params, stream=True)
    dst_dir = os.path.dirname(dst)
    with tempfile.NamedTemporaryFile(delete=False, dir=dst_dir) as f:
        logger.info(
            f"Downloading model {model_name} from Google Drive object {model_id} to {f.name}")
        file_size = _get_file_size(response, model_name)
        _save_response_content(response, f, file_size, progress)
    logger.info(f"Renaming {f.name} to {dst}")
    shutil.move(f.name, dst)


def _get_confirm_token(response: requests.Response) -> Optional[str]:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def _get_file_size(response: requests.Response, model_name: str) -> int:
    _CONTENT_LENGTH_KEY = "Content-length"
    if _CONTENT_LENGTH_KEY in response.headers:
        # NB: content length is not sent while the stream is not closed
        file_size = response.headers[_CONTENT_LENGTH_KEY]
    else:
        # static data, may be inaccurate
        file_size = _GDRIVE_MODELS[model_name]["size_b"]
    return file_size


def _save_response_content(
    response: requests.Response, dst: BinaryIO, file_size: int, progress: bool = True
) -> None:
    CHUNK_SIZE = 32768
    with tqdm(
        total=file_size, disable=not progress, unit="B", unit_scale=True, unit_divisor=1024
    ) as pbar:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                dst.write(chunk)
                pbar.update(len(chunk))


def _flownet_model(model_name: str, args: FlowNetArgs):
    model_class = tools.module_to_dict(models)[model_name]
    kwargs = tools.kwargs_from_args(args, "model")
    model = model_class(args, **kwargs)
    return model


ENV_TORCH_HOME = "TORCH_HOME"
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
DEFAULT_CACHE_DIR = "~/.cache"


def _get_torch_home():
    torch_home = os.path.expanduser(
        os.getenv(
            ENV_TORCH_HOME, os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), "torch")
        )
    )
    return torch_home


def _flownet(model_name: str, pretrained=False, progress=True, **kwargs):
    model_args = FlowNetArgs()
    for key, val in kwargs.items():
        setattr(model_args, key, val)
    model = _flownet_model(model_name, model_args)
    if pretrained:
        torch_home = _get_torch_home()
        model_dir = os.path.join(torch_home, "checkpoints")
        cached_fname = f"{model_name}.pth.tar"
        cached_fpath = os.path.join(model_dir, cached_fname)
        _download_url_to_file_gdrive(model_name, cached_fpath, progress=progress)
        checkpoint = torch.load(cached_fpath)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
    return model


def flownet2(pretrained=False, progress=True, **kwargs):
    """ # This docstring shows up in hub.help()
    FlowNet2 model
    pretrained (bool): load pretrained weights into the model
    """
    model_name = "FlowNet2"
    model = _flownet(model_name, pretrained, progress, **kwargs)
    return model


def apply_model(model, images_from, images_to):
    """
    Applies optical flow model to the pairs of images

    Args:

    images_from: torch.Tensor of size [B, H, W, C] containing RGB data for
        images that serve as optical flow source images
    images_to: torch.Tensor of size [B, H, W, C] containing RGB data for
        images that serve as optical flow destination images

    Return:

    optical_flow: torch.Tensor of size [B, H, W, 2]
    """
    D = 64
    # 2x [B, H, W, C] -> [B, C, 2, H, W]
    images = torch.stack([images_from, images_to]).permute(1, 4, 0, 2, 3)
    B, C, T, H, W = images.shape
    He = (H + D - 1) // D * D
    We = (W + D - 1) // D * D
    images_enhanced = torch.zeros(B, C, T, He, We, device=images.device)
    images_enhanced[:, :, :, :H, :W] = images
    with torch.no_grad():
        flow = model(images_enhanced)
        return flow
