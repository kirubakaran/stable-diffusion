import random
import string
from flask import Flask, request, jsonify

import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor


# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    zmodel = instantiate_from_config(config.model)
    m, u = zmodel.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    zmodel.cuda()
    zmodel.eval()
    return zmodel


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


# https://stackoverflow.com/a/23689767
class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def params(req=None):
    parsed = {
        "name": None,
        "prompt": None,
        "ddim_steps": 50,
        "plms": True,
        "laion400m": False,
        "fixed_code": True,
        "ddim_eta": 0.0,
        "H": 512,
        "W": 512,
        "C": 4,
        "f": 8,
        "n_samples": 1,
        "scale": 7.5,
        "from_file": None,
        "ckpt": "models/ldm/stable-diffusion-v1/model.ckpt",
        "seed": 42,
        "precision": "autocast",
        "config": "configs/stable-diffusion/v1-inference.yaml",
    }

    if req is not None:
        strkeys = ["prompt", "name", "seed"]
        for key in strkeys:
            parsed[key] = req.args.get(key, parsed[key])
        intkeys = ["seed", "H", "W"]
        for key in intkeys:
            parsed[key] = int(req.args.get(key, parsed[key]))

    return DotDict(parsed)


# Globals
# TODO : don't use globals
start_code = None
sampler = None
model = None
device = None
preloaded = False


def preload():
    tic = time.time()
    global start_code
    global sampler
    global model
    global device
    global preloaded

    opt = params()

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    toc = time.time()
    preloaded = True
    return toc - tic


def render(r):
    global start_code
    global sampler
    global model
    global device

    if not preloaded:
        preload()

    sample_path = "/db/output/txt2img/static/"

    opt = params(r)

    batch_size = opt.n_samples
    prompt = opt.prompt
    assert prompt is not None
    data = [batch_size * [prompt]]

    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    nsfw = False
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()

                for prompts in tqdm(data, desc="data"):
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                     conditioning=c,
                                                     batch_size=opt.n_samples,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=opt.ddim_eta,
                                                     x_T=start_code)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                    nsfw = nsfw or any(has_nsfw_concept)

                    x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                    for x_sample in x_checked_image_torch:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))

                        # png
                        # filename = os.path.join(sample_path, f"{opt.name}.png")
                        # img.save(filename)

                        # jpegs
                        img_rgb = img.convert("RGB")
                        filename_512 = os.path.join(sample_path, f"{opt.name}-512.jpg")
                        img_rgb.save(filename_512, "JPEG")
                        filename_256 = os.path.join(sample_path, f"{opt.name}-256.jpg")
                        img_rgb.resize((256,256)).save(filename_256, "JPEG")

                toc = time.time()

    return {
        "item_id": opt.name,
        "nsfw": nsfw,
        "elapsed": toc - tic,
        "status": 200,
        "error": False,
        "errmsg": None,
        "errcode": 0,
    }


def create_app():
    preload()
    app = Flask(__name__)

    @app.route("/boot")
    def boot():
        elapsed = preload()
        return jsonify({
            "elapsed": elapsed,
            "status": 200,
            "error": False,
            "errmsg": None,
            "errcode": 0,
        })

    @app.route("/")
    def home():
        name = request.args.get("name")
        prompt = request.args.get("prompt")
        if name is None or prompt is None:
            return jsonify({
                "error": True,
                "errcode": 1,
                "errmsg": "name or prompt not sent",
                "status": 400,
            }), 400
        result = render(request)
        return jsonify(result)

    return app
