from argparse import ArgumentParser
import os
from typing import List

import torch
import torchvision
import tqdm

from arguments import ExtractedModelParams, ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import Scene
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from train import SPARSE_ADAM_AVAILABLE
from utils.general import safe_state


def render_set(
    model_path: str,
    name: str,
    iteration: int,
    views: List[Camera],
    gaussians: GaussianModel,
    pipeline: PipelineParams,
    background: torch.Tensor,
    train_test_exp: bool,
    separate_sh: bool,
):
    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(
            view,
            gaussians,
            pipeline,
            background,
            use_trained_exp=train_test_exp,
            separate_sh=separate_sh,
        )["render"]
        gt = view.original_image[0:3, :, :]

        if args.train_test.exp:
            rendering = rendering[..., rendering.shape[-1] // 2 :]
            gt = gt[..., gt.shape[-1] // 2 :]

        torchvision.utils.save_image(
            rendering, os.path.join(render_path, f"{idx:05d}.png")
        )
        torchvision.utils.save_image(gt, os.path.join(gts_path, f"{idx:05d}.png"))


def render_sets(
    train_params: ExtractedModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    separate_sh: bool,
):
    with torch.no_grad():
        gaussians = GaussianModel(train_params.sh_degree)
        scene = Scene(train_params, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if train_params.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(
                train_params.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                train_params.train_test_exp,
                separate_sh,
            )

        if not skip_test:
            render_set(
                train_params.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                train_params.train_test_exp,
                separate_sh,
            )

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)