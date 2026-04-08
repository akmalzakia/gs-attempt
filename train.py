from argparse import ArgumentParser, Namespace
import os
import sys
import uuid

import torch

from arguments import (
    ExtractedModelParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
)
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general import get_expon_lr_func, safe_state

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam

    SPARSE_ADAM_AVAILABLE = True
except Exception as e:
    SPARSE_ADAM_AVAILABLE = False


def training(
    train_params: ExtractedModelParams,
    opt_params: OptimizationParams,
    pipe_params: PipelineParams,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
):
    if not SPARSE_ADAM_AVAILABLE and opt_params.optimizer_type == "sparse_adam":
        sys.exit(
            "Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel]."
        )

    first_iter = 0
    tb_writer = prepare_output_and_logger(train_params)
    gaussians = GaussianModel(train_params.sh_degree, opt_params.optimizer_type)
    scene = Scene(train_params, gaussians)
    gaussians.training_setup(opt_params)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt_params)

    bg_color = [1, 1, 1] if train_params.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    use_sparse_adam = (
        opt_params.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    )
    depth_l1_weight = get_expon_lr_func(
        opt_params.depth_l1_weight_init,
        opt_params.depth_l1_weight_final,
        max_steps=opt_params.iterations,
    )

    


def prepare_output_and_logger(args: ExtractedModelParams):
    oar_key = "OAR_JOB_ID"
    if not args.model_path:
        if os.getenv(oar_key):
            unique_str = os.getenv(oar_key)
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_viewer", action="store_true", default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    # if not args.disable_viewer:
    #     network_gui.init(args.ip, args.port)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
    )
