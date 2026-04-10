from argparse import ArgumentParser, Namespace
import os
from random import randint
import sys
import uuid

import torch
import tqdm

from arguments import (
    ExtractedModelParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
)
from gaussian_renderer import network_gui, render
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general import get_expon_lr_func, safe_state
from utils.loss import l1_loss, ssim

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim

    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False

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

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(
        range(first_iter, opt_params.iterations), desc="Training progress"
    )
    first_iter += 1
    for iteration in range(first_iter, opt_params.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe_params.convert_SHs_python,
                    pipe_params.compute_cov3D_python,
                    keep_alive,
                    scaling_modifier,
                ) = network_gui.receive()
                if custom_cam != None:
                    net_image = render(
                        custom_cam,
                        gaussians,
                        pipe_params,
                        background,
                        scaling_modifier,
                        SPARSE_ADAM_AVAILABLE,
                        use_trained_exp=train_params.train_text_exp,
                    )["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, train_params.source_path)
                if do_training and (
                    (iteration < int(opt_params.iterations)) or not keep_alive
                ):
                    break
            except Exception as e:
                network_gui.conn = None

    iter_start.record()

    gaussians.update_learning_rate(iteration)

    if iteration % 1000 == 0:
        gaussians.oneupSHdegree()

    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_indices = list(range(len(viewpoint_stack)))
    rand_idx = randint(0, len(viewpoint_indices) - 1)
    viewpoint_cam = viewpoint_stack.pop(rand_idx)
    vind = viewpoint_indices.pop(rand_idx)

    if iteration - 1 == debug_from:
        pipe_params.debug = True

    bg = torch.rand((3), device="cuda") if opt_params.random_background else background

    render_pkg = render(
        viewpoint_cam,
        gaussians,
        pipe_params,
        bg,
        use_trained_exp=train_params.train_text_exp,
        separate_sh=SPARSE_ADAM_AVAILABLE,
    )
    image, viewspace_point_tensor, visibility_filter, radii = (
        render_pkg.render,
        render_pkg.viewspace_points,
        render_pkg.visibility_filter,
        render_pkg.radii,
    )

    gt_image = viewpoint_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    if FUSED_SSIM_AVAILABLE:
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
    else:
        ssim_value = ssim(image, gt_image)

    loss = (1.0 - opt_params.lambda_dssim) * Ll1 + opt_params.lambda_dssim * (
        1.0 - ssim_value
    )

    Ll1depth_pure = 0.0
    if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
        invDepth = render_pkg.depth
        mono_invdepth = viewpoint_cam.invdepthmap.cuda()
        depth_mask = viewpoint_cam.depth_mask.cuda()

        Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
        Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
        loss += Ll1depth
        Ll1depth = Ll1depth.item()
    else:
        Ll1depth = 0

    loss.backward()


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
