from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from html import parser
import os
import sys


# TODO: Try to move these into dataclass later
class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t is bool:
                    group.add_argument(
                        "--" + key, ("-") + key[0:1], default=value, action="store_true"
                    )
                else:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, type=t
                    )
            else:
                if t is bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])

        return group


@dataclass
class ExtractedModelParams:
    sh_degree: int = 3
    source_path: str = ""
    model_path: str = ""
    images: str = "images"
    depths: str = ""
    resolution: int = -1
    white_background: bool = False
    train_test_exp: bool = False
    data_device: str = "cuda"
    eval: bool = False


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        typed_group = ExtractedModelParams(
            sh_degree=g.sh_degree,
            source_path=os.path.abspath(g.source_path),
            model_path=g.model_path,
            images=g.images,
            depths=g.depths,
            resolution=g.resolution,
            white_background=g.white_background,
            train_test_exp=g.train_test_exp,
            data_device=g.data_device,
            eval=g.eval,
        )
        return typed_group


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 16e-5
        self.position_lr_final = 16e-7
        self.position_lr_delay_mult = 1e-2
        self.position_lr_max_steps = 30_000
        self.feature_lr = 25e-4
        self.opacity_lr = 25e-3
        self.scaling_lr = 5e-3
        self.rotation_lr = 1e-3
        self.exposure_lr_init = 1e-2
        self.exposure_lr_final = 1e-3
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 1e-2
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 2e-4
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 1e-2
        self.random_background = False
        self.optimizer_type = "default"
        super().__init__(parser, "Optimization Parameters")


def get_combined_args():
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)