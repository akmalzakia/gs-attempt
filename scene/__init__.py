import os

from arguments import ExtractedModelParams
from scene.gaussian_model import GaussianModel
from utils.system import searchForMaxIteration


class Scene:
    gaussians: GaussianModel

    def __init__(
        self,
        args: ExtractedModelParams,
        gaussians: GaussianModel,
        load_iteration=None,
        shuffle=True,
        resolution_scales=[1.0],
    ):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration == -1:
            self.loaded_iter = searchForMaxIteration(
                os.path.join(self.model_path, "point_cloud")
            )
        else:
            self.loaded_iter = load_iteration
            print(f"Loading trained model at iteration {self.loaded_iter}")
        
        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoad

    def getTrainCameras(self, scale=1.0):
        return self.train_ca
