import os

from arguments import ExtractedModelParams
from scene.gaussian_model import GaussianModel
from utils.system import searchForMaxIteration
from dataset_readers import readColmapSceneInfo, sceneLoadTypeCallbacks


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
            scene_info = readColmapSceneInfo(args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"
        
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"), "wb") as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras
