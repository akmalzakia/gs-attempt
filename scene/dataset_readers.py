import json
import os
import sys

import numpy as np

from scene.colmap_loader import (
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_intrinsics_text,
)
from utils.graphics import getWorld2View2


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    return {"translate": translate, "radius": radius}


def readColmapCameras(
    cam_extrinsics,
    cam_instrinsics,
    depth_params,
    images_folder,
    depth_folder,
    test_cam_names_list,
):
    pass


def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    colmap_sparse = "sparse/0"
    try:
        cameras_extrinsic_file = os.path.join(path, colmap_sparse, "images.bin")
        cameras_intrinsic_file = os.path.join(path, colmap_sparse, "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except Exception as e:
        cameras_extrinsic_file = os.path.join(path, colmap_sparse, "images.txt")
        cameras_intrinsic_file = os.path.join(path, colmap_sparse, "cameras.txt")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, colmap_sparse, "depth_params.json")

    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array(
                [depths_params[key]["scale"] for key in depths_params]
            )
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0

            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale
        except FileNotFoundError:
            print(
                f"Error: depth_params.json file not found at path '{depth_params_file}'."
            )
            sys.exit(1)
        except Exception as e:
            print(
                f"An unexpected error occurred when trying to open depth_params.json file: {e}"
            )
            sys.exit(1)
    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [
                name for idx, name in enumerate(cam_names) if idx % llffhold == 0
            ]
        else:
            with open(os.path.join(path, colmap_sparse, "test.txt"), "r") as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_instrinsics=cam_intrinsics,
        depth_params=depths_params,
        images_folder=os.path.join(path, reading_dir),
        depths_folder=os.path.join(path, depths) if depths != "" else "",
        test_cam_names_list=test_cam_names_list,
    )

    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    sparse_path = "sparse/0"
    ply_path = os.path.join(path, f"{sparse_path}/points3D.ply")
    bin_path = os.path.join(path, f"{sparse_path}/points3D.bin")
    txt_path = os.path.join(path, f"{sparse_path}/points3D.txt")

    if not os.path.exists(ply_path):
        try:
            xyz, rgb, _ = readpoint
        except expression as identifier:
            pass
