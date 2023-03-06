import os
import os.path as osp
import torch
import numpy as np
import trimesh
import random
import argparse
import copy
from scipy.spatial.transform import Rotation
from ndf_robot.share.globals import bad_shapenet_mug_ids_list, bad_shapenet_bowls_ids_list, bad_shapenet_bottles_ids_list

from ndf_robot.utils import path_util
import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
import matplotlib.pyplot as plt
from ndf_robot.eval.ndf_2d_heatmap import NDFHeatmap


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--show_recon', action='store_true')
    parser.add_argument('--sigma', type=float, default=0.025)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)

    # valid object list for mug
    obj_class = 'mug'
    shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(), obj_class + '_centered_obj_normalized')
    avoid_shapenet_ids = bad_shapenet_mug_ids_list
    shapenet_id_list = [fn.split('_')[0] for fn in os.listdir(shapenet_obj_dir)]
    valid_id_list = []
    for i in shapenet_id_list:
        if i not in avoid_shapenet_ids:
            valid_id_list.append(i)

    print(valid_id_list[0])

    # set demo objects
    obj_model1 = osp.join(path_util.get_ndf_demo_obj_descriptions(),
                          'mug_centered_obj_normalized/28f1e7bc572a633cb9946438ed40eeb9/models/model_normalized.obj')
    #                       'mug_centered_obj_normalized/586e67c53f181dc22adf8abaa25e0215/models/model_normalized.obj')
    obj_model2 = osp.join(path_util.get_ndf_demo_obj_descriptions(),
    #                       'mug_centered_obj_normalized/586e67c53f181dc22adf8abaa25e0215/models/model_normalized.obj')
                          'mug_centered_obj_normalized/8f6c86feaa74698d5c91ee20ade72edc/models/model_normalized.obj')
    model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_demo_mug_weights.pth')

    scale1 = 0.25
    scale2 = 0.4

    mesh1 = trimesh.load(obj_model1, process=False)
    mesh1.apply_scale(scale1)
    mesh2 = trimesh.load(obj_model2, process=False)  # different instance, different scaling
    mesh2.apply_scale(scale2)

    # apply a random initial rotation to the new shape
    # quat = np.random.random(4)
    # quat = quat / np.linalg.norm(quat)
    # rot = np.eye(4)
    # rot[:-1, :-1] = Rotation.from_quat(quat).as_matrix()
    # mesh2.apply_transform(rot)

    # show two objects
    if args.visualize:
        show_mesh1 = mesh1.copy()
        show_mesh2 = mesh2.copy()

        offset = 0.1
        show_mesh1.apply_translation([-1.0 * offset, 0, 0])
        show_mesh2.apply_translation([offset, 0, 0])

        scene = trimesh.Scene()
        scene.add_geometry([show_mesh1, show_mesh2])
        # scene.show()

    # point cloud
    pcd1 = mesh1.sample(5000)
    pcd2 = mesh2.sample(5000)

    # load model
    model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=True,
                                            sigmoid=True).cuda()
    model.load_state_dict(torch.load(model_path))

    ndf_alignment = NDFHeatmap(model, pcd1, pcd2, sigma=args.sigma, trimesh_viz=args.visualize)
    ndf_alignment.sample_pts()
