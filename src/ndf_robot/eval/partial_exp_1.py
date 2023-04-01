import os
import sys
import os.path as osp
import torch
import numpy as np
import trimesh
import random
import argparse
import copy
from scipy.spatial.transform import Rotation

from ndf_robot.eval.ndf_paritial_map import NDFPartialMap
from ndf_robot.utils import path_util
import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))

if __name__ == '__main__':
    # load two point clouds
    dir_pcd_1 = BASE_DIR + '/data/scene_0.npz'
    dir_pcd_2 = BASE_DIR + '/data/scene_1.npz'
    pcd_1 = np.load(dir_pcd_1, allow_pickle=True)
    pcd_2 = np.load(dir_pcd_2, allow_pickle=True)
    pcd1 = pcd_1["pcd"]
    pcd2 = pcd_2["pcd"]

    # # show point clouds
    # tpcd1 = trimesh.PointCloud(pcd1)
    # tpcd2 = trimesh.PointCloud(pcd2)
    # tpcd1.show()
    # tpcd2.show()

    # preprocess the point cloud and randomly sample 2000 points
    # first filter out outliers
    pcd_1_mean = np.mean(pcd1, axis=0)
    # index_1 = np.where(np.linalg.norm(pcd1 - pcd_1_mean, 2, 1) < 0.16)[0]
    # pcd1 = pcd1[index_1]
    index_1 = np.where(pcd1[:, -1] >= np.min(pcd1[:, -1]) + 0.00004)
    pcd1 = pcd1[index_1]
    pcd_2_mean = np.mean(pcd1, axis=0)
    # index_2 = np.where(np.linalg.norm(pcd2 - pcd_2_mean, 2, 1) < 0.16)[0]
    # pcd2 = pcd2[index_2]
    index_2 = np.where(pcd2[:, -1] >= np.min(pcd2[:, -1]) + 0.00004)
    pcd2 = pcd2[index_2]

    # # show point clouds
    # tpcd1 = trimesh.PointCloud(pcd1)
    # tpcd2 = trimesh.PointCloud(pcd2)
    # tpcd1.show()
    # tpcd2.show()

    obj_model1 = osp.join(path_util.get_ndf_demo_obj_descriptions(),
                          'mug_centered_obj_normalized/28f1e7bc572a633cb9946438ed40eeb9/models/model_normalized.obj')
    scale1 = 1
    mesh1 = trimesh.load(obj_model1, process=False)
    mesh1.apply_scale(scale1)

    scale2 = 1
    obj_model2 = osp.join(path_util.get_ndf_demo_obj_descriptions(),
                          'mug_centered_obj_normalized/586e67c53f181dc22adf8abaa25e0215/models/model_normalized.obj')
    mesh2 = trimesh.load(obj_model2, process=False)  # different instance, different scaling
    mesh2.apply_scale(scale2)

    r = Rotation.from_euler('x', [90], degrees=True)
    quat = r.as_quat()
    rot = np.eye(4)
    rot[:-1, :-1] = Rotation.from_quat(quat).as_matrix()
    mesh2.apply_transform(rot)

    pcd3 = mesh1.sample(5000)

    pcd4 = mesh2.sample(5000)

    # tpcd3 = trimesh.PointCloud(pcd3)
    # tpcd3.show()

    # reference grasp in world coordinate
    # grasp_ref = np.array([[-0.75646931,  0.49684362,  0.42532411, -0.04318861],
    #                       [ 0.65286087,  0.53478415,  0.53645   ,  0.8694924 ],
    #                       [ 0.03907517,  0.68348543, -0.72891755,  0.94439177],
    #                       [ 0.        ,  0.        ,  0.        ,  1.        ]])

    grasp_ref = np.array([[9.77081195e-01, 1.05711363e-01, -1.84763217e-01,
      1.47614338e-01],
     [1.07638491e-01, -9.94190019e-01, 4.02482777e-04,
      1.02524533e+00],
     [-1.83647199e-01, -2.02808923e-02, -9.82782983e-01,
      1.00742185e+00],
     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
      1.00000000e+00]])


    # grasp_ref = np.array([[ 0.98304272, -0.15708371, -0.09461355, -0.02952739],
    #    [-0.17916859, -0.71286937, -0.67802351,  1.13515868],
    #    [ 0.03905934,  0.68347785, -0.72892551,  0.94439438],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]])
    # grasp_ref = np.array([[-0.72923333,  0.45574179,  0.51040981, -0.06389819],
    #                       [ 0.68355171,  0.45113277,  0.57379115,  0.85777503],
    #                       [ 0.03123801,  0.76731913, -0.64050413,  0.92599702],
    #                       [ 0.        ,  0.        ,  0.        ,  1.        ]])

    # load model
    model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_demo_mug_weights.pth')
    model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=True,
                                            sigmoid=True).cuda()
    model.load_state_dict(torch.load(model_path))

    ndf_alignment = NDFPartialMap(model, pcd1, pcd2, grasp_ref, sigma=0.025)
    # ndf_alignment = NDFPartialMap(model, pcd4, pcd1, grasp_ref, sigma=0.025)
    ndf_alignment.sample_pts()
