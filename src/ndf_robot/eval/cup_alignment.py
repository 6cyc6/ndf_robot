import os
import sys
import os.path as osp
import torch
import numpy as np
import polyscope as ps
import trimesh
import copy

from ndf_robot.eval.ndf_grasp_generator import NDFGraspGenerator
from ndf_robot.utils import path_util
import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))

if __name__ == '__main__':
    # load two point clouds
    dir_pcd_1 = BASE_DIR + '/data/0.npz'
    data = np.load(dir_pcd_1, allow_pickle=True)
    pcd1 = data["pcd_local"]
    pcd2 = data["pcd_local2"]
    pcd3 = data["pcd_local3"]
    pcd4 = data["pcd_local4"]

    # dir_mesh = BASE_DIR + '/data/scaled_2.5_2.5_4.obj'
    # mesh1 = trimesh.load(dir_mesh, process=False)
    # pcd3 = mesh1.sample(5000)

    # pcd2 = pcd2 - np.mean(pcd2, axis=0)
    # pcd3 = pcd3 - np.mean(pcd3, axis=0) - np.array([0, -0.01, 0.04])

    # grasp_ref = np.array([[0, 0, -1, 0.09],
    #                       [1, 0, 0, 0.98],
    #                       [0, -1, 0, 0.95],
    #                       [0, 0, 0, 1]])
    grasp_ref = np.array([[1, 0, 0, -0.09],
                          [0, 0, 1, 0.8],
                          [0, -1, 0, 0.92],
                          [0, 0, 0, 1]])

    # pcd1_mean = np.mean(pcd1, axis=0)
    # pcd3_mean = np.mean(pcd3, axis=0)
    # n = 500
    # query_x = np.random.uniform(-0.02, 0.02, n)
    # query_y = np.random.uniform(-0.04, 0.04, n)
    # query_z = np.random.uniform(-0.05 + 0.1, 0.02 + 0.1, n)
    # # query_z = np.random.uniform(-0.05 + 0.1, 0.01 + 0.1, n)
    # ones = np.ones(n)
    # ref_pts_gripper = np.vstack([query_x, query_y, query_z])
    # ref_pts_gripper = ref_pts_gripper.T
    # hom_query_pts = np.vstack([query_x, query_y, query_z, ones])
    #
    # # transform
    # ref_query_pts = grasp_ref @ hom_query_pts
    # ref_query_pts = ref_query_pts[:3, :] - pcd1_mean[:, None]
    # ref_query_pts = ref_query_pts.T
    #
    # ps.init()
    # ps.set_up_dir("z_up")
    # ps.register_point_cloud("grasp", ref_query_pts, radius=0.005, enabled=True)
    # # ps.register_point_cloud("pcd1", pcd1, radius=0.005, enabled=True)
    # ps.register_point_cloud("pcd2", pcd1 - pcd1_mean, radius=0.005, enabled=True)
    #
    # ps.show()

    # ps.init()
    # ps.set_up_dir("z_up")
    # ps.register_point_cloud("pcd1", pcd1, radius=0.005, enabled=True)
    # ps.register_point_cloud("pcd2", pcd2, radius=0.005, enabled=True)
    # ps.register_point_cloud("pcd3", pcd3, radius=0.005, enabled=True)
    # ps.register_point_cloud("pcd4", pcd4, radius=0.005, enabled=True)
    #
    # ps.show()

    # load model
    model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_demo_mug_weights.pth')
    model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=True,
                                            sigmoid=True).cuda()
    model.load_state_dict(torch.load(model_path))

    ndf_alignment = NDFGraspGenerator(model, pcd1, pcd2, pcd3, pcd4, grasp_ref)
    ndf_alignment.sample_pts()
    ndf_alignment.save_grasps()
