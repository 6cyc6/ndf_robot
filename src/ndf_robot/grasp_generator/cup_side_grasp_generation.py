import os
import sys
import os.path as osp
import torch
import numpy as np
import polyscope as ps
import trimesh
import copy

from ndf_robot.grasp_generator.ndf_grasp_generator_env import NDFGraspGeneratorSide

from ndf_robot.utils import path_util
import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from task_oriented_grasping.utils.path_utils import get_scene_path
from task_oriented_grasping.utils.utils import matrix_from_pose

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))

if __name__ == '__main__':
    # load two point clouds
    pcd_dir = get_scene_path(file_name="0/0.npz")
    data = np.load(pcd_dir, allow_pickle=True)
    pcd1 = data["pcd_local"]
    pcd2 = data["pcd_local2"]
    pcd3 = data["pcd_local3"]
    pcd4 = data["pcd_local4"]
    obj_pos = data["obj_pos"]
    obj_quat = data["obj_quat"]
    obj_mat = matrix_from_pose(obj_pos, obj_quat)

    # pcd_mean_merged = np.mean(pcd_merged, axis=0)
    # pcd_merged = pcd_merged - pcd_mean_merged
    #
    # pcd_mean_sampled = np.mean(pcd_sampled, axis=0)
    # pcd_sampled = pcd_sampled - pcd_mean_sampled

    # grasp_ref = np.array([[0, 0, -1, 0.09],
    #                       [1, 0, 0, 0.98],
    #                       [0, -1, 0, 0.95],
    #                       [0, 0, 0, 1]])
    grasp_ref = np.array([[1, 0, 0, 0.0],
                          [0, 0, 1, 0.93],
                          [0, -1, 0, 0.94],
                          [0, 0, 0, 1]])

    # # for visualization
    # n = 500
    # query_x = np.random.uniform(-0.02, 0.02, n)
    # query_y = np.random.uniform(-0.04, 0.04, n)
    # query_z = np.random.uniform(-0.05 + 0.1, 0.02 + 0.1, n)
    # ones = np.ones(n)
    # ref_pts_gripper = np.vstack([query_x, query_y, query_z])
    # ref_pts_gripper = ref_pts_gripper.T
    # hom_query_pts = np.vstack([query_x, query_y, query_z, ones])
    #
    # ref_query_pts = grasp_ref @ hom_query_pts
    # ref_query_pts = ref_query_pts[:3, :]
    # ref_query_pts = ref_query_pts.T
    #
    # gripper_control_points = np.array([[0.00000, 0.00000, 0.00000],
    #                                    [0.00000, 0.00000, 0.05840],
    #                                    [0.05269, -0.00006, 0.05840],
    #                                    [0.05269, -0.00006, 0.10527],
    #                                    [0.05269, -0.00006, 0.05840],
    #                                    [-0.05269, 0.00006, 0.05840],
    #                                    [-0.05269, 0.00006, 0.10527]])
    # gripper_control_points_armar = np.array([[0.00000, 0.00000, 0.00000],
    #                                          [0.00000, 0.00000, 0.06000],
    #                                          [-0.0500, 0.00000, 0.11000],
    #                                          [0.07000, 0.00000, 0.13000]])
    # coords = np.concatenate((gripper_control_points, np.ones((7, 1))), axis=1)
    # coords = grasp_ref @ coords.T
    # coords = coords[0:3, :]
    # coords = coords.T
    # nodes = coords
    #
    # coords = np.concatenate((gripper_control_points_armar, np.ones((4, 1))), axis=1)
    # coords = grasp_ref @ coords.T
    # coords = coords[0:3, :]
    # coords = coords.T
    # nodes_armar = coords
    #
    # ps.init()
    # ps.set_up_dir("z_up")
    # # ps.register_point_cloud("pcd1", pcd1, radius=0.005, color=[1, 0, 0], enabled=True)
    # # ps.register_point_cloud("pcd2", pcd2 + np.array([0.3, 0, 0]), radius=0.005, enabled=True)
    # # ps.register_point_cloud("pcd3", pcd3 + np.array([0.6, 0, 0]), radius=0.005, enabled=True)
    # # ps.register_point_cloud("pcd4", pcd4 + np.array([0.9, 0, 0]), radius=0.005, enabled=True)
    # ps.register_point_cloud("pcd", pcd1, radius=0.005, enabled=True)
    # ps.register_point_cloud("query", ref_query_pts, radius=0.005, enabled=True)
    # # ps.register_curve_network("edge_1", nodes[[0, 1]], np.array([[0, 1]]),
    # #                           enabled=True, radius=0.0015, color=(0, 0, 1))
    # # ps.register_curve_network("edge_2", nodes[[2, 5]], np.array([[0, 1]]),
    # #                           enabled=True, radius=0.0015, color=(0, 0, 1))
    # # ps.register_curve_network("edge_3", nodes[[2, 3]], np.array([[0, 1]]),
    # #                           enabled=True, radius=0.0015, color=(1, 0, 0))
    # # ps.register_curve_network("edge_4", nodes[[5, 6]], np.array([[0, 1]]),
    # #                           enabled=True, radius=0.0015, color=(0, 1, 0))
    #
    # ps.register_curve_network("edge_1", nodes_armar[[0, 1]], np.array([[0, 1]]),
    #                           enabled=True, radius=0.0015, color=(0, 0, 1))
    # ps.register_curve_network("edge_2", nodes_armar[[1, 2]], np.array([[0, 1]]),
    #                           enabled=True, radius=0.0015, color=(1, 0, 0))
    # ps.register_curve_network("edge_3", nodes_armar[[1, 3]], np.array([[0, 1]]),
    #                           enabled=True, radius=0.0015, color=(0, 1, 0))
    #
    # ps.show()

    # load model
    model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_demo_mug_weights.pth')
    model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=True,
                                            sigmoid=True).cuda()
    model.load_state_dict(torch.load(model_path))

    ndf_alignment = NDFGraspGeneratorSide(model, pcd1, pcd2, pcd3, pcd4, grasp_ref, obj_mat)
    ndf_alignment.sample_grasps()
    ndf_alignment.save_grasps(filename="1")

    # ps.init()
    # ps.set_up_dir("z_up")
    # ps.register_point_cloud("pcd1", ndf_alignment.pcd1, radius=0.005, color=[1, 0, 0], enabled=True)
    # # ps.register_point_cloud("pcd2", ndf_alignment.pcd2 + np.array([0.3, 0, 0]), radius=0.005, enabled=True)
    # # ps.register_point_cloud("pcd3", ndf_alignment.pcd3 + np.array([0.6, 0, 0]), radius=0.005, enabled=True)
    # # ps.register_point_cloud("pcd4", ndf_alignment.pcd4 + np.array([0.9, 0, 0]), radius=0.005, enabled=True)
    # # ps.register_point_cloud("pcd", ndf_alignment.pcd_merged + np.array([0, 0.3, 0]), radius=0.005, enabled=True)
    # ps.register_point_cloud("query", ndf_alignment.pcd_sampled, radius=0.005, enabled=True)

    # ps.show()
