import copy
import os
import sys
import os.path as osp
import torch
import numpy as np
import trimesh
import polyscope as ps

from scipy.spatial.transform import Rotation

from ndf_robot.eval.ndf_paritial_map import NDFPartialMap
from ndf_robot.utils import path_util, torch_util
import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))

if __name__ == '__main__':
    ref_id = 102

    # load two point clouds
    dir_pcd_1 = BASE_DIR + '/data/pcd_1.8.npz'
    dir_pcd_2 = BASE_DIR + '/data/pcd_2.5.npz'
    pcd_1 = np.load(dir_pcd_1, allow_pickle=True)
    pcd_2 = np.load(dir_pcd_2, allow_pickle=True)
    pcd1 = pcd_1["pcd"]
    pcd2 = pcd_2["pcd"]

    dir_grasp_1 = BASE_DIR + '/data/1.8.npz'
    dir_grasp_2 = BASE_DIR + '/data/2.5.npz'
    pcd_1 = np.load(dir_grasp_1, allow_pickle=True)
    pcd_2 = np.load(dir_grasp_2, allow_pickle=True)
    grasps_1 = pcd_1["grasps"]
    grasps_2 = pcd_2["grasps"]
    # # show point clouds
    # tpcd1 = trimesh.PointCloud(pcd1)
    # tpcd2 = trimesh.PointCloud(pcd2)
    # tpcd1.show()
    # tpcd2.show()

    # preprocess the point cloud and randomly sample 2000 points
    # first filter out outliers
    pcd_1_mean = np.mean(pcd1, axis=0)
    index_1 = np.where(np.linalg.norm(pcd1 - pcd_1_mean, 2, 1) < 0.2)[0]
    pcd1 = pcd1[index_1]
    index_1 = np.where(pcd1[:, -1] >= np.min(pcd1[:, -1]) + 0.00004)
    pcd1 = pcd1[index_1]
    pcd_2_mean = np.mean(pcd1, axis=0)
    index_2 = np.where(np.linalg.norm(pcd2 - pcd_2_mean, 2, 1) < 0.19)[0]
    pcd2 = pcd2[index_2]
    index_2 = np.where(pcd2[:, -1] >= np.min(pcd2[:, -1]) + 0.00005)
    pcd2 = pcd2[index_2]

    mean_1 = np.mean(pcd1, axis=0)
    mean_2 = np.mean(pcd2, axis=0)
    pcd1 = pcd1 - mean_1
    pcd2 = pcd2 - mean_2
    np.random.shuffle(pcd1)
    np.random.shuffle(pcd2)
    # # show point clouds
    # tpcd1 = trimesh.PointCloud(pcd1)
    # tpcd2 = trimesh.PointCloud(pcd2)
    # tpcd1.show()
    # tpcd2.show()

    # load model
    model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_demo_mug_weights.pth')
    model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=True,
                                            sigmoid=True).cuda()
    model.load_state_dict(torch.load(model_path))

    # device
    dev = torch.device('cuda:0')

    # loss
    loss_fn = torch.nn.L1Loss()

    # get query points
    # sample query points
    n = 500
    query_x = np.random.uniform(-0.02, 0.02, n)
    query_y = np.random.uniform(-0.04, 0.04, n)
    query_z = np.random.uniform(-0.05 + 0.1, 0.02 + 0.1, n)
    # query_z = np.random.uniform(-0.05 + 0.1, 0.01 + 0.1, n)
    ones = np.ones(n)
    ref_pts_gripper = np.vstack([query_x, query_y, query_z])
    ref_pts_gripper = ref_pts_gripper.T
    hom_query_pts = np.vstack([query_x, query_y, query_z, ones])

    # transform
    ref_query_pts = grasps_1[ref_id] @ hom_query_pts
    ref_query_pts = ref_query_pts[:3, :] - mean_1[:, None]
    ref_query_pts = ref_query_pts.T

    query_pts_vis = copy.deepcopy(ref_query_pts)

    # ref
    reference_model_input = {}
    ref_query_pts = torch.from_numpy(ref_query_pts).float().to(dev)
    ref_shape_pcd = torch.from_numpy(pcd1[:2000]).float().to(dev)
    reference_model_input['coords'] = ref_query_pts[None, :, :]
    reference_model_input['point_cloud'] = ref_shape_pcd[None, :, :]

    # get the descriptors for these reference query points
    reference_latent = model.extract_latent(reference_model_input).detach()
    reference_act_hat = model.forward_latent(reference_latent, reference_model_input['coords']).detach()

    # get the descriptors for all other grasps
    # n_grasp = grasps_2.shape[0]
    n_grasp = 30
    opt_query_pts = torch.from_numpy(ref_pts_gripper).float().to(dev)
    opt_query_pts = opt_query_pts[None, :, :].repeat((n_grasp, 1, 1))
    trans = torch.from_numpy(grasps_2[60:90]).squeeze().float().to(dev)
    X = torch_util.transform_pcd_torch(opt_query_pts, trans)
    mu_2 = torch.from_numpy(mean_2)[None, None, :].float().to(dev)
    X = X - mu_2
    opt_model_input = {}
    opt_model_input['coords'] = X

    mi_point_cloud = []
    for ii in range(n_grasp):
        mi_point_cloud.append(torch.from_numpy(pcd2[:2000]).float().to(dev))
    mi_point_cloud = torch.stack(mi_point_cloud, 0)
    opt_model_input['point_cloud'] = mi_point_cloud
    opt_latent = model.extract_latent(opt_model_input).detach()
    act_hat = model.forward_latent(opt_latent, X)

    t_size = reference_act_hat.size()
    losses = [loss_fn(act_hat[ii].view(t_size), reference_act_hat) for ii in range(n_grasp)]
    losses_str = ['%f' % val.item() for val in losses]
    loss_str = ', '.join(losses_str)
    print(f'i: {0}, losses: {loss_str}')
    loss_np = [losses[ii].detach().cpu().numpy() for ii in range(n_grasp)]

    # visualization
    coords = np.array([[0.1, 0, 0, 0],
                       [0., 0.1, 0, 0],
                       [0, 0, 0.1, 0],
                       [1, 1, 1, 1]])

    coords_1 = grasps_1[ref_id] @ coords
    coords_1 = coords_1[0:3, :] - mean_1[:, None]
    coords_1 = coords_1.T

    # ps.init()
    # ps.set_up_dir("z_up")
    #
    # ps.register_point_cloud("pcd1", pcd1[:2000], radius=0.005, enabled=True)
    # ps.register_point_cloud("pcd2", query_pts_vis, radius=0.005, enabled=True)
    # ps.register_point_cloud("pcd3", pcd2[:2000], radius=0.005, enabled=True)
    # ps.register_point_cloud("pcd4", X[0].detach().cpu().numpy(), radius=0.005, enabled=True)
    #
    # ps.register_curve_network("edge_x" + str(1), coords_1[[0, 3]], np.array([[0, 1]]),
    #                           enabled=True, radius=0.0003, color=(1, 0, 0))
    # ps.register_curve_network("edge_y" + str(1), coords_1[[1, 3]], np.array([[0, 1]]),
    #                           enabled=True, radius=0.0003, color=(0, 1, 0))
    # ps.register_curve_network("edge_z" + str(1), coords_1[[2, 3]], np.array([[0, 1]]),
    #                           enabled=True, radius=0.0003, color=(0, 0, 1))
    #
    # ps.show()

    gripper_control_points = np.array([[0.00000, 0.00000, 0.00000],
                                       [0.00000, 0.00000, 0.05840],
                                       [0.05269, -0.00006, 0.05840],
                                       [0.05269, -0.00006, 0.10527],
                                       [0.05269, -0.00006, 0.05840],
                                       [-0.05269, 0.00006, 0.05840],
                                       [-0.05269, 0.00006, 0.10527]])

    all_nodes = []
    all_edges = []
    index = 0
    n_pts = 7
    # show grasp (gripper)
    for i in range(n_grasp):
        coords = np.concatenate((gripper_control_points, np.ones((7, 1))), axis=1)
        coords = grasps_2[i+60] @ coords.T
        coords = coords[0:3, :]
        coords = coords.T
        nodes = coords
        all_nodes.append(nodes)
        all_edges.append(np.array([[index, index + 1],
                                   [index + 2, index + 5],
                                   [index + 2, index + 3],
                                   [index + 5, index + 6]]))
        index += n_pts
    all_nodes = np.vstack(all_nodes)
    all_edges = np.vstack(all_edges)
    print(all_edges.shape)
    probs = np.asarray(loss_np)
    probs = np.repeat(probs, 4)

    ps.init()
    ps.set_up_dir("z_up")
    ps.register_point_cloud("pcd3", query_pts_vis + mean_1, radius=0.005, enabled=True)
    ps.register_point_cloud("pcd1", pcd1[:2000] + mean_1, radius=0.005, enabled=True)
    ps.register_point_cloud("pcd2", pcd2[:2000] + mean_2, radius=0.005, enabled=True)
    ps3 = ps.register_curve_network("gripper", all_nodes, all_edges, radius=0.002, enabled=True)
    ps3.add_scalar_quantity("probs", probs, defined_on='edges', cmap='coolwarm', enabled=True)

    ps.show()

