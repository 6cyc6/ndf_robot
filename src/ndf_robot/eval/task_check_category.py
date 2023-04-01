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
    # ref_id_1 = 0
    # ref_id_2 = 73
    # ref_id_3 = 36

    ref_id_1 = 0
    ref_id_2 = 50
    ref_id_3 = 36

    # load data
    # dir_1 = BASE_DIR + '/data/data_1.npz'
    # dir_2 = BASE_DIR + '/data/data_2.npz'
    dir_2 = BASE_DIR + '/data/data_bottle_1.npz'
    dir_1 = BASE_DIR + '/data/data_bottle_2.npz'
    data_1 = np.load(dir_1, allow_pickle=True)
    data_2 = np.load(dir_2, allow_pickle=True)
    pcd1 = data_1["pcd"]
    grasps1 = data_1["grasps"]
    pcd2 = data_2["pcd"]
    grasps2 = data_2["grasps"]

    n_grasps = grasps2.shape[0]
    # # show point clouds
    # tpcd1 = trimesh.PointCloud(pcd1)
    # tpcd2 = trimesh.PointCloud(pcd2)
    # tpcd1.show()
    # tpcd2.show()

    # preprocess the point cloud and randomly sample 2000 points
    # first filter out outliers
    pcd_1_mean = np.mean(pcd1, axis=0)
    index_1 = np.where(np.linalg.norm(pcd1 - pcd_1_mean, 2, 1) < 0.22)[0]
    pcd1 = pcd1[index_1]
    index_1 = np.where(pcd1[:, -1] >= np.min(pcd1[:, -1]) + 0.00004)
    pcd1 = pcd1[index_1]
    pcd_2_mean = np.mean(pcd1, axis=0)
    index_2 = np.where(np.linalg.norm(pcd2 - pcd_2_mean, 2, 1) < 0.22)[0]
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
    tpcd1 = trimesh.PointCloud(pcd1)
    tpcd2 = trimesh.PointCloud(pcd2)
    tpcd1.show()
    tpcd2.show()

    # load model
    model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_demo_mug_weights.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'multi_category_weights.pth')
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
    ref_query_pts_1 = grasps1[ref_id_1] @ hom_query_pts
    ref_query_pts_1 = ref_query_pts_1[:3, :] - mean_1[:, None]
    ref_query_pts_1 = ref_query_pts_1.T

    ref_query_pts_2 = grasps1[ref_id_2] @ hom_query_pts
    ref_query_pts_2 = ref_query_pts_2[:3, :] - mean_1[:, None]
    ref_query_pts_2 = ref_query_pts_2.T

    query_pts_vis_1 = copy.deepcopy(ref_query_pts_1)
    query_pts_vis_2 = copy.deepcopy(ref_query_pts_2)

    # ref
    reference_model_input_1 = {}
    ref_query_pts = torch.from_numpy(ref_query_pts_1).float().to(dev)
    ref_shape_pcd = torch.from_numpy(pcd1[:2000]).float().to(dev)
    reference_model_input_1['coords'] = ref_query_pts[None, :, :]
    reference_model_input_1['point_cloud'] = ref_shape_pcd[None, :, :]

    reference_model_input_2 = {}
    ref_query_pts = torch.from_numpy(ref_query_pts_2).float().to(dev)
    ref_shape_pcd = torch.from_numpy(pcd1[:2000]).float().to(dev)
    reference_model_input_2['coords'] = ref_query_pts[None, :, :]
    reference_model_input_2['point_cloud'] = ref_shape_pcd[None, :, :]

    # get the descriptors for these reference query points
    reference_latent_1 = model.extract_latent(reference_model_input_1).detach()
    reference_act_hat_1 = model.forward_latent(reference_latent_1, reference_model_input_1['coords']).detach()

    reference_latent_2 = model.extract_latent(reference_model_input_2).detach()
    reference_act_hat_2 = model.forward_latent(reference_latent_2, reference_model_input_2['coords']).detach()

    # get the descriptors for all other grasps
    # n_grasp = grasps_2.shape[0]
    n_batch = 30
    index = 0
    k = n_grasps // 30
    n_grasp_all = 30 * k
    grasp_all = grasps2[:n_grasp_all]
    loss_all_1 = []
    loss_all_2 = []
    for i in range(k):
        opt_query_pts = torch.from_numpy(ref_pts_gripper).float().to(dev)
        opt_query_pts = opt_query_pts[None, :, :].repeat((n_batch, 1, 1))
        trans = torch.from_numpy(grasps2[index:index+30]).squeeze().float().to(dev)
        X = torch_util.transform_pcd_torch(opt_query_pts, trans)
        mu_2 = torch.from_numpy(mean_2)[None, None, :].float().to(dev)
        X = X - mu_2
        opt_model_input = {}
        opt_model_input['coords'] = X

        mi_point_cloud = []
        for ii in range(n_batch):
            mi_point_cloud.append(torch.from_numpy(pcd2[:2000]).float().to(dev))
        mi_point_cloud = torch.stack(mi_point_cloud, 0)
        opt_model_input['point_cloud'] = mi_point_cloud
        opt_latent = model.extract_latent(opt_model_input).detach()
        act_hat = model.forward_latent(opt_latent, X)

        t_size = reference_act_hat_1.size()
        losses_1 = [loss_fn(act_hat[ii].view(t_size), reference_act_hat_1) for ii in range(n_batch)]
        losses_str = ['%f' % val.item() for val in losses_1]
        loss_str = ', '.join(losses_str)
        print(f'i: {0}, losses: {loss_str}')
        loss_np_1 = [losses_1[ii].detach().cpu().numpy() for ii in range(n_batch)]
        loss_all_1 += loss_np_1

        t_size = reference_act_hat_2.size()
        losses_2 = [loss_fn(act_hat[ii].view(t_size), reference_act_hat_2) for ii in range(n_batch)]
        losses_str = ['%f' % val.item() for val in losses_2]
        loss_str = ', '.join(losses_str)
        print(f'i: {0}, losses: {loss_str}')
        loss_np_2 = [losses_2[ii].detach().cpu().numpy() for ii in range(n_batch)]
        loss_all_2 += loss_np_2

        index += n_batch

    probs_1 = np.asarray(loss_all_1)
    probs_2 = np.asarray(loss_all_2)
    probs = np.vstack([probs_1, probs_2])
    label = np.argmin(probs, axis=0)
    grasp_task_1 = grasp_all[label == 0]
    grasp_task_2 = grasp_all[label == 1]
    # visualization
    bias = np.array([[0.4, 0, 0]])
    coords = np.array([[0.1, 0, 0, 0],
                       [0., 0.1, 0, 0],
                       [0, 0, 0.1, 0],
                       [1, 1, 1, 1]])

    coords_1 = grasps1[ref_id_1] @ coords
    coords_1 = coords_1[0:3, :] - mean_1[:, None]
    coords_1 = coords_1.T

    coords_2 = grasps1[ref_id_2] @ coords
    coords_2 = coords_2[0:3, :] - mean_1[:, None]
    coords_2 = coords_2.T

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

    all_nodes_1 = []
    all_nodes_2 = []
    all_edges = []
    index = 0
    n_pts = 7
    # show grasp (gripper)
    for i in range(n_grasp_all):
        coords = np.concatenate((gripper_control_points, np.ones((7, 1))), axis=1)
        coords = grasps2[i] @ coords.T
        coords = coords[0:3, :]
        coords = coords.T
        nodes_1 = coords + bias
        nodes_2 = coords + bias * 2
        all_nodes_1.append(nodes_1)
        all_nodes_2.append(nodes_2)
        all_edges.append(np.array([[index, index + 1],
                                   [index + 2, index + 5],
                                   [index + 2, index + 3],
                                   [index + 5, index + 6]]))

        index += n_pts
    all_nodes_1 = np.vstack(all_nodes_1)
    all_nodes_2 = np.vstack(all_nodes_2)
    all_edges = np.vstack(all_edges)

    probs_1_ = np.repeat(probs_1, 4)
    probs_2_ = np.repeat(probs_2, 4)
    ps.init()
    ps.set_up_dir("z_up")
    ps.register_point_cloud("query_1", query_pts_vis_1 + mean_1, radius=0.005, enabled=True)
    ps.register_point_cloud("query_2", query_pts_vis_2 + mean_1, radius=0.005, enabled=True)
    ps.register_point_cloud("pcd_1", pcd1[:2000] + mean_1, radius=0.005, enabled=True)
    ps.register_point_cloud("pcd_2", pcd2[:2000] + mean_2 + bias, radius=0.005, enabled=True)
    ps.register_point_cloud("pcd_3", pcd2[:2000] + mean_2 + bias * 2, radius=0.005, enabled=True)
    ps3 = ps.register_curve_network("gripper_1", all_nodes_1, all_edges, radius=0.002, enabled=True)
    ps3.add_scalar_quantity("probs", probs_1_, defined_on='edges', cmap='coolwarm', enabled=True)
    ps4 = ps.register_curve_network("gripper_2", all_nodes_2, all_edges, radius=0.002, enabled=True)
    ps4.add_scalar_quantity("probs", probs_2_, defined_on='edges', cmap='coolwarm', enabled=True)
    ps.show()
