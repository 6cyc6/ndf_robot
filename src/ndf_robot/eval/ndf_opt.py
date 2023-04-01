import os
import sys
import os.path as osp
import time

import torch
import numpy as np
import trimesh
import random
import argparse
import copy
import polyscope as ps
from scipy.spatial.transform import Rotation

from ndf_robot.eval.ndf_paritial_map import NDFPartialMap
from ndf_robot.utils import path_util, torch_util
import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))

start = time.time()
# set some parameters
id_grasp = 7

n = 500  # num query points
n_pts = 2000  # num points of point cloud

recon = True
thresh = 0.15
n_recon = 1000

full_opt = 10
opt_iterations = 500

perturb_scale = 0.0001
perturb_decay = 0.5

# load data (ref point cloud + grasp / opt point cloud)
dir_1 = BASE_DIR + '/data/data2.npz'
dir_2 = BASE_DIR + '/data/data1.npz'
data_1 = np.load(dir_1, allow_pickle=True)
data_2 = np.load(dir_2, allow_pickle=True)
pcd1 = data_1["pcd"]
grasps1 = data_1["grasps"]
pcd2 = data_2["pcd"]
grasps2 = data_2["grasps"]

# preprocess the point cloud
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

# center the object
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

# sample query points
query_x = np.random.uniform(-0.02, 0.02, n)
query_y = np.random.uniform(-0.04, 0.04, n)
query_z = np.random.uniform(-0.05 + 0.1, 0.02 + 0.1, n)

ones = np.ones(n)
ref_pts_gripper = np.vstack([query_x, query_y, query_z])
ref_pts_gripper = ref_pts_gripper.T
hom_query_pts = np.vstack([query_x, query_y, query_z, ones])

# transform
ref_query_pts = grasps1[id_grasp] @ hom_query_pts
ref_query_pts = ref_query_pts[:3, :] - mean_1[:, None]
ref_query_pts = ref_query_pts.T

query_pts_vis = copy.deepcopy(ref_query_pts)

# shape completion
shape_1 = {}
shape_1['point_cloud'] = torch.from_numpy(pcd1[:n_pts]).float().to(dev)[None, :, :]
shape_pcd = trimesh.PointCloud(pcd1)
bb = shape_pcd.bounding_box
bb_scene = trimesh.Scene();
bb_scene.add_geometry([shape_pcd, bb])
# bb_scene.show()

eval_pts = bb.sample_volume(20000)
shape_1['coords'] = torch.from_numpy(eval_pts)[None, :, :].float().to(dev).detach()
out = model(shape_1)

in_inds = torch.where(out['occ'].squeeze() > thresh)[0].cpu().numpy()
in_pts_1 = eval_pts[in_inds]
np.random.shuffle(in_pts_1)

if recon:
    pcd1 = np.vstack([pcd1, in_pts_1[:n_recon]])
    np.random.shuffle(pcd1)

shape_2 = {}
shape_2['point_cloud'] = torch.from_numpy(pcd2[:n_pts]).float().to(dev)[None, :, :]
shape_pcd = trimesh.PointCloud(pcd2)
bb = shape_pcd.bounding_box
bb_scene = trimesh.Scene();
bb_scene.add_geometry([shape_pcd, bb])
# bb_scene.show()

eval_pts = bb.sample_volume(20000)
shape_2['coords'] = torch.from_numpy(eval_pts)[None, :, :].float().to(dev).detach()
out = model(shape_2)
in_inds = torch.where(out['occ'].squeeze() > thresh)[0].cpu().numpy()
in_pts_2 = eval_pts[in_inds]
np.random.shuffle(in_pts_2)

if recon:
    pcd2 = np.vstack([pcd2, in_pts_2[:n_recon]])
    np.random.shuffle(pcd2)

# start optimization
# get descriptor of the ref grasp of the ref point cloud
reference_model_input = {}
ref_query_pts = torch.from_numpy(ref_query_pts).float().to(dev)
ref_shape_pcd = torch.from_numpy(pcd1[:n_pts]).float().to(dev)
reference_model_input['coords'] = ref_query_pts[None, :, :]
reference_model_input['point_cloud'] = ref_shape_pcd[None, :, :]

# get the descriptors for these reference query points
reference_latent = model.extract_latent(reference_model_input).detach()
reference_act_hat = model.forward_latent(reference_latent, reference_model_input['coords']).detach()

best_loss = np.inf
best_tf = np.eye(4)
best_idx = 0
tf_list = []
M = full_opt

# parameters for optimization
trans = (torch.rand((M, 3)) * 0.1).float().to(dev)
rot = torch.rand(M, 3).float().to(dev)
trans.requires_grad_()
rot.requires_grad_()
opt = torch.optim.Adam([trans, rot], lr=1e-2)

# initialization
rand_rot_init = (torch.rand((M, 3)) * 2 * np.pi).float().to(dev)
rand_mat_init = torch_util.angle_axis_to_rotation_matrix(rand_rot_init)
rand_mat_init = rand_mat_init.squeeze().float().to(dev)

# now randomly initialize a copy of the query points
opt_query_pts = torch.from_numpy(ref_pts_gripper).float().to(dev)
opt_query_pts = opt_query_pts[None, :, :].repeat((M, 1, 1))
X = torch_util.transform_pcd_torch(opt_query_pts, rand_mat_init)

opt_model_input = {}
opt_model_input['coords'] = X

mi_point_cloud = []
for ii in range(M):
    # mi_point_cloud.append(torch.from_numpy(self.pcd2[:self.n_pts]).float().to(self.dev))
    mi_point_cloud.append(torch.from_numpy(pcd2[:n_pts]).float().to(dev))
mi_point_cloud = torch.stack(mi_point_cloud, 0)
opt_model_input['point_cloud'] = mi_point_cloud
opt_latent = model.extract_latent(opt_model_input).detach()

loss_values = []
vid_plot_idx = None

# run optimization
pcd_traj_list = {}
for jj in range(M):
    pcd_traj_list[jj] = []
    pcd_traj_list[jj].append(np.mean(X[jj].detach().cpu().numpy(), axis=0))

for i in range(opt_iterations):
    T_mat = torch_util.angle_axis_to_rotation_matrix(rot).squeeze()
    noise_vec = (torch.randn(X.size()) * (perturb_scale / ((i + 1) ** (perturb_decay)))).to(dev)
    X_perturbed = X + noise_vec
    trans_ = trans - (torch.from_numpy(mean_2[None, :])).float().to(dev)
    # X_new = torch_util.transform_pcd_torch(X_perturbed, T_mat) + trans[:, None, :].repeat((1, X.size(1), 1))
    X_new = torch_util.transform_pcd_torch(X_perturbed, T_mat) + trans_[:, None, :].repeat((1, X.size(1), 1))

    act_hat = model.forward_latent(opt_latent, X_new)
    t_size = reference_act_hat.size()

    losses = [loss_fn(act_hat[ii].view(t_size), reference_act_hat) for ii in range(M)]
    loss = torch.mean(torch.stack(losses))
    if i % 100 == 0:
        losses_str = ['%f' % val.item() for val in losses]
        loss_str = ', '.join(losses_str)
        print(f'i: {i}, losses: {loss_str}')
    loss_values.append(loss.item())
    opt.zero_grad()
    loss.backward()
    opt.step()

best_idx = torch.argmin(torch.stack(losses)).item()
best_loss = losses[best_idx]
print('best loss: %f, best_idx: %d' % (best_loss, best_idx))

best_X = X_new[best_idx].detach().cpu().numpy()
grasp_obj = T_mat[best_idx].detach().cpu().numpy()
grasp_trans = trans_[best_idx].detach().cpu().numpy()

print(time.time() - start)

ps.init()
ps.set_up_dir("z_up")
ps.register_point_cloud("pcd_1", pcd1[:2000] + mean_1, radius=0.005, enabled=True)
ps.register_point_cloud("pcd_2", pcd2[:2000] + mean_2, radius=0.005, enabled=True)
# ps.register_point_cloud("inlier_1", in_pts_1 + mean_1, radius=0.005, enabled=True)
# ps.register_point_cloud("inlier_2", in_pts_2 + mean_2, radius=0.005, enabled=True)
ps.register_point_cloud("grasp_1", query_pts_vis + mean_1, radius=0.005, enabled=True)
ps.register_point_cloud("opt_g", best_X + mean_2, radius=0.005, enabled=True)
ps.show()
