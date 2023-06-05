import os
import sys
import os.path as osp
import time

import torch
import numpy as np
import trimesh
import copy
import polyscope as ps
import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network

from ndf_robot.utils import path_util, torch_util
from task_oriented_grasping.utils.path_utils import get_scene_path

pcd_dir = get_scene_path(file_name="0/0.npz")
data = np.load(pcd_dir, allow_pickle=True)
pcd1 = data["pcd_local"]
pcd2 = data["pcd_local2"]
pcd3 = data["pcd_local3"]
pcd4 = data["pcd_local4"]
obj_pos = data["obj_pos"]
obj_quat = data["obj_quat"]

n_pts = 2000

pcd2_mean = np.mean(pcd2, axis=0)
pcd2 = pcd2 - pcd2_mean
# mat_t = np.array([[1, 0, 0, 0],
#                   [0, 0, -1, 0],
#                   [0, 1, 0, 0],
#                   [0, 0, 0, 1]])
mat_t = np.array([[1, 0, 0, 0],
                  [0, -1, 0, 0],
                  [0, 0, -1, 0],
                  [0, 0, 0, 1]])
n = pcd2.shape[0]
ones = np.ones((n, 1))
hom_pcd2 = np.hstack([pcd2, ones])
pcd_trans = mat_t @ hom_pcd2.T
pcd2 = pcd_trans[0:3, :].T

pcd1_mean = np.mean(pcd1, axis=0)
pcd1 = pcd1 - pcd1_mean

opt_iterations = 500
# for visualization
n_pts_gripper = 500
radius = 0.07
u_th = np.random.rand(n_pts_gripper, 1)
u_r = np.random.rand(n_pts_gripper, 1)
x = radius * np.sqrt(u_r) * np.cos(2 * np.pi * u_th)
y = radius * np.sqrt(u_r) * np.sin(2 * np.pi * u_th) + 1.1
z = np.random.rand(n_pts_gripper, 1) * 0.05 + 0.725
# z = np.ones((n_pts, 1)) * 0.76
ref_query_pts = np.hstack([x, y, z])
ref_query_pts = ref_query_pts - pcd1_mean
ref_pts_gripper = ref_query_pts

ones = np.ones((n_pts_gripper, 1))
ref_query_pts = np.hstack([ref_query_pts, ones])
trans_ref = mat_t @ ref_query_pts.T
ref_query_pts = trans_ref[:3, :].T
query_pts_vis = copy.deepcopy(ref_query_pts)

# load model
model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_demo_mug_weights.pth')
model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=True,
                                        sigmoid=True).cuda()
model.load_state_dict(torch.load(model_path))

# device
dev = torch.device('cuda:0')

# loss
loss_fn = torch.nn.L1Loss()

np.random.shuffle(pcd1)
np.random.shuffle(pcd2)

# # shape completion
# thresh = 0.2
# shape = {}
# shape['point_cloud'] = torch.from_numpy(pcd1[:n_pts]).float().to(dev)[None, :, :]
# shape_pcd = trimesh.PointCloud(pcd1)
# bb = shape_pcd.bounding_box
#
# eval_pts = bb.sample_volume(20000)
# shape['coords'] = torch.from_numpy(eval_pts)[None, :, :].float().to(dev).detach()
# out = model(shape)
# in_inds = torch.where(out['occ'].squeeze() > thresh)[0].cpu().numpy()
# in_pts = eval_pts[in_inds]
#
# pcd = np.vstack([pcd1, in_pts[:1000]])
# np.random.shuffle(pcd)
# pcd1 = pcd

# start optimization
# get descriptor of the ref grasp of the ref point cloud
reference_model_input = {}
ref_query_pts = torch.from_numpy(ref_query_pts).float().to(dev)
ref_shape_pcd = torch.from_numpy(pcd2[:n_pts]).float().to(dev)
reference_model_input['coords'] = ref_query_pts[None, :, :]
reference_model_input['point_cloud'] = ref_shape_pcd[None, :, :]  # get the descriptors for these reference query points
reference_latent = model.extract_latent(reference_model_input).detach()
reference_act_hat = model.forward_latent(reference_latent, reference_model_input['coords']).detach()

best_loss = np.inf
best_tf = np.eye(4)
best_idx = 0
tf_list = []
M = 10

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
    # if ii % 5 == 0:
    #     idx = torch.randperm(pcd1.shape[0])
    #     pcd1 = pcd1[idx]
    mi_point_cloud.append(torch.from_numpy(pcd1[:n_pts]).float().to(dev))
mi_point_cloud = torch.stack(mi_point_cloud, 0)
opt_model_input['point_cloud'] = mi_point_cloud
opt_latent = model.extract_latent(opt_model_input).detach()

loss_values = []
vid_plot_idx = None

for i in range(opt_iterations):
    T_mat = torch_util.angle_axis_to_rotation_matrix(rot).squeeze()
    trans_ = trans - (torch.from_numpy(pcd1_mean[None, :])).float().to(dev)
    # X_new = torch_util.transform_pcd_torch(X_perturbed, T_mat) + trans[:, None, :].repeat((1, X.size(1), 1))
    X_new = torch_util.transform_pcd_torch(X, T_mat) + trans_[:, None, :].repeat((1, X.size(1), 1))

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

ps.init()
ps.set_up_dir("z_up")
ps.register_point_cloud("pcd_1", pcd1[:2000], radius=0.005, enabled=True)
ps.register_point_cloud("pcd_2", pcd2[:2000], radius=0.005, enabled=True)
ps.register_point_cloud("ref", query_pts_vis, radius=0.005, enabled=True)
ps.register_point_cloud("opt_g", best_X, radius=0.005, enabled=True)
ps.show()
