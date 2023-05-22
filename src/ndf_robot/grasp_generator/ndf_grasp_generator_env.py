import os, os.path as osp
import sys

import polyscope as ps
import torch
import numpy as np
import trimesh
import random
import copy

from ndf_robot.utils import torch_util
from task_oriented_grasping.utils.path_utils import set_ndfs_grasps_save_dir

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))


class NDFGraspGeneratorSide:
    def __init__(self, model, pcd1, pcd2, pcd3, pcd4, ref_grasp, obj_mat, model_type='pointnet', opt_iterations=500):
        self.hom_query_pts = None
        self.ref_pts_gripper = None
        self.query_pts_vis = None
        self.ref_query_pts = None
        self.grasp_list = []
        self.model = model
        self.model_type = model_type
        self.opt_iterations = opt_iterations
        self.ref_grasp = ref_grasp
        self.obj_mat = obj_mat

        self.n_samples = 200
        self.m = 4
        self.n_pts = 1500
        self.n_opt_pts = 200
        self.thresh = 0.3
        self.z = 0.1

        # pcd sampled
        dir_mesh = BASE_DIR + '/data/scaled_2.5_2.5_4.obj'
        mesh1 = trimesh.load(dir_mesh, process=False)
        pcd_sampled = mesh1.sample(5000)
        self.pcd_mean_sampled = np.mean(pcd_sampled, axis=0)
        self.pcd_sampled = pcd_sampled - self.pcd_mean_sampled

        pcd_merged = np.vstack([pcd1, pcd2, pcd3, pcd4])
        self.pcd_mean_merged = np.mean(pcd_merged, axis=0)
        self.pcd_merged = pcd_merged - self.pcd_mean_merged

        self.mean1 = np.mean(pcd1, axis=0)
        self.mean2 = np.mean(pcd2, axis=0)
        self.mean3 = np.mean(pcd3, axis=0)
        self.mean4 = np.mean(pcd4, axis=0)
        self.pcd1 = pcd1 - self.mean1
        self.pcd2 = pcd2 - self.mean2
        self.pcd3 = pcd3 - self.mean3
        self.pcd4 = pcd4 - self.mean4
        np.random.shuffle(self.pcd1)
        np.random.shuffle(self.pcd2)
        np.random.shuffle(self.pcd3)
        np.random.shuffle(self.pcd4)

        self.mean_sampled_world = self.mean1 - np.array([0.002, -0.01, 0.04])

        self.pcd_ref = None
        self.pcd1_c = None
        self.pcd2_c = None
        self.pcd3_c = None
        self.pcd4_c = None

        self.grasp_list_1 = []
        self.grasp_list_2 = []
        self.grasp_list_3 = []
        self.grasp_list_4 = []

        self.loss_fn = torch.nn.L1Loss()
        if torch.cuda.is_available():
            self.dev = torch.device('cuda:0')
        else:
            self.dev = torch.device('cpu')
        self.model = self.model.to(self.dev)
        self.model.eval()

        self.sample_gripper_pts()
        # self.shape_completion_ref()

    def shape_completion_ref(self):
        shape = {}
        shape['point_cloud'] = torch.from_numpy(self.pcd1[:self.n_pts]).float().to(self.dev)[None, :, :]
        shape_pcd = trimesh.PointCloud(self.pcd1)
        bb = shape_pcd.bounding_box

        eval_pts = bb.sample_volume(20000)
        shape['coords'] = torch.from_numpy(eval_pts)[None, :, :].float().to(self.dev).detach()
        out = self.model(shape)
        in_inds = torch.where(out['occ'].squeeze() > self.thresh)[0].cpu().numpy()
        in_pts = eval_pts[in_inds]

        pcd = np.vstack([self.pcd1, in_pts[:1000]])

        np.random.shuffle(pcd)
        self.pcd_ref = pcd

    def shape_completion(self):
        # 1
        shape = {}
        shape['point_cloud'] = torch.from_numpy(self.pcd1[:self.n_pts]).float().to(self.dev)[None, :, :]
        shape_pcd = trimesh.PointCloud(self.pcd1)
        bb = shape_pcd.bounding_box

        eval_pts = bb.sample_volume(20000)
        shape['coords'] = torch.from_numpy(eval_pts)[None, :, :].float().to(self.dev).detach()
        out = self.model(shape)
        in_inds = torch.where(out['occ'].squeeze() > self.thresh)[0].cpu().numpy()
        in_pts = eval_pts[in_inds]

        pcd = np.vstack([self.pcd1, in_pts[:1000]])
        np.random.shuffle(pcd)
        self.pcd1_c = pcd

        # 2
        shape = {}
        shape['point_cloud'] = torch.from_numpy(self.pcd2[:self.n_pts]).float().to(self.dev)[None, :, :]
        shape_pcd = trimesh.PointCloud(self.pcd2)
        bb = shape_pcd.bounding_box

        eval_pts = bb.sample_volume(20000)
        shape['coords'] = torch.from_numpy(eval_pts)[None, :, :].float().to(self.dev).detach()
        out = self.model(shape)
        in_inds = torch.where(out['occ'].squeeze() > self.thresh)[0].cpu().numpy()
        in_pts = eval_pts[in_inds]

        pcd = np.vstack([self.pcd2, in_pts[:1000]])
        np.random.shuffle(pcd)
        self.pcd2_c = pcd

        # 3
        shape = {}
        shape['point_cloud'] = torch.from_numpy(self.pcd3[:self.n_pts]).float().to(self.dev)[None, :, :]
        shape_pcd = trimesh.PointCloud(self.pcd3)
        bb = shape_pcd.bounding_box

        eval_pts = bb.sample_volume(20000)
        shape['coords'] = torch.from_numpy(eval_pts)[None, :, :].float().to(self.dev).detach()
        out = self.model(shape)
        in_inds = torch.where(out['occ'].squeeze() > self.thresh)[0].cpu().numpy()
        in_pts = eval_pts[in_inds]

        pcd = np.vstack([self.pcd3, in_pts[:1000]])
        np.random.shuffle(pcd)
        self.pcd3_c = pcd

        # 4
        shape = {}
        shape['point_cloud'] = torch.from_numpy(self.pcd4[:self.n_pts]).float().to(self.dev)[None, :, :]
        shape_pcd = trimesh.PointCloud(self.pcd4)
        bb = shape_pcd.bounding_box

        eval_pts = bb.sample_volume(20000)
        shape['coords'] = torch.from_numpy(eval_pts)[None, :, :].float().to(self.dev).detach()
        out = self.model(shape)
        in_inds = torch.where(out['occ'].squeeze() > self.thresh)[0].cpu().numpy()
        in_pts = eval_pts[in_inds]

        pcd = np.vstack([self.pcd4, in_pts[:1000]])
        np.random.shuffle(pcd)
        self.pcd4_c = pcd

    def sample_gripper_pts(self):
        # sample query points
        n = self.n_opt_pts
        query_x = np.random.uniform(-0.02, 0.02, n)
        query_y = np.random.uniform(-0.04, 0.04, n)
        query_z = np.random.uniform(-0.05 + self.z, 0.02 + self.z, n)
        # query_z = np.random.uniform(-0.05 + 0.1, 0.01 + 0.1, n)
        ones = np.ones(n)
        ref_pts_gripper = np.vstack([query_x, query_y, query_z])
        ref_pts_gripper = ref_pts_gripper.T
        hom_query_pts = np.vstack([query_x, query_y, query_z, ones])

        # transform
        ref_query_pts = self.ref_grasp @ hom_query_pts
        ref_query_pts = ref_query_pts[:3, :] - self.mean1[:, None]
        ref_query_pts = ref_query_pts.T

        self.hom_query_pts = hom_query_pts
        self.ref_pts_gripper = ref_pts_gripper
        self.ref_query_pts = ref_query_pts
        self.query_pts_vis = copy.deepcopy(ref_query_pts)

    def sample_grasps(self):
        ref_query_pts = self.ref_query_pts
        query_pts_vis = self.query_pts_vis
        ref_pts_gripper = self.ref_pts_gripper

        # start
        reference_model_input = {}
        ref_query_pts = torch.from_numpy(ref_query_pts).float().to(self.dev)
        # ----------------------------- pcd_ref -------------------------------------- #
        ref_shape_pcd = torch.from_numpy(self.pcd1[:self.n_pts]).float().to(self.dev)
        # ref_shape_pcd = torch.from_numpy(self.pcd_ref[:self.n_pts]).float().to(self.dev)
        reference_model_input['coords'] = ref_query_pts[None, :, :]
        reference_model_input['point_cloud'] = ref_shape_pcd[None, :, :]

        # get the descriptors for these reference query points
        reference_latent = self.model.extract_latent(reference_model_input).detach()
        reference_act_hat = self.model.forward_latent(reference_latent, reference_model_input['coords']).detach()

        # set up the optimization
        M = self.m

        # parameters for optimization
        trans = (torch.rand((M, 3)) * 0.1).float().to(self.dev)
        rot = torch.rand(M, 3).float().to(self.dev)
        trans.requires_grad_()
        rot.requires_grad_()
        opt = torch.optim.Adam([trans, rot], lr=1e-2)

        for itr in range(self.n_samples):
            print("itr: " + str(itr))
            # initialization
            rand_rot_init = (torch.rand((M, 3)) * 2 * np.pi).float().to(self.dev)
            rand_mat_init = torch_util.angle_axis_to_rotation_matrix(rand_rot_init)
            rand_mat_init = rand_mat_init.squeeze().float().to(self.dev)

            # now randomly initialize a copy of the query points
            opt_query_pts = torch.from_numpy(ref_pts_gripper).float().to(self.dev)
            opt_query_pts = opt_query_pts[None, :, :].repeat((M, 1, 1))
            X = torch_util.transform_pcd_torch(opt_query_pts, rand_mat_init)

            opt_model_input = {}
            opt_model_input['coords'] = X

            mi_point_cloud = []
            np.random.shuffle(self.pcd_merged)
            # ----------------------------- pcds -------------------------------------- #
            for ii in range(M):
                # mi_point_cloud.append(torch.from_numpy(self.pcd2[:self.n_pts]).float().to(self.dev))
                # mi_point_cloud.append(torch.from_numpy(self.pcd1[:self.n_pts]).float().to(self.dev))
                mi_point_cloud.append(torch.from_numpy(self.pcd_merged[:self.n_pts]).float().to(self.dev))

            mi_point_cloud = torch.stack(mi_point_cloud, 0)
            opt_model_input['point_cloud'] = mi_point_cloud
            opt_latent = self.model.extract_latent(opt_model_input).detach()

            loss_values = []

            # run optimization
            for i in range(self.opt_iterations):
                T_mat = torch_util.angle_axis_to_rotation_matrix(rot).squeeze()
                # noise_vec = (torch.randn(X.size()) * (self.perturb_scale / ((i + 1) ** (self.perturb_decay)))).to(self.dev)
                # X_perturbed = X + noise_vec
                # ----------------------------- trans -------------------------------------- #
                # trans_ = trans - (torch.from_numpy(self.mean1[None, :])).float().to(self.dev)
                trans_ = trans - (torch.from_numpy(self.pcd_mean_merged[None, :])).float().to(self.dev)

                # X_new = torch_util.transform_pcd_torch(X_perturbed, T_mat) + trans[:, None, :].repeat((1, X.size(1), 1))
                X_new = torch_util.transform_pcd_torch(X, T_mat) + trans_[:, None, :].repeat((1, X.size(1), 1))

                act_hat = self.model.forward_latent(opt_latent, X_new)
                t_size = reference_act_hat.size()

                losses = [self.loss_fn(act_hat[ii].view(t_size), reference_act_hat) for ii in range(M)]
                loss = torch.mean(torch.stack(losses))
                if i % 100 == 0:
                    losses_str = ['%f' % val.item() for val in losses]
                    loss_str = ', '.join(losses_str)
                    print(f'i: {i}, losses: {loss_str}')
                loss_values.append(loss.item())
                opt.zero_grad()
                loss.backward()
                opt.step()

            # save the grasps
            best_idx = torch.argmin(torch.stack(losses)).item()
            grasp_obj = T_mat[best_idx].detach().cpu().numpy()
            grasp_trans = trans_[best_idx].detach().cpu().numpy()

            rand_mat_init_ = rand_mat_init[best_idx].detach().cpu().numpy()

            # ----------------------------- trans_back -------------------------------------- #
            # grasp_obj[:3, -1] = grasp_trans + self.mean1
            grasp_obj[:3, -1] = grasp_trans + self.pcd_mean_merged
            self.grasp_list.append(grasp_obj @ rand_mat_init_)

        # visualization
        vis_grasp_list = [(self.grasp_list[i] @ self.hom_query_pts).T[:, 0:3] for i in range(self.n_samples)]

        gripper_control_points_armar = np.array([[0.00000, 0.00000, 0.00000],
                                                 [0.00000, 0.00000, 0.06000],
                                                 [-0.0500, 0.00000, 0.11000],
                                                 [0.07000, 0.00000, 0.13000]])

        ps.init()
        ps.set_up_dir("z_up")
        for i in range(self.n_samples):
            ps.register_point_cloud("g_" + str(i), vis_grasp_list[i], radius=0.004, enabled=True)

            coords = np.concatenate((gripper_control_points_armar, np.ones((4, 1))), axis=1)
            coords = self.grasp_list[i] @ coords.T
            coords = coords[0:3, :]
            coords = coords.T
            nodes_armar = coords
            ps.register_curve_network("edge_1" + str(i), nodes_armar[[0, 1]], np.array([[0, 1]]),
                                      enabled=True, radius=0.0015, color=(0, 0, 1))
            ps.register_curve_network("edge_2" + str(i), nodes_armar[[1, 2]], np.array([[0, 1]]),
                                      enabled=True, radius=0.0015, color=(1, 0, 0))
            ps.register_curve_network("edge_3" + str(i), nodes_armar[[1, 3]], np.array([[0, 1]]),
                                      enabled=True, radius=0.0015, color=(0, 1, 0))

        ps.register_point_cloud("query", self.query_pts_vis + self.mean1 - np.array([0.3, 0, 0]), radius=0.01, color=[1, 0, 0], enabled=True)
        ps.register_point_cloud("obj1", self.pcd1[:self.n_pts] + self.mean1 - np.array([0.3, 0, 0]), radius=0.005, color=[0.5, 0.5, 0], enabled=True)
        ps.register_point_cloud("obj2", self.pcd_merged[:self.n_pts] + self.pcd_mean_merged, radius=0.005, enabled=True)

        ps.show()

    def save_grasps(self, filename='0'):
        save_dir = set_ndfs_grasps_save_dir()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        grasps_obj = [np.linalg.inv(self.obj_mat) @ self.grasp_list[i] for i in range(self.n_samples)]
        np.savez(save_dir + '/' + filename, grasps_obj=grasps_obj)

        print("grasps saved.")
