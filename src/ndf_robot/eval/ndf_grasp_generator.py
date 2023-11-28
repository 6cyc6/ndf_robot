import os, os.path as osp
import sys

import polyscope as ps
import torch
import numpy as np
import trimesh
import random
import copy

from ndf_robot.utils import torch_util

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))


class NDFGraspGenerator:
    def __init__(self, model, pcd1, pcd2, pcd3, pcd4, ref_grasp, model_type='pointnet', opt_iterations=500):
        self.grasp_list = None
        self.model = model
        self.model_type = model_type
        self.opt_iterations = opt_iterations
        self.ref_grasp = ref_grasp

        self.n_samples = 50
        self.m = 4
        self.n_pts = 2000
        self.n_opt_pts = 500
        self.thresh = 0.3
        self.z = 0.12

        self.mean1 = np.mean(pcd1, axis=0)
        self.mean2 = np.mean(pcd2, axis=0)
        self.mean3 = np.mean(pcd3, axis=0)
        self.mean4 = np.mean(pcd4, axis=0)
        # self.mean2 = np.mean(pcd2, axis=0) + np.array([0, -0.01, 0.04])
        self.pcd1 = pcd1 - self.mean1
        self.pcd2 = pcd2 - self.mean2
        self.pcd3 = pcd3 - self.mean3
        self.pcd4 = pcd4 - self.mean4
        np.random.shuffle(self.pcd1)
        np.random.shuffle(self.pcd2)
        np.random.shuffle(self.pcd3)
        np.random.shuffle(self.pcd4)

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

        self.shape_completion_ref()

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

        # ps.init()
        # ps.set_up_dir("z_up")
        # ps.register_point_cloud("pcd2", self.pcd2, radius=0.005, enabled=True)
        # ps.register_point_cloud("pts", in_pts, radius=0.005, enabled=True)
        # ps.register_point_cloud("pcd_new", pcd3, radius=0.005, enabled=True)
        # ps.show()

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

        # ps.init()
        # ps.set_up_dir("z_up")
        # ps.register_point_cloud("pcd2", self.pcd2, radius=0.005, enabled=True)
        # ps.register_point_cloud("pts", in_pts, radius=0.005, enabled=True)
        # ps.register_point_cloud("pcd_new", pcd3, radius=0.005, enabled=True)
        # ps.show()

    def save_grasps(self):
        dirs_scene = BASE_DIR + '/result/'
        if not os.path.exists(dirs_scene):
            os.makedirs(dirs_scene)

        np.savez(dirs_scene + 'ndf_grasps', grasps=self.grasp_list)

        print("grasps saved.")

    def sample_pts(self):
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

        query_pts_vis = copy.deepcopy(ref_query_pts)

        # start
        reference_model_input = {}
        ref_query_pts = torch.from_numpy(ref_query_pts).float().to(self.dev)
        # ref_shape_pcd = torch.from_numpy(self.pcd1[:self.n_pts]).float().to(self.dev)
        ref_shape_pcd = torch.from_numpy(self.pcd_ref[:self.n_pts]).float().to(self.dev)
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
            # update pcd1 - pcd4
            self.shape_completion()
            for k in range(4):
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
                for ii in range(M):
                    # mi_point_cloud.append(torch.from_numpy(self.pcd2[:self.n_pts]).float().to(self.dev))
                    if k == 0:
                        mi_point_cloud.append(torch.from_numpy(self.pcd1_c[:self.n_pts]).float().to(self.dev))
                    elif k == 1:
                        mi_point_cloud.append(torch.from_numpy(self.pcd2_c[:self.n_pts]).float().to(self.dev))
                    elif k == 2:
                        mi_point_cloud.append(torch.from_numpy(self.pcd3_c[:self.n_pts]).float().to(self.dev))
                    else:
                        mi_point_cloud.append(torch.from_numpy(self.pcd4_c[:self.n_pts]).float().to(self.dev))

                mi_point_cloud = torch.stack(mi_point_cloud, 0)
                opt_model_input['point_cloud'] = mi_point_cloud
                opt_latent = self.model.extract_latent(opt_model_input).detach()

                loss_values = []

                # run optimization
                for i in range(self.opt_iterations):
                    T_mat = torch_util.angle_axis_to_rotation_matrix(rot).squeeze()
                    # noise_vec = (torch.randn(X.size()) * (self.perturb_scale / ((i + 1) ** (self.perturb_decay)))).to(self.dev)
                    # X_perturbed = X + noise_vec
                    if k == 0:
                        trans_ = trans - (torch.from_numpy(self.mean1[None, :])).float().to(self.dev)
                    elif k == 1:
                        trans_ = trans - (torch.from_numpy(self.mean2[None, :])).float().to(self.dev)
                    elif k == 2:
                        trans_ = trans - (torch.from_numpy(self.mean3[None, :])).float().to(self.dev)
                    else:
                        trans_ = trans - (torch.from_numpy(self.mean4[None, :])).float().to(self.dev)
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

                if k == 0:
                    grasp_obj[:3, -1] = grasp_trans + self.mean1
                    self.grasp_list_1.append(grasp_obj @ rand_mat_init_)
                elif k == 1:
                    grasp_obj[:3, -1] = grasp_trans + self.mean2
                    self.grasp_list_2.append(grasp_obj @ rand_mat_init_)
                elif k == 2:
                    grasp_obj[:3, -1] = grasp_trans + self.mean3
                    self.grasp_list_3.append(grasp_obj @ rand_mat_init_)
                else:
                    grasp_obj[:3, -1] = grasp_trans + self.mean4
                    self.grasp_list_4.append(grasp_obj @ rand_mat_init_)

        # visualization
        n_grasps_all = self.n_samples
        vis_grasp_list1 = [(self.grasp_list_1[i] @ hom_query_pts).T[:, 0:3] for i in range(n_grasps_all)]
        vis_grasp_list2 = [(self.grasp_list_2[i] @ hom_query_pts).T[:, 0:3] for i in range(n_grasps_all)]
        vis_grasp_list3 = [(self.grasp_list_3[i] @ hom_query_pts).T[:, 0:3] for i in range(n_grasps_all)]
        vis_grasp_list4 = [(self.grasp_list_4[i] @ hom_query_pts).T[:, 0:3] for i in range(n_grasps_all)]

        ps.init()
        ps.set_up_dir("z_up")
        for i in range(n_grasps_all):
            # ps.register_point_cloud("pcd" + str(i), X_new[i].detach().cpu().numpy(), radius=0.005, enabled=True)
            ps.register_point_cloud("g_1_" + str(i), vis_grasp_list1[i], radius=0.005, enabled=True)
            ps.register_point_cloud("g_2_" + str(i), vis_grasp_list2[i], radius=0.005, enabled=True)
            ps.register_point_cloud("g_3_" + str(i), vis_grasp_list3[i], radius=0.005, enabled=True)
            ps.register_point_cloud("g_4_" + str(i), vis_grasp_list4[i], radius=0.005, enabled=True)

        ps.register_point_cloud("query", query_pts_vis + self.mean1, radius=0.005, enabled=True)
        ps.register_point_cloud("obj1", self.pcd1_c[:self.n_pts] + self.mean1, radius=0.005, enabled=True)
        ps.register_point_cloud("obj2", self.pcd2_c[:self.n_pts] + self.mean2, radius=0.005, enabled=True)
        ps.register_point_cloud("obj3", self.pcd3_c[:self.n_pts] + self.mean3, radius=0.005, enabled=True)
        ps.register_point_cloud("obj4", self.pcd4_c[:self.n_pts] + self.mean4, radius=0.005, enabled=True)

        ps.show()

        self.grasp_list = self.grasp_list_1 + self.grasp_list_2 + self.grasp_list_3 + self.grasp_list_4
