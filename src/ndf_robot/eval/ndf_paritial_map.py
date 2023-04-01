import os, os.path as osp
from typing import Optional, Any

import polyscope as ps
import torch
import numpy as np
import trimesh
import random
import copy
import plotly.graph_objects as go

from ndf_robot.utils import torch_util, trimesh_util
from ndf_robot.utils.plotly_save import plot3d


class NDFPartialMap:
    def __init__(self, model, pcd1, pcd2, ref_grasp, model_type='pointnet', opt_iterations=500, sigma=0.025, trimesh_viz=False):
        self.model = model
        self.model_type = model_type
        self.opt_iterations = opt_iterations
        self.sigma = sigma
        self.trimesh_viz = trimesh_viz
        self.ref_grasp = ref_grasp

        self.perturb_scale = 0.001
        self.perturb_decay = 0.5
        self.n_pts = 2000
        self.n_opt_pts = 500
        self.prepare_inputs(pcd1, pcd2)

        self.loss_fn = torch.nn.L1Loss()
        if torch.cuda.is_available():
            self.dev = torch.device('cuda:0')
        else:
            self.dev = torch.device('cpu')
        self.model = self.model.to(self.dev)
        self.model.eval()

        self.viz_path = 'visualization'
        if not osp.exists(self.viz_path):
            os.makedirs(self.viz_path)

        self._cam_frame_scene_dict()

    def _cam_frame_scene_dict(self):
        self.cam_frame_scene_dict = {}
        cam_up_vec = [0, 1, 0]
        plotly_camera = {
            'up': {'x': cam_up_vec[0], 'y': cam_up_vec[1], 'z': cam_up_vec[2]},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'eye': {'x': -0.6, 'y': -0.6, 'z': 0.4},
        }

        plotly_scene = {
            'xaxis':
                {
                    'backgroundcolor': 'rgb(255, 255, 255)',
                    'gridcolor': 'white',
                    'zerolinecolor': 'white',
                    'tickcolor': 'rgb(255, 255, 255)',
                    'showticklabels': False,
                    'showbackground': False,
                    'showaxeslabels': False,
                    'visible': False,
                    'range': [-0.5, 0.5]},
            'yaxis':
                {
                    'backgroundcolor': 'rgb(255, 255, 255)',
                    'gridcolor': 'white',
                    'zerolinecolor': 'white',
                    'tickcolor': 'rgb(255, 255, 255)',
                    'showticklabels': False,
                    'showbackground': False,
                    'showaxeslabels': False,
                    'visible': False,
                    'range': [-0.5, 0.5]},
            'zaxis':
                {
                    'backgroundcolor': 'rgb(255, 255, 255)',
                    'gridcolor': 'white',
                    'zerolinecolor': 'white',
                    'tickcolor': 'rgb(255, 255, 255)',
                    'showticklabels': False,
                    'showbackground': False,
                    'showaxeslabels': False,
                    'visible': False,
                    'range': [-0.5, 0.5]},
        }
        self.cam_frame_scene_dict['camera'] = plotly_camera
        self.cam_frame_scene_dict['scene'] = plotly_scene

    def prepare_inputs(self, pcd1, pcd2):
        pcd1_mean = np.mean(pcd1, axis=0)
        pcd2_mean = np.mean(pcd2, axis=0)

        # sample 2000 points
        # rix_1 = np.random.permutation(pcd1.shape[0])
        # pcd1 = pcd1[rix_1[:2000]]
        # rix_2 = np.random.permutation(pcd2.shape[0])
        # pcd2 = pcd2[rix_2[:2000]]
        np.random.shuffle(pcd1)
        np.random.shuffle(pcd2)

        # filter outliers
        inliers = np.where(np.linalg.norm(pcd1 - pcd1_mean, 2, 1) < 0.2)[0]
        pcd1 = pcd1[inliers]
        inliers = np.where(np.linalg.norm(pcd2 - pcd2_mean, 2, 1) < 0.2)[0]
        pcd2 = pcd2[inliers]

        # a = np.argmax(np.abs(pcd1[:, 0]))
        # b = np.argmax(np.abs(pcd1[:, 1]))
        # c = np.argmax(np.abs(pcd1[:, 2]))
        # a_ = pcd1[a]
        # b_ = pcd1[b]
        # c_ = pcd1[c]
        #
        # self.ref_grasp[0:3, -1] = b_[None, :]

        self.mean1 = np.mean(pcd1, axis=0)
        self.mean2 = np.mean(pcd2, axis=0)
        self.pcd1 = pcd1 - self.mean1
        self.pcd2 = pcd2 - self.mean2

        # # visualization
        # ps.init()
        # ps.set_up_dir("z_up")
        # ps1 = ps.register_point_cloud("pcd_whole_mujoco", pcd1, radius=0.005, enabled=True)
        # ps2 = ps.register_point_cloud("object", pcd2, radius=0.005, enabled=True)
        # # tpcd1 = trimesh.PointCloud(self.pcd1[:self.n_pts])
        # # tpcd2 = trimesh.PointCloud(self.pcd2[:self.n_pts])
        # # tpcd1.show()
        # # tpcd2.show()
        # ps.show()

    def sample_pts(self, show_recon=False, return_scene=False, visualize_all_inits=False, render_video=False):
        # sample query points
        n = self.n_opt_pts
        query_x = np.random.uniform(-0.02, 0.02, n)
        query_y = np.random.uniform(-0.04, 0.04, n)
        query_z = np.random.uniform(-0.05 + 0.1, 0.02 + 0.1, n)
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

        # show query points and object
        pcd3 = np.vstack([ref_query_pts, self.pcd1])
        tpcd1 = trimesh.PointCloud(pcd3)
        # tpcd1.show()

        # shape completion
        thresh = 0.2
        shape_1 = {}
        shape_1['point_cloud'] = torch.from_numpy(self.pcd1[:self.n_pts]).float().to(self.dev)[None, :, :]
        shape_pcd = trimesh.PointCloud(self.pcd1)
        bb = shape_pcd.bounding_box
        bb_scene = trimesh.Scene();
        bb_scene.add_geometry([shape_pcd, bb])
        # bb_scene.show()

        eval_pts = bb.sample_volume(20000)
        shape_1['coords'] = torch.from_numpy(eval_pts)[None, :, :].float().to(self.dev).detach()
        out = self.model(shape_1)

        in_inds = torch.where(out['occ'].squeeze() > thresh)[0].cpu().numpy()
        in_pts = eval_pts[in_inds]

        ps.init()
        ps.set_up_dir("z_up")
        ps.register_point_cloud("pcd1", self.pcd1, radius=0.005, enabled=True)
        ps.register_point_cloud("pcd2", in_pts, radius=0.005, enabled=True)
        ps.show()

        pcd1 = np.vstack([self.pcd1, in_pts[:1000]])
        np.random.shuffle(pcd1)

        shape_2 = {}
        shape_2['point_cloud'] = torch.from_numpy(self.pcd2[:self.n_pts]).float().to(self.dev)[None, :, :]
        shape_pcd = trimesh.PointCloud(self.pcd2)
        bb = shape_pcd.bounding_box
        bb_scene = trimesh.Scene();
        bb_scene.add_geometry([shape_pcd, bb])
        # bb_scene.show()

        eval_pts = bb.sample_volume(20000)
        shape_2['coords'] = torch.from_numpy(eval_pts)[None, :, :].float().to(self.dev).detach()
        out = self.model(shape_2)
        in_inds = torch.where(out['occ'].squeeze() > thresh)[0].cpu().numpy()
        in_pts = eval_pts[in_inds]

        ps.init()
        ps.set_up_dir("z_up")
        ps.register_point_cloud("pcd1", self.pcd2, radius=0.005, enabled=True)
        ps.register_point_cloud("pcd2", in_pts, radius=0.005, enabled=True)
        ps.show()

        pcd2 = np.vstack([self.pcd2, in_pts[:2000]])
        np.random.shuffle(pcd2)

        # start
        reference_model_input = {}
        ref_query_pts = torch.from_numpy(ref_query_pts).float().to(self.dev)
        # ref_shape_pcd = torch.from_numpy(self.pcd1[:self.n_pts]).float().to(self.dev)
        ref_shape_pcd = torch.from_numpy(pcd1[:self.n_pts]).float().to(self.dev)
        reference_model_input['coords'] = ref_query_pts[None, :, :]
        reference_model_input['point_cloud'] = ref_shape_pcd[None, :, :]

        # get the descriptors for these reference query points
        reference_latent = self.model.extract_latent(reference_model_input).detach()
        reference_act_hat = self.model.forward_latent(reference_latent, reference_model_input['coords']).detach()

        # set up the optimization
        if 'dgcnn' in self.model_type:
            full_opt = 5  # dgcnn can't fit 10 initialization in memory
        else:
            full_opt = 10
        best_loss = np.inf
        best_tf = np.eye(4)
        best_idx = 0
        tf_list = []
        M = full_opt

        # parameters for optimization
        trans = (torch.rand((M, 3)) * 0.1).float().to(self.dev)
        rot = torch.rand(M, 3).float().to(self.dev)
        trans.requires_grad_()
        rot.requires_grad_()
        opt = torch.optim.Adam([trans, rot], lr=1e-2)

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
            mi_point_cloud.append(torch.from_numpy(pcd2[:self.n_pts]).float().to(self.dev))
        mi_point_cloud = torch.stack(mi_point_cloud, 0)
        opt_model_input['point_cloud'] = mi_point_cloud
        opt_latent = self.model.extract_latent(opt_model_input).detach()

        loss_values = []
        vid_plot_idx = None

        # run optimization

        pcd_traj_list = {}
        for jj in range(M):
            pcd_traj_list[jj] = []
            pcd_traj_list[jj].append(np.mean(X[jj].detach().cpu().numpy(), axis=0))
        for i in range(self.opt_iterations):
            if i == 0:
                jj = 0
                shape_mi = {}
                shape_mi['point_cloud'] = opt_model_input['point_cloud'][jj][None, :, :].detach()
                shape_np = shape_mi['point_cloud'].cpu().numpy().squeeze()
                shape_mean = np.mean(shape_np, axis=0)
                inliers = np.where(np.linalg.norm(shape_np - shape_mean, 2, 1) < 0.2)[0]
                shape_np = shape_np[inliers]
                shape_pcd = trimesh.PointCloud(shape_np)
                bb = shape_pcd.bounding_box
                bb_scene = trimesh.Scene(); bb_scene.add_geometry([shape_pcd, bb])
                # bb_scene.show()

                eval_pts = bb.sample_volume(50000)
                shape_mi['coords'] = torch.from_numpy(eval_pts)[None, :, :].float().to(self.dev).detach()
                out = self.model(shape_mi)
                thresh = 0.1
                in_inds = torch.where(out['occ'].squeeze() > thresh)[0].cpu().numpy()
                out_inds = torch.where(out['occ'].squeeze() < thresh)[0].cpu().numpy()

                in_pts = eval_pts[in_inds]
                out_pts = eval_pts[out_inds]
                # if self.trimesh_viz:
                #     scene = trimesh_util.trimesh_show([in_pts])
                #     in_scene = trimesh_util.trimesh_show([in_pts, shape_np])
                # fig = plot3d(
                #     [in_pts, shape_np],
                #     ['blue', 'black'],
                #     osp.join(self.viz_path, 'recon_overlay_test.html'),
                #     scene_dict=self.cam_frame_scene_dict,
                #     z_plane=False)
                # fig.show()

                # ps.init()
                # ps.set_up_dir("z_up")
                # ps1 = ps.register_point_cloud("pcd_whole_mujoco", in_pts, radius=0.005, enabled=True)
                # ps2 = ps.register_point_cloud("object", pcd2, radius=0.005, enabled=True)
                # # tpcd1 = trimesh.PointCloud(self.pcd1[:self.n_pts])
                # # tpcd2 = trimesh.PointCloud(self.pcd2[:self.n_pts])
                # # tpcd1.show()
                # # tpcd2.show()
                # ps.show()


            T_mat = torch_util.angle_axis_to_rotation_matrix(rot).squeeze()
            noise_vec = (torch.randn(X.size()) * (self.perturb_scale / ((i + 1) ** (self.perturb_decay)))).to(self.dev)
            X_perturbed = X + noise_vec
            trans_ = trans - (torch.from_numpy(self.mean2[None, :])).float().to(self.dev)
            # X_new = torch_util.transform_pcd_torch(X_perturbed, T_mat) + trans[:, None, :].repeat((1, X.size(1), 1))
            X_new = torch_util.transform_pcd_torch(X_perturbed, T_mat) + trans_[:, None, :].repeat((1, X.size(1), 1))

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

            # try to guess which run in the batch will lead to the lowest cost
            if i > 5 and (vid_plot_idx is None):
                vid_plot_idx = torch.argmin(torch.stack(losses)).item()
                print('vid plot idx: ', vid_plot_idx)
                plot_rand_mat_np = rand_mat_init[vid_plot_idx].detach().cpu().numpy()
        #
        #     if i < 200:
        #         render_iter = (i % 4 == 0)
        #     else:
        #         render_iter = (i % 8 == 0)
        #
        best_idx = torch.argmin(torch.stack(losses)).item()
        best_loss = losses[best_idx]
        print('best loss: %f, best_idx: %d' % (best_loss, best_idx))

        best_X = X_new[best_idx].detach().cpu().numpy()
        grasp_obj = T_mat[best_idx].detach().cpu().numpy()
        grasp_trans = trans_[best_idx].detach().cpu().numpy()
        # # show query points and object
        # pcd4 = np.vstack([best_X, self.pcd2])
        # tpcd2 = trimesh.PointCloud(pcd4)
        # tpcd2.show()

        coords = np.array([[0.1, 0, 0, 0],
                           [0., 0.1, 0, 0],
                           [0, 0, 0.1, 0],
                           [1, 1, 1, 1]])

        coords_1 = self.ref_grasp @ coords
        coords_1 = coords_1[0:3, :] - self.mean1[:, None]
        coords_1 = coords_1.T

        coords_2 = grasp_obj @ coords
        # coords_2 = coords_2[0:3, :] + grasp_trans[:, None]
        coords_2 = coords_2[0:3, :]
        coords_2 = coords_2.T

        ps.init()
        ps.set_up_dir("z_up")
        # ps.register_point_cloud("pcd1", self.pcd1[:self.n_pts], radius=0.005, enabled=True)
        ps.register_point_cloud("pcd1", pcd1[:self.n_pts], radius=0.005, enabled=True)
        ps.register_point_cloud("pcd2", query_pts_vis, radius=0.005, enabled=True)
        # ps.register_point_cloud("pcd3", self.pcd2[:self.n_pts], radius=0.005, enabled=True)
        ps.register_point_cloud("pcd3", pcd2[:self.n_pts], radius=0.005, enabled=True)
        ps.register_point_cloud("pcd4", best_X, radius=0.005, enabled=True)

        ps.register_curve_network("edge_x" + str(1), coords_1[[0, 3]], np.array([[0, 1]]),
                                  enabled=True, radius=0.0003, color=(1, 0, 0))
        ps.register_curve_network("edge_y" + str(1), coords_1[[1, 3]], np.array([[0, 1]]),
                                  enabled=True, radius=0.0003, color=(0, 1, 0))
        ps.register_curve_network("edge_z" + str(1), coords_1[[2, 3]], np.array([[0, 1]]),
                                  enabled=True, radius=0.0003, color=(0, 0, 1))

        # ps.register_curve_network("edge_x" + str(2), coords_2[[0, 3]], np.array([[0, 1]]),
        #                           enabled=True, radius=0.0003, color=(1, 0, 0))
        # ps.register_curve_network("edge_y" + str(2), coords_2[[1, 3]], np.array([[0, 1]]),
        #                           enabled=True, radius=0.0003, color=(0, 1, 0))
        # ps.register_curve_network("edge_z" + str(2), coords_2[[2, 3]], np.array([[0, 1]]),
        #                           enabled=True, radius=0.0003, color=(0, 0, 1))
        # tpcd1 = trimesh.PointCloud(self.pcd1[:self.n_pts])
        # tpcd2 = trimesh.PointCloud(self.pcd2[:self.n_pts])
        # tpcd1.show()
        # tpcd2.show()
        ps.show()
        #
        # offset = np.array([0.4, 0, 0])
        # vpcd1: Optional[Any] = copy.deepcopy(self.pcd1)
        # vquery1 = copy.deepcopy(reference_query_pts)
        #
        # vpcd1 += offset
        # vquery1 += offset
        #
        # trans_best, rot_best = trans[best_idx], rot[best_idx]
        # transform_mat_np = torch_util.angle_axis_to_rotation_matrix(
        #     rot_best.view(1, -1)).squeeze().detach().cpu().numpy()
        # transform_mat_np[:-1, -1] = trans_best.detach().cpu().numpy()
        # rand_mat_np = rand_mat_init[best_idx].detach().cpu().numpy()
        #
        # frame1_tf = np.eye(4)
        # frame1_tf[:-1, -1] = (q_offset + offset)
        # frame2_tf = np.matmul(transform_mat_np, rand_mat_np)
        #
        # frame1 = self.plotly_create_local_frame(transform=frame1_tf)
        # frame2 = self.plotly_create_local_frame(transform=rand_mat_np)
        # frame3 = self.plotly_create_local_frame(transform=frame2_tf)
        # frame_data = frame1 + frame2 + frame3
        #
        # best_scene = None
