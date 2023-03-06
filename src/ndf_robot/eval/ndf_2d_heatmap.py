import os, os.path as osp
import torch
import numpy as np
import polyscope as ps
import trimesh
import random
import copy
import plotly.graph_objects as go

from ndf_robot.utils import torch_util, trimesh_util
from ndf_robot.utils.plotly_save import plot3d


class NDFHeatmap:
    def __init__(self, model, pcd1, pcd2, model_type='pointnet', opt_iterations=500, sigma=0.025, trimesh_viz=False):
        self.model = model
        self.model_type = model_type
        self.opt_iterations = opt_iterations
        self.sigma = sigma
        self.trimesh_viz = trimesh_viz

        self.perturb_scale = 0.001
        self.perturb_decay = 0.5
        self.n_pts = 1500
        self.n_opt_pts = 50000
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

        self.video_viz_path = 'vid_visualization'
        if not osp.exists(self.video_viz_path):
            os.makedirs(self.video_viz_path)

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
        pcd1 = pcd1 - np.mean(pcd1, axis=0)
        pcd2 = pcd2 - np.mean(pcd2, axis=0)
        self.pcd1 = pcd1
        self.pcd2 = pcd2

        self.pcd1_max_z = np.argmax(np.abs(pcd1[:, 2]))
        self.pcd2_max_x = np.max(np.abs(pcd2[:, 0]))
        self.pcd2_max_y = np.max(np.abs(pcd2[:, 1]))
        self.pcd2_max_z = np.argmax(np.abs(pcd2[:, 2]))
        # print(self.pcd2_max_x, self.pcd2_max_y, self.pcd2_max_z)
        print(self.pcd2[self.pcd2_max_z])

        tpcd1 = trimesh.PointCloud(self.pcd1[:self.n_pts])
        tpcd2 = trimesh.PointCloud(self.pcd2[:self.n_pts])
        # tpcd1.show()
        # tpcd2.show()

    def sample_pts(self, show_recon=False, return_scene=False, visualize_all_inits=False, render_video=False):
        # put the query point at one of the points in the point cloud
        # q_offset_ind = np.random.randint(self.pcd1.shape[0])
        id_ = np.argmax(np.abs(self.pcd1[:, 1]))
        id_ = np.argmax(self.pcd1[:, 1])
        q_offset_ind = id_
        reference_query_pt = self.pcd1[q_offset_ind]
        noise = np.array([0.02, -0.02, -0.00])
        reference_query_pt += noise
        reference_query_pt = reference_query_pt[None, :]

        reference_model_input = {}
        ref_query_pt = torch.from_numpy(reference_query_pt[:self.n_opt_pts]).float().to(self.dev)
        ref_shape_pcd = torch.from_numpy(self.pcd1[:self.n_pts]).float().to(self.dev)
        reference_model_input['coords'] = ref_query_pt[None, :, :]
        reference_model_input['point_cloud'] = ref_shape_pcd[None, :, :]

        # get the descriptor for this reference query point
        reference_latent = self.model.extract_latent(reference_model_input).detach()
        reference_act_hat = self.model.forward_latent(reference_latent, reference_model_input['coords']).detach()

        # now calculate descriptors for object 2 at sampled query points
        # sample query points
        obs_query_pts = 0.5 * np.random.sample(size=(self.n_opt_pts, 3)) - 0.25
        # obs_query_pts = self.pcd2
        obs_query_pts_numpy = copy.deepcopy(obs_query_pts)

        # put the query points at one of the points in the point cloud
        obs_query_pts = torch.from_numpy(obs_query_pts[:self.n_opt_pts]).float().to(self.dev)
        obs_shape_pcd = torch.from_numpy(self.pcd2[:self.n_pts]).float().to(self.dev)

        obs_model_input = {}
        obs_model_input['point_cloud'] = obs_shape_pcd[None, :, :]
        obs_model_input['coords'] = obs_query_pts[None, :, :]

        # get the descriptor for query points
        obs_latent = self.model.extract_latent(obs_model_input).detach()
        obs_act_hat = self.model.forward_latent(obs_latent, obs_model_input['coords']).detach()

        # calculate losses
        reference_act_hat = torch.squeeze(reference_act_hat, 0)
        obs_act_hat = torch.squeeze(obs_act_hat, 0)
        t_size = reference_act_hat.size()

        losses = [self.loss_fn(obs_act_hat[ii].view(t_size), reference_act_hat) for ii in range(obs_act_hat.shape[0])]

        loss_querys = [losses[ii].detach().cpu().numpy() for ii in range(obs_act_hat.shape[0])]

        loss_querys = np.stack(loss_querys[:])
        # sort 20 points
        ids = np.argsort(loss_querys)

        points_1 = obs_query_pts_numpy[ids][0: 10, :]
        loss_2 = loss_querys[ids][0:100]
        points_2 = obs_query_pts_numpy[ids][10: 20, :]
        points_3 = obs_query_pts_numpy[ids][4990: 5000, :]
        loss_max = np.max(loss_querys)

        offset = 1.5
        scene = trimesh_util.trimesh_show([self.pcd1 - np.array([offset * 0.1, 0, 0]),
                                           reference_query_pt - np.array([offset * 0.1, 0, 0]),
                                           self.pcd2 + np.array([offset * 0.1, 0, 0]),
                                           points_1 + np.array([offset * 0.1, 0, 0]),
                                           # points_3 + np.array([offset * 0.1, 0, 0]),
                                           points_3 + np.array([offset * 0.1, 0, 0])], show=False)
        # scene = trimesh_util.trimesh_show([self.pcd1, reference_query_pt], show=True)
        scene = trimesh_util.trimesh_show([self.pcd2, points_1], show=False)
        scene.show()

        # Initialize polyscope
        ps.init()

        pscloud_1 = ps.register_point_cloud("pcd1", self.pcd1)
        pscloud_1.set_position(np.array([0.3, 0, 0]))
        ref_1 = ps.register_point_cloud("ref", self.pcd1[q_offset_ind][None, :])
        ref_1.set_position(np.array([0.3, 0, 0]))
        ref_1.set_radius(rad=0.02)

        pscloud_2 = ps.register_point_cloud("pcd2", self.pcd2, transparency=0.6)
        pscloud_2.set_color((0, 1, 0))
        sample_points = ps.register_point_cloud("samples", obs_query_pts_numpy[ids][0:2000], transparency=0.8)

        # manually specify a range for colormapping
        # sample_points.add_scalar_quantity("rand vals with range", loss_2, vminmax=(-5., 5.), enabled=True)

        # use a different colormap
        sample_points.add_scalar_quantity("rand vals with range", loss_querys[ids][0:2000],
                                          cmap='coolwarm', enabled=True)

        # view the point cloud with all of these quantities
        ps.show()
