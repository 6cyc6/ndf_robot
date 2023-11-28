import os
import os.path as osp
import sys

import polyscope as ps
import torch
import numpy as np
import trimesh
import time
import random
import copy

from ndf_robot.utils import torch_util, path_util
import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn_nift as vnn_occupancy_network

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))


class NIFTInference:
    def __init__(self, model_type='pointnet', obj="cups"):
        self.hom_query_pts = None
        self.ref_pts_gripper = None
        self.ref_pts_plane = None
        self.query_pts_vis = None
        self.ref_query_pts = None

        self.ref_grasp = None
        self.ref_place = None
        self.model = None
        self.model_type = model_type
        self.obj = obj

        self.pcd_ref = None
        self.mean_ref = None
        self.pcd_inf = None
        self.mean_inf = None
        self.ref_pose_mat = None
        self.ref_pose_mat_tensor = None

        # loss function and device
        self.loss_fn = torch.nn.L1Loss()
        if torch.cuda.is_available():
            self.dev = torch.device('cuda:0')
        else:
            self.dev = torch.device('cpu')

        # hyper parameters
        self.opt_iterations = 500
        self.n_init = 8
        self.n_pts = 1500
        self.n_rb_pts = 300

        self.place_sample_x = 0.1  # region to sample the plane
        self.place_sample_z = 0.02

        self.thresh = 0.15
        self.z = 0.045

        # load model and weights
        self.load_weights()

        # load canonical shape of the cup
        # self.load_canonical()

    def load_weights(self):
        model = vnn_occupancy_network.VNNOccNet(latent_dim=256, o_dim=5, return_features=True).cuda()

        model_path = osp.join(path_util.get_ndf_model_weights(), 'nift_mug.pth')
        model.load_state_dict(torch.load(model_path))

        print("NIFT model is successfully loaded.")
        self.model = model.to(self.dev)
        self.model.eval()

    def shape_completion(self, pcd, tensor=False):
        shape = {}
        if not tensor:
            shape['point_cloud'] = torch.from_numpy(pcd[:self.n_pts]).float().to(self.dev)[None, :, :]
        else:
            shape['point_cloud'] = pcd[:self.n_pts].float().to(self.dev)[None, :, :]
        shape_pcd = trimesh.PointCloud(pcd)
        bb = shape_pcd.bounding_box

        eval_pts = bb.sample_volume(20000)
        shape['coords'] = torch.from_numpy(eval_pts)[None, :, :].float().to(self.dev).detach()
        out = self.model(shape)
        # in_inds = torch.where(out['occ'].squeeze() > self.thresh)[0].cpu().numpy()
        print(out['occ'])
        in_inds = torch.where(out['occ'].squeeze() < -0.1)[0].cpu().numpy()
        in_pts = eval_pts[in_inds]

        return in_pts

    def sample_gripper_pts(self):
        # sample query points
        n = self.n_rb_pts
        query_x = np.random.uniform(-0.02, 0.02, n)
        query_y = np.random.uniform(-0.04, 0.04, n)
        # query_z = np.random.uniform(-0.06 + self.z, 0.000 + self.z, n)
        query_z = np.random.uniform(0.05, 0.12, n)
        # query_z = np.random.uniform(-0.05 + 0.1, 0.01 + 0.1, n)
        ones = np.ones(n)
        ref_pts_gripper = np.vstack([query_x, query_y, query_z])
        ref_pts_gripper = ref_pts_gripper.T
        hom_query_pts = np.vstack([query_x, query_y, query_z, ones])

        # transform
        ref_query_pts = self.ref_grasp @ hom_query_pts
        ref_query_pts = ref_query_pts[:3, :] - self.mean_ref[:, None]
        ref_query_pts = ref_query_pts.T

        self.hom_query_pts = hom_query_pts
        self.ref_pts_gripper = ref_pts_gripper
        self.ref_query_pts = ref_query_pts
        self.query_pts_vis = copy.deepcopy(ref_query_pts)

    def sample_place_ref_pts(self):
        # sample query points for placement
        n = self.n_rb_pts
        query_x = np.random.uniform(-self.place_sample_x, self.place_sample_x, n)
        query_y = np.random.uniform(-self.place_sample_x, self.place_sample_x, n)
        query_z = np.random.uniform(0 - 0.01, self.place_sample_z - 0.01, n)

        ones = np.ones(n)
        ref_pts_plane = np.vstack([query_x, query_y, query_z])
        ref_pts_plane = ref_pts_plane.T
        hom_query_pts = np.vstack([query_x, query_y, query_z, ones])

        # transform
        ref_query_pts = self.ref_place @ hom_query_pts
        ref_query_pts = ref_query_pts[:3, :] - self.mean_ref[:, None]
        ref_query_pts = ref_query_pts.T

        self.hom_query_pts = hom_query_pts
        self.ref_pts_plane = ref_pts_plane
        self.ref_query_pts = ref_query_pts
        self.query_pts_vis = copy.deepcopy(ref_query_pts)

    def inference_grasp(self, pcd_obj, vis=False):
        # ---------------- sample gripper control points and get reference query points --------------------- #
        self.sample_gripper_pts()

        # load inference point cloud
        if not torch.is_tensor(pcd_obj):
            pcd_obj = torch.from_numpy(pcd_obj).float().to(self.dev)

        pcd_obj_mean = torch.mean(pcd_obj, 0)
        pcd_obj = pcd_obj - pcd_obj_mean

        idx = torch.randperm(pcd_obj.shape[0])
        pcd_obj = pcd_obj[idx].view(pcd_obj.size())

        # -------------------------- start algorithm ----------------------------------- #
        reference_model_input = {}
        ref_query_pts = torch.from_numpy(self.ref_query_pts).float().to(self.dev)
        # ----------------------------- pcd_ref -------------------------------------- #
        ref_shape_pcd = torch.from_numpy(self.pcd_ref[:self.n_pts]).float().to(self.dev)
        reference_model_input['coords'] = ref_query_pts[None, :, :]
        reference_model_input['point_cloud'] = ref_shape_pcd[None, :, :]

        # get the descriptors for these reference query points
        reference_latent = self.model.extract_latent(reference_model_input).detach()
        reference_act_hat = self.model.forward_latent(reference_latent, reference_model_input['coords']).detach()

        # # parameters for optimization
        # trans = (torch.rand((self.n_init, 3)) * 0.1).float().to(self.dev)
        # rot = torch.rand(self.n_init, 3).float().to(self.dev)
        # trans.requires_grad_()
        # rot.requires_grad_()
        # opt = torch.optim.Adam([trans, rot], lr=1e-2)
        #
        # # initialization
        # rand_rot_init = (torch.rand((self.n_init, 3)) * 2 * np.pi).float().to(self.dev)
        # rand_mat_init = torch_util.angle_axis_to_rotation_matrix(rand_rot_init)
        # rand_mat_init = rand_mat_init.squeeze().float().to(self.dev)

        # parameters for optimization
        t1 = np.eye(3) * 0.1
        t2 = np.eye(3) * -0.1
        t3 = np.random.rand(self.n_init - 6, 3) * 0.1
        t = np.vstack([t1, t2, t3])
        trans = torch.from_numpy(t).float().to(self.dev)
        rot = torch.rand(self.n_init, 3).float().to(self.dev)

        trans.requires_grad_()
        rot.requires_grad_()
        opt = torch.optim.Adam([trans, rot], lr=1e-2)

        # initialization
        r1 = np.array([[-1.2091996, -1.2091996, 1.2091996],
                       [0, -2.2214415, 2.2214415],
                       [0, np.pi, 0],
                       [2.4183992, -2.4183992, 2.4183992],
                       [-1.5707963, 0, 0],
                       [0.001, 0.001, 0.001]])
        r2 = np.random.rand(self.n_init - 6, 3) * 2 * np.pi
        r = np.vstack([r1, r2])
        rand_rot_init = torch.from_numpy(r).float().to(self.dev)
        # rand_rot_init = (torch.rand((self.n_init, 3)) * 2 * np.pi).float().to(self.dev)
        rand_mat_init = torch_util.angle_axis_to_rotation_matrix(rand_rot_init)
        rand_mat_init = rand_mat_init.squeeze().float().to(self.dev)

        # now randomly initialize a copy of the query points
        opt_query_pts = torch.from_numpy(self.ref_pts_gripper).float().to(self.dev)
        opt_query_pts = opt_query_pts[None, :, :].repeat((self.n_init, 1, 1))
        X = torch_util.transform_pcd_torch(opt_query_pts, rand_mat_init)

        opt_model_input = {}
        opt_model_input['coords'] = X

        mi_point_cloud = []
        # ----------------------------- pcds -------------------------------------- #
        for ii in range(self.n_init):
            if ii % 2 == 0:
                idx = torch.randperm(pcd_obj.shape[0])
                pcd_obj = pcd_obj[idx].view(pcd_obj.size())
            mi_point_cloud.append(pcd_obj[:self.n_pts])

        mi_point_cloud = torch.stack(mi_point_cloud, 0)
        opt_model_input['point_cloud'] = mi_point_cloud
        opt_latent = self.model.extract_latent(opt_model_input).detach()

        loss_values = []

        losses = None
        T_mat = None

        t = time.time()
        # run optimization
        for i in range(self.opt_iterations):
            T_mat = torch_util.angle_axis_to_rotation_matrix(rot).squeeze()
            # noise_vec = (torch.randn(X.size()) * (self.perturb_scale / ((i + 1) ** (self.perturb_decay)))).to(self.dev)
            # X_perturbed = X + noise_vec

            # X_new = torch_util.transform_pcd_torch(X_perturbed, T_mat) + trans[:, None, :].repeat((1, X.size(1), 1))
            X_new = torch_util.transform_pcd_torch(X, T_mat) + trans[:, None, :].repeat((1, X.size(1), 1))

            act_hat = self.model.forward_latent(opt_latent, X_new)
            t_size = reference_act_hat.size()

            losses = [self.loss_fn(act_hat[ii].view(t_size), reference_act_hat) for ii in range(self.n_init)]
            loss = torch.mean(torch.stack(losses))
            if i % 100 == 0:
                losses_str = ['%f' % val.item() for val in losses]
                loss_str = ', '.join(losses_str)
                print(f'i: {i}, losses: {loss_str}')
            loss_values.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()

        # get the best one as output
        best_idx = torch.argmin(torch.stack(losses)).item()
        grasp_obj = T_mat[best_idx].detach().cpu().numpy()
        grasp_trans = trans[best_idx].detach().cpu().numpy()

        rand_mat_init_ = rand_mat_init[best_idx].detach().cpu().numpy()

        grasp_obj_tensor = T_mat[best_idx].detach()
        rand_mat_init_tensor = rand_mat_init[best_idx].detach()
        grasp_trans_tensor = trans[best_idx].detach()
        # ----------------------------- trans_back -------------------------------------- #
        grasp_obj[:3, -1] = grasp_trans + pcd_obj_mean.cpu().numpy()
        grasp_obj_tensor[:3, -1] = grasp_trans_tensor + pcd_obj_mean

        grasp_opt_tensor = grasp_obj_tensor @ rand_mat_init_tensor
        grasp_opt = grasp_obj @ rand_mat_init_

        if vis:
            # visualization
            vis_grasp = (grasp_opt @ self.hom_query_pts).T[:, 0:3]

            gripper_control_points_armar = np.array([[0.00000, 0.00000, -0.06000],
                                                     [0.00000, 0.00000, 0.00000],
                                                     [-0.0500, 0.00000, 0.05000],
                                                     [0.07000, 0.00000, 0.07000]])
            gripper_control_points_armar += np.array([0, 0, 0.06])
            ps.init()
            ps.set_up_dir("z_up")

            ps.register_point_cloud("grasp", vis_grasp, radius=0.004, enabled=True)

            coords = np.concatenate((gripper_control_points_armar, np.ones((4, 1))), axis=1)
            coords = grasp_opt @ coords.T
            coords = coords[0:3, :]
            coords = coords.T
            nodes_armar = coords
            ps.register_curve_network("edge_1", nodes_armar[[0, 1]], np.array([[0, 1]]),
                                      enabled=True, radius=0.0015, color=(0, 0, 1))
            ps.register_curve_network("edge_2", nodes_armar[[1, 2]], np.array([[0, 1]]),
                                      enabled=True, radius=0.0015, color=(1, 0, 0))
            ps.register_curve_network("edge_3", nodes_armar[[1, 3]], np.array([[0, 1]]),
                                      enabled=True, radius=0.0015, color=(0, 1, 0))

            coords = np.concatenate((gripper_control_points_armar, np.ones((4, 1))), axis=1)
            coords = self.ref_grasp @ coords.T
            coords = coords[0:3, :]
            coords = coords.T
            nodes_armar = coords - self.mean_ref - np.array([0.4, 0, 0])
            ps.register_curve_network("edge_1_ref", nodes_armar[[0, 1]], np.array([[0, 1]]),
                                      enabled=True, radius=0.0015, color=(0, 0, 1))
            ps.register_curve_network("edge_2_ref", nodes_armar[[1, 2]], np.array([[0, 1]]),
                                      enabled=True, radius=0.0015, color=(1, 0, 0))
            ps.register_curve_network("edge_3_ref", nodes_armar[[1, 3]], np.array([[0, 1]]),
                                      enabled=True, radius=0.0015, color=(0, 1, 0))

            pcd_obj_mean = pcd_obj_mean.cpu().numpy()
            ps.register_point_cloud("query", self.query_pts_vis - np.array([0.4, 0, 0]) + pcd_obj_mean, radius=0.004,
                                    color=[1, 0, 0], enabled=True)
            ps.register_point_cloud("obj_ref", self.pcd_ref[:self.n_pts] - np.array([0.4, 0, 0]) + pcd_obj_mean, radius=0.004,
                                    color=[0.5, 0.5, 0], enabled=True)
            ps.register_point_cloud("obj", pcd_obj.cpu().numpy()[:self.n_pts] + pcd_obj_mean, radius=0.004, enabled=True)

            ps.show()

        print(str(time.time() - t) + 's.')
        return grasp_opt_tensor

    def inference_place(self, pcd_obj, vis=False):
        # ---------------- sample gripper control points and get reference query points --------------------- #
        ref_place = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, -0.101],
                              [0, 0, 0, 1]])
        self.ref_place = self.ref_pose_mat @ ref_place
        self.sample_place_ref_pts()
        self.n_pts = 1500

        # load inference point cloud
        if not torch.is_tensor(pcd_obj):
            pcd_obj = torch.from_numpy(pcd_obj).float().to(self.dev)

        pcd_obj_mean = torch.mean(pcd_obj, 0)
        pcd_obj = pcd_obj - pcd_obj_mean

        idx = torch.randperm(pcd_obj.shape[0])
        pcd_obj = pcd_obj[idx].view(pcd_obj.size())

        # -------------------------- start algorithm ----------------------------------- #
        reference_model_input = {}
        ref_query_pts = torch.from_numpy(self.ref_query_pts).float().to(self.dev)
        # ----------------------------- pcd_ref -------------------------------------- #
        ref_shape_pcd = torch.from_numpy(self.pcd_ref[:self.n_pts]).float().to(self.dev)
        reference_model_input['coords'] = ref_query_pts[None, :, :]
        reference_model_input['point_cloud'] = ref_shape_pcd[None, :, :]

        # get the descriptors for these reference query points
        reference_latent = self.model.extract_latent(reference_model_input).detach()
        reference_act_hat = self.model.forward_latent(reference_latent, reference_model_input['coords']).detach()

        # # parameters for optimization
        # trans = (torch.rand((self.n_init, 3)) * 0.1).float().to(self.dev)
        # rot = torch.rand(self.n_init, 3).float().to(self.dev)
        # trans.requires_grad_()
        # rot.requires_grad_()
        # opt = torch.optim.Adam([trans, rot], lr=1e-2)
        #
        # # initialization
        # rand_rot_init = (torch.rand((self.n_init, 3)) * 2 * np.pi).float().to(self.dev)
        # rand_mat_init = torch_util.angle_axis_to_rotation_matrix(rand_rot_init)
        # rand_mat_init = rand_mat_init.squeeze().float().to(self.dev)

        # parameters for optimization
        t1 = np.eye(3) * 0.1
        t2 = np.eye(3) * -0.1
        t3 = np.random.rand(self.n_init - 6, 3) * 0.1
        t = np.vstack([t1, t2, t3])
        trans = torch.from_numpy(t).float().to(self.dev)
        rot = torch.rand(self.n_init, 3).float().to(self.dev)

        trans.requires_grad_()
        rot.requires_grad_()
        opt = torch.optim.Adam([trans, rot], lr=1e-2)

        # initialization
        r1 = np.array([[-1.2091996, -1.2091996, 1.2091996],
                       [0, -2.2214415, 2.2214415],
                       [0, np.pi, 0],
                       [2.4183992, -2.4183992, 2.4183992],
                       [-1.5707963, 0, 0],
                       [0.001, 0.001, 0.001]])
        r2 = np.random.rand(self.n_init - 6, 3) * 2 * np.pi
        r = np.vstack([r1, r2])
        rand_rot_init = torch.from_numpy(r).float().to(self.dev)
        # rand_rot_init = (torch.rand((self.n_init, 3)) * 2 * np.pi).float().to(self.dev)
        rand_mat_init = torch_util.angle_axis_to_rotation_matrix(rand_rot_init)
        rand_mat_init = rand_mat_init.squeeze().float().to(self.dev)

        # now randomly initialize a copy of the query points
        opt_query_pts = torch.from_numpy(self.ref_pts_plane).float().to(self.dev)
        opt_query_pts = opt_query_pts[None, :, :].repeat((self.n_init, 1, 1))
        X = torch_util.transform_pcd_torch(opt_query_pts, rand_mat_init)

        opt_model_input = {}
        opt_model_input['coords'] = X

        mi_point_cloud = []
        # ----------------------------- pcds -------------------------------------- #
        for ii in range(self.n_init):
            if ii % 2 == 0:
                idx = torch.randperm(pcd_obj.shape[0])
                pcd_obj = pcd_obj[idx].view(pcd_obj.size())
            mi_point_cloud.append(pcd_obj[:self.n_pts])

        mi_point_cloud = torch.stack(mi_point_cloud, 0)
        opt_model_input['point_cloud'] = mi_point_cloud
        opt_latent = self.model.extract_latent(opt_model_input).detach()

        loss_values = []

        losses = None
        T_mat = None

        t = time.time()
        # run optimization
        for i in range(self.opt_iterations):
            T_mat = torch_util.angle_axis_to_rotation_matrix(rot).squeeze()

            # X_new = torch_util.transform_pcd_torch(X_perturbed, T_mat) + trans[:, None, :].repeat((1, X.size(1), 1))
            X_new = torch_util.transform_pcd_torch(X, T_mat) + trans[:, None, :].repeat((1, X.size(1), 1))

            act_hat = self.model.forward_latent(opt_latent, X_new)
            t_size = reference_act_hat.size()

            losses = [self.loss_fn(act_hat[ii].view(t_size), reference_act_hat) for ii in range(self.n_init)]
            loss = torch.mean(torch.stack(losses))
            if i % 100 == 0:
                losses_str = ['%f' % val.item() for val in losses]
                loss_str = ', '.join(losses_str)
                print(f'i: {i}, losses: {loss_str}')
            loss_values.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()

        # get the best one as output
        best_idx = torch.argmin(torch.stack(losses)).item()
        place_obj = T_mat[best_idx].detach().cpu().numpy()
        place_trans = trans[best_idx].detach().cpu().numpy()
        best_x = X_new[best_idx].detach().cpu().numpy()

        rand_mat_init_ = rand_mat_init[best_idx].detach().cpu().numpy()

        place_obj_tensor = T_mat[best_idx].detach()
        rand_mat_init_tensor = rand_mat_init[best_idx].detach()
        place_trans_tensor = trans[best_idx].detach()
        # ----------------------------- trans_back -------------------------------------- #
        place_obj[:3, -1] = place_trans + pcd_obj_mean.cpu().numpy()
        place_obj_tensor[:3, -1] = place_trans_tensor + pcd_obj_mean

        place_opt_tensor = place_obj_tensor @ rand_mat_init_tensor
        place_opt = place_obj @ rand_mat_init_

        if vis:
            # visualization
            coords = np.array([[0.1, 0, 0, 0],
                               [0., 0.1, 0, 0],
                               [0, 0, 0.1, 0],
                               [1, 1, 1, 1]])
            coords = place_opt @ coords
            coords = coords[0:3, :]
            coords = coords.T
            nodes = coords

            ps.init()
            ps.set_up_dir("z_up")

            ps.register_curve_network("edge_x_ref", nodes[[0, 3]], np.array([[0, 1]]), enabled=True, radius=0.002,
                                      color=(1, 0, 0))
            ps.register_curve_network("edge_y_ref", nodes[[1, 3]], np.array([[0, 1]]), enabled=True, radius=0.002,
                                      color=(0, 1, 0))
            ps.register_curve_network("edge_z_ref", nodes[[2, 3]], np.array([[0, 1]]), enabled=True, radius=0.002,
                                      color=(0, 0, 1))

            pcd_obj_mean = pcd_obj_mean.cpu().numpy()
            ps.register_point_cloud("query", self.query_pts_vis - np.array([0.3, 0, 0]) + pcd_obj_mean, radius=0.005,
                                    color=[1, 0, 0], enabled=True)
            ps.register_point_cloud("obj_ref", self.pcd_ref[:self.n_pts] - np.array([0.3, 0, 0]) + pcd_obj_mean,
                                    radius=0.005, color=[0.5, 0.5, 0], enabled=True)
            ps.register_point_cloud("obj", pcd_obj.cpu().numpy()[:self.n_pts] + pcd_obj_mean,
                                    radius=0.005, enabled=True)
            ps.register_point_cloud("best", best_x + pcd_obj_mean, radius=0.005, enabled=True)

            ps.show()

        print(str(time.time() - t) + 's.')
        return place_opt_tensor
