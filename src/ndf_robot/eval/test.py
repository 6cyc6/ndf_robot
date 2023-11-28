import sys

import polyscope as ps
import os
import os.path as osp
import torch
import numpy as np
import trimesh
import random
import argparse
import copy
from scipy.spatial.transform import Rotation
from ndf_robot.share.globals import bad_shapenet_mug_ids_list, bad_shapenet_bowls_ids_list, bad_shapenet_bottles_ids_list

from ndf_robot.utils import path_util
import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from ndf_robot.eval.ndf_2d_heatmap import NDFHeatmap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))


dir_mesh = BASE_DIR + '/data/scaled_2.5_2.5_4.obj'
mesh1 = trimesh.load(dir_mesh, process=False)
pcd1 = mesh1.sample(5000)

dir_pcd = BASE_DIR + '/data/0.npz'

data = np.load(dir_pcd, allow_pickle=True)
pcd2 = data["pcd_local"]
pcd3 = data["pcd"]

a = np.random.rand(10)
b = np.random.rand(10)
print(a)
print(b)
