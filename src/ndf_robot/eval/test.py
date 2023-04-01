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


r = R.from_euler('xy', [-90, 90], degrees=True)
quat = r.as_quat()
print(quat)

a = np.array([0, 1, 2, 3, 4])
b = np.array([4, 3, 2, 1, 0])
c = np.vstack([a, b])
d = np.argmax(c, axis=0)
e = a[d==0]
f = a[d==1]
# valid object list for mug
obj_class = 'bottle'
shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(), obj_class + '_centered_obj_normalized')
avoid_shapenet_ids = bad_shapenet_bottles_ids_list
shapenet_id_list = [fn.split('_')[0] for fn in os.listdir(shapenet_obj_dir)]
valid_id_list = []
for i in shapenet_id_list:
    if i not in avoid_shapenet_ids:
        valid_id_list.append(i)

obj_model1 = osp.join(path_util.get_ndf_demo_obj_descriptions(), 'mug_centered_obj_normalized/28f1e7bc572a633cb9946438ed40eeb9/models/model_normalized.obj')
obj_model2 = osp.join(path_util.get_ndf_demo_obj_descriptions(), 'mug_centered_obj_normalized/586e67c53f181dc22adf8abaa25e0215/models/model_normalized.obj')
obj_model3 = BASE_DIR + '/descriptions/objects/bottle_centered_obj_normalized/' + valid_id_list[6] + '/models/model_normalized.obj'

mesh1 = trimesh.load(obj_model3, process=False)
pcd3 = mesh1.sample(5000)


dir_pcd_1 = BASE_DIR + '/data/scene_0.npz'
dir_pcd_2 = BASE_DIR + '/data/scene_1.npz'
pcd_1 = np.load(dir_pcd_1, allow_pickle=True)
pcd_2 = np.load(dir_pcd_2, allow_pickle=True)
pcd1 = pcd_1["pcd"]
pcd2 = pcd_2["pcd"]

pcd1 = pcd1 - np.mean(pcd1, axis=0)
pcd3 = pcd3 - np.mean(pcd3, axis=0)

ps.init()
ps.set_up_dir("z_up")
ps.register_point_cloud("pcd1", pcd1, radius=0.005, enabled=True)
ps.register_point_cloud("pcd2", pcd3, radius=0.005, enabled=True)
ps.show()
