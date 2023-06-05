from task_oriented_grasping.utils.path_utils import get_scene_path

import numpy as np
import polyscope as ps
import pyransac3d as pyrsc

pcd_dir = get_scene_path(file_name="1/0.npz")
data = np.load(pcd_dir, allow_pickle=True)
pcd1 = data["pcd_local"]
pcd2 = data["pcd_local2"]
pcd3 = data["pcd_local3"]
pcd4 = data["pcd_local4"]
pcd_table = data["table_local"]
obj_pos = data["obj_pos"]
obj_quat = data["obj_quat"]

# RANSAC
plane1 = pyrsc.Plane()
best_eq, best_inliers = plane1.fit(pcd_table, 0.01)
pcd_fit = pcd_table[best_inliers]
np.random.shuffle(pcd_fit)
pcd_fit = pcd_fit[:20000, :]

# circle1 = pyrsc.Circle()
# center, axis, radius, best_inliers = circle1.fit(pcd1, thresh=0.01, maxIteration=2000)
# print(center, axis, radius)
# pcd_fit_circle = pcd1[best_inliers]

# get points
n = pcd_fit.shape[0]
dist = -2 * pcd_fit @ pcd_fit.T
dist += np.sum(pcd_fit ** 2, -1).reshape((n, 1))
dist += np.sum(pcd_fit ** 2, -1).reshape((1, n))
temp = np.where(dist < 0.06, 1, 0)
temp = np.sum(temp, 0)
pts = np.where(temp > 1000)[0]
n_pts = pts.shape[0]
idx = np.random.randint(low=0, high=n_pts)
center = pcd_fit[pts[idx]]

dist = pcd_fit - center
dist = np.sum(dist ** 2, -1)
pts = np.where(dist < 0.06)[0]
n_pts = pts.shape[0]
idx = np.random.randint(low=0, high=n_pts, size=500)
pcd_query = pcd_fit[pts[idx]]

ps.init()
ps.set_up_dir("z_up")
ps.register_point_cloud("pcd_1", pcd1, radius=0.002, enabled=True)
# ps.register_point_cloud("table", pcd_table, radius=0.002, enabled=True)
ps.register_point_cloud("plane", pcd_fit, radius=0.002, enabled=True)
ps.register_point_cloud("query", pcd_query, radius=0.002, enabled=True)
# ps.register_point_cloud("circle", pcd_fit_circle, radius=0.002, enabled=True)

ps.show()

