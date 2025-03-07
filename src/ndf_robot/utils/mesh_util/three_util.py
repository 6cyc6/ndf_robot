import numpy as np
import trimesh
from mesh_to_sdf import mesh_to_voxels

from rndf_robot.utils import fork_pdb

def trimesh_show(np_pcd_list, color_list=None, show=True):
    if color_list is None:
        color_list = []
        for i in range(len(np_pcd_list)):
            color_list.append((np.random.rand(3) * 255).astype(np.int32).tolist() + [255])
    
    tpcd_list = []
    for i, pcd in enumerate(np_pcd_list):
        tpcd = trimesh.PointCloud(pcd)
        tpcd.colors = np.tile(color_list[i], (tpcd.vertices.shape[0], 1))

        tpcd_list.append(tpcd)
    
    scene = trimesh.Scene()
    scene.add_geometry(tpcd_list)
    if show:
        scene.show() 

    return scene


def trimesh_combine(mesh_files, mesh_poses, mesh_scales=None):
    meshes = []
    if mesh_scales is None:
        default = [1.0, 1.0, 1.0]
        mesh_scales = [default] * len(mesh_files)
    for i, mesh in enumerate(mesh_files):
        tmesh = trimesh.load(mesh, process=False)
        tmesh.apply_scale(mesh_scales[i])
        tmesh.apply_transform(mesh_poses[i])
        meshes.append(tmesh) 

    concat_mesh = trimesh.util.concatenate(meshes)
    return concat_mesh


def get_raster_points(voxel_resolution):
    points = np.meshgrid(
        np.linspace(-0.5, 0.5, voxel_resolution),
        np.linspace(-0.5, 0.5, voxel_resolution),
        np.linspace(-0.5, 0.5, voxel_resolution)
        # np.linspace(-0.55, 0.55, voxel_resolution),
        # np.linspace(-0.55, 0.55, voxel_resolution),
        # np.linspace(-0.55, 0.55, voxel_resolution)
    )
    points = np.stack(points)
    points = np.swapaxes(points, 1, 2)
    points = points.reshape(3, -1).transpose().astype(np.float32)
    return points


def scale_mesh(mesh, offset=None, scaling=None):
    if offset is None:
        vertices = mesh.vertices - mesh.bounding_box.centroid
    else:
        vertices = mesh.vertices - offset
    
    if scaling is None:
        vertices *= 1 / np.max(mesh.bounding_box.extents)
    else:
        vertices *= scaling

    scaled_mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces)
    return scaled_mesh


def get_occ(obj_mesh, VOXEL_RES, offset=None, scaling=None, sample_points=None):
    if sample_points is None:
        sample_points = get_raster_points(VOXEL_RES)

    if offset is None:
        vertices = obj_mesh.vertices - obj_mesh.bounding_box.centroid
    else:
        vertices = obj_mesh.vertices - offset

    if scaling is None:
        vertices *= 1 / np.max(obj_mesh.bounding_box.extents)
    else:
        vertices *= scaling

    obj_mesh = trimesh.Trimesh(vertices=vertices, faces=obj_mesh.faces)

    # get voxelized SDF and occupancies
    voxels_sdf = mesh_to_voxels(obj_mesh, VOXEL_RES, pad=False)
    occ = voxels_sdf <= 0.0

    return sample_points, occ.reshape(-1), voxels_sdf.reshape(-1)
