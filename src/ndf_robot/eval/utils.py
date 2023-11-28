import math
import numpy as np

from scipy.spatial.transform import Rotation as R

path2dataset_ycb_16 = '/home/ikun/ycb_model_16k'
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0


def quat2eulers(q0: float, q1: float, q2: float, q3: float) -> tuple:
    """
    Compute yaw-pitch-roll Euler angles from a quaternion.

    Args
    ----
        q0: Scalar component of quaternion.
        q1, q2, q3: Vector components of quaternion.

    Returns
    -------
        (roll, pitch, yaw) (tuple): 321 Euler angles in radians
    """
    roll = math.atan2(
        2 * ((q2 * q3) + (q0 * q1)),
        q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2
    )  # radians
    pitch = math.asin(2 * ((q1 * q3) - (q0 * q2)))
    yaw = math.atan2(
        2 * ((q1 * q2) + (q0 * q3)),
        q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2
    )
    return roll, pitch, yaw


def matrix_from_pose(pos, quat):
    """
    homogeneous transformation matrix from transformation and quaternion
    :param pos: (3, ) numpy array
    :param quat: (4, ) numpy array
    :return: 4x4 numpy array
    """
    t = np.eye(4)
    t[:3, :3] = quat_to_mat(quat)
    pos = pos.reshape((1, 3))
    t[0:3, 3] = pos

    return t


def pose_from_matrix(t, stack=False):
    """
    transformation and quaternion from homogeneous transformation matrix
    :param stack: if stack pose and quaternion
    :param t: 4x4 numpy array
    :return: pos: (3, ) numpy array
             quat: (4, ) numpy array
    """
    pos = t[0:3, -1]
    r_mat = t[0:3, 0:3]
    quat = mat_to_quat(r_mat)

    if stack:
        return np.hstack([pos, quat])
    else:
        return pos, quat


def get_transform(pose_target, pose_origin):
    """
    Find transformation from original pose to target pose
    :param pose_target:
    :param pose_origin:

    """
    t = pose_target @ np.linalg.inv(pose_origin)
    pos, quat = pose_from_matrix(t)
    pose = np.hstack([pos, quat])
    return pose


def quat_to_mat(quat):
    # first change the order to use scipy package: scalar-last (x, y, z, w) format
    # id_ord = [1, 2, 3, 0]
    # quat = quat[id_ord]
    r = R.from_quat(quat)
    return r.as_matrix()


def mat_to_quat(mat):
    r = R.from_matrix(mat)
    quat = r.as_quat()
    # change order
    # id_ord = [3, 0, 1, 2]

    return quat
    # return quat[id_ord]


def quat2mat(quat):
    """Convert Quaternion to Euler Angles.  See rotation.py for notes"""
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))


def mat2euler(mat):
    """Convert Rotation Matrix to Euler Angles.  See rotation.py for notes"""
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(
        condition,
        -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
        -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]),
    )
    euler[..., 1] = np.where(
        condition, -np.arctan2(-mat[..., 0, 2], cy), -np.arctan2(-mat[..., 0, 2], cy)
    )
    euler[..., 0] = np.where(
        condition, -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]), 0.0
    )
    return euler


def quat2euler(quat):
    """Convert Quaternion to Euler Angles.  See rotation.py for notes"""
    return mat2euler(quat2mat(quat))
