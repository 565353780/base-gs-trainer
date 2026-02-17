import torch
import numpy as np
from typing import List

from camera_control.Module.camera import Camera


def cameras_extent_from_list(cam_list: List[Camera]):
    """从 camera_control Camera 列表计算 cameras_extent (radius)。"""
    if not cam_list:
        return 1.0
    centers = []
    for c in cam_list:
        pos = c.pos
        if torch.is_tensor(pos):
            pos = pos.detach().cpu().numpy()
        centers.append(pos.reshape(3, 1))
    centers = np.hstack(centers)
    center = np.mean(centers, axis=1, keepdims=True)
    dist = np.linalg.norm(centers - center, axis=0, keepdims=True)
    diagonal = float(np.max(dist))
    return diagonal * 1.1

def get_cameras_spatial_extent(cameras:List[Camera]):
    cam_centers = torch.cat([camera.pos.view(1, 3) for camera in cameras], dim=0)

    avg_cam_center = torch.mean(cam_centers, dim=0, keepdim=True)
    dist = torch.norm(cam_centers - avg_cam_center, dim=1, keepdim=True)

    half_diagonal = torch.max(dist)
    radius = half_diagonal * 1.1

    translate = -avg_cam_center

    return {"translate": translate, "radius": radius, "avg_cam_center": avg_cam_center}
