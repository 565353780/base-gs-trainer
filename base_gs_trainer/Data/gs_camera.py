import torch

from camera_control.Module.camera import Camera

from base_gs_trainer.Method.graphics_utils import (
    focal2fov,
    getProjectionMatrix,
    getGSProjectionMatrix_from_intrinsics,
)


class GSCamera:
    def __init__(
        self,
        cam: Camera,
        device: str = "cuda:0",
    ) -> None:
        self._cam = cam
        self._cam.to(torch.float32, device)

        self.uid = id(cam)

        self.FoVx = float(focal2fov(cam.fx, cam.width))
        self.FoVy = float(focal2fov(cam.fy, cam.height))

        self.image_width = cam.width
        self.image_height = cam.height

        # 图像：(H, W, 3) -> (3, H, W)，并应用 mask（若有）
        self.original_image = cam.toMaskedImage().permute(2, 0, 1).clamp(0.0, 1.0).to(device)
        #self.original_image = cam.image.permute(2, 0, 1).clamp(0.0, 1.0).to(device)
        self.image_name = cam.image_id

        self.world_view_transform = cam.world2cameraColmap.transpose(0, 1).cuda()

        # self.projection_matrix = getProjectionMatrix(self.FoVx, self.FoVy).transpose(0, 1).cuda()
        # 2. 抛弃旧的基于 FOV 的 getProjectionMatrix
        self.projection_matrix = getGSProjectionMatrix_from_intrinsics(
            fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy,
            width=cam.width, height=cam.height,
        ).transpose(0, 1).cuda()

        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

        self.camera_center = cam.camera2worldColmap[:3, 3].cuda()

        self.R = self.world_view_transform[:3, :3]
        self.T = self.camera_center
        return
