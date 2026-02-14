import math
import torch


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def getProjectionMatrix(fovX, fovY, znear: float=0.01, zfar: float=100.0):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getGSProjectionMatrix_from_intrinsics(
    fx: float, fy: float, cx: float, cy: float, 
    width: int, height: int,
    znear: float = 0.01, zfar: float = 100.0
) -> torch.Tensor:
    P = torch.zeros(4, 4)
    z_sign = 1.0

    P[0, 0] = 2.0 * fx / width
    P[1, 1] = 2.0 * fy / height

    # 找回丢失的主点偏移
    P[0, 2] = (2.0 * cx / width) - 1.0
    P[1, 2] = 1.0 - (2.0 * cy / height)  # 适配左下角cy定义 + 3DGS朝向

    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P
