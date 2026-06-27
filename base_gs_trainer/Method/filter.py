import torch
import numpy as np

from torch import nn
from typing import List, Optional, Tuple

from camera_control.Method.filter import searchMainClusterPointMask


_GS_FIELD_KEYS = (
    '_xyz',
    '_features_dc',
    '_features_rest',
    '_opacity',
    '_scaling',
    '_rotation',
)

_GS_AUX_KEYS = (
    'max_radii2D',
    'xyz_gradient_accum',
    'xyz_gradient_accum_abs',
    'denom',
    'tmp_radii',
)


def manualPruneGS(gs, valid_mask: torch.Tensor) -> None:
    '''
    在 optimizer 尚未初始化的情况下, 直接对模型张量做切分,
    保持与 GaussianModel.prune_points 一致的语义。

    对 fast_gs / twod_gs 等不同表达均适用: 只处理实际存在且非空的属性,
    自动跳过某些表达中缺失的辅助张量 (如 2DGS 没有 xyz_gradient_accum_abs)。
    '''
    def _slice_param(param: torch.Tensor) -> nn.Parameter:
        new_data = param.data[valid_mask]
        new_param = nn.Parameter(new_data)
        new_param.requires_grad_(param.requires_grad)
        return new_param

    for key in _GS_FIELD_KEYS:
        param = getattr(gs, key, None)
        if isinstance(param, nn.Parameter):
            setattr(gs, key, _slice_param(param))

    for key in _GS_AUX_KEYS:
        tensor = getattr(gs, key, None)
        if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
            setattr(gs, key, tensor[valid_mask])

    return


def _applyGSValidMask(gs, valid_mask: torch.Tensor) -> None:
    '''
    按 valid_mask (True 表示保留) 原地裁剪 GS, 自动选择裁剪路径:
      - optimizer 尚未初始化: 直接切张量 (manualPruneGS);
      - optimizer 已初始化: 走 prune_points (mask 取反), 同步 optimizer state /
        densify 统计 / 可见性 / 法向定向等所有跟随张量。

    抽出此函数作为 removeFloatGSGeneric / removeOutlierGSGeneric 的公共裁剪出口,
    避免两套裁剪同步逻辑分叉。
    '''
    if getattr(gs, 'optimizer', None) is None:
        manualPruneGS(gs, valid_mask)
    else:
        if not hasattr(gs, 'tmp_radii'):
            gs.tmp_radii = None
        gs.prune_points(~valid_mask)

    return


def _cameraCenterAndBackDirection(
    cam,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    '''
    从相机对象抽取世界系相机中心 C 与后方向单位向量 b (= 视线方向的反方向,
    等价于相机的极坐标方向)。

    兼容两类相机来源:
      1. camera_control.Camera (拥有 R/t world2camera 约定):
         C = -R^T @ t, b = R^T @ [0,0,1] (与 getCamerasSphere / forwardDirection 一致)。
         GSCamera 会把原始 Camera 存在 `_cam`, 优先从这里取严格几何。
      2. 仅有 GS 渲染字段的相机 (GSCamera): 用 camera_center 作为 C,
         world_view_transform[:3,:3] 为行向量约定的 W2C 旋转, 其行 2 即相机 +Z 轴在
         世界系下的方向, 故 b = world_view_transform[:3,:3][2] (= R_w2c^T @ z 的转置约定)。

    返回 (C, b), 失败返回 None。
    '''
    src = getattr(cam, '_cam', None)
    if src is not None and hasattr(src, 'R') and hasattr(src, 't'):
        try:
            R = src.R.to(device=device, dtype=dtype)
            t = src.t.to(device=device, dtype=dtype).reshape(3)
            center = -(R.T @ t)
            z_cam = torch.tensor([0., 0., 1.], dtype=dtype, device=device)
            back = R.T @ z_cam
            return center, back
        except Exception:
            pass

    center_attr = getattr(cam, 'camera_center', None)
    wvt = getattr(cam, 'world_view_transform', None)
    if center_attr is not None and wvt is not None:
        try:
            center = center_attr.to(device=device, dtype=dtype).reshape(3)
            # world_view_transform 是行向量右乘约定的 W2C (= 列约定 W2C 的转置),
            # 其前 3x3 块的第 2 行即相机 +Z 轴在世界系下的单位方向。
            back = wvt[:3, :3].to(device=device, dtype=dtype)[2]
            return center, back
        except Exception:
            pass

    return None


def _collectCameraOrbitGeometry(
    camera_list: List,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    '''
    批量抽取相机中心与单位后方向。

    返回 (centers: (N,3), back_dirs: (N,3)), 任一相机无法解析或列表为空则返回 None。
    forward_dirs 由调用方按 -back_dirs 取得。
    '''
    if camera_list is None or len(camera_list) == 0:
        return None

    centers = []
    back_dirs = []
    for cam in camera_list:
        cb = _cameraCenterAndBackDirection(cam, device, dtype)
        if cb is None:
            return None
        center, back = cb
        centers.append(center)
        back_dirs.append(back)

    centers = torch.stack(centers, dim=0)
    back_dirs = torch.stack(back_dirs, dim=0)
    back_dirs = back_dirs / (torch.linalg.norm(back_dirs, dim=1, keepdim=True) + 1e-8)

    return centers, back_dirs


def _estimateCameraFocus(
    centers: torch.Tensor,
    forward_dirs: torch.Tensor,
    eps: float = 1e-6,
) -> Optional[torch.Tensor]:
    '''
    估计相机列表的关注中心 (focus): 与所有相机视线距离平方和最小的世界点。

    第一性原理 (与 CameraConvertor.getZAxisNormalizeTransform 的 Step 3 同源):
      每条视线为 {C_i + s * d_i}, 点 x 到该直线的平方距离的梯度给出投影矩阵
      P_i = I - d_i d_i^T, 最小二乘法方程:
          (sum_i P_i) x = sum_i P_i C_i
    当所有视线近似平行 (sum_i P_i 秩亏 / 病态) 时无解, 返回 None。
    '''
    n = centers.shape[0]
    if n < 2:
        return None

    device = centers.device
    dtype = centers.dtype
    I3 = torch.eye(3, dtype=dtype, device=device)

    A = torch.zeros((3, 3), dtype=dtype, device=device)
    b = torch.zeros((3,), dtype=dtype, device=device)
    for i in range(n):
        d = forward_dirs[i]
        P = I3 - d[:, None] @ d[None, :]
        A = A + P
        b = b + P @ centers[i]

    # 病态判据: A 为半正定, 最小特征值过小说明视线近似平行, focus 不可观测。
    try:
        eigvals = torch.linalg.eigvalsh(0.5 * (A + A.T))
    except Exception:
        return None
    if float(eigvals[0].item()) <= eps * max(1.0, float(n)):
        return None

    try:
        focus = torch.linalg.solve(A, b)
    except Exception:
        return None

    if not torch.all(torch.isfinite(focus)):
        return None

    return focus


def _estimateObservationSphere(
    centers: torch.Tensor,
    back_dirs: torch.Tensor,
    focus: torch.Tensor,
    spread_threshold: float = 0.0076,
) -> Optional[Tuple[torch.Tensor, float]]:
    '''
    以关注中心 focus 为球心, 估计能完全包围物体的相机观测球半径。

    鲁棒半径策略 (基于"人们围绕物体拍摄、物体被相机环绕包围"的假设):
        radius = max_i || C_i - focus ||
    即球心到所有相机位置的最大欧氏距离。由于物体被所有相机环绕在内侧,
    物体上任意真实表面点到关注中心的距离都不会超过最远相机到中心的距离,
    故该球必然完整包住物体, 只会裁掉比最远相机还远的离群/背景漂浮点,
    绝不会误删内部物体点。

    与 CameraConvertor.getCamerasSphere 同源的近似平行判据: 后方向的角散度
        spread = max_eig(sum_i (b_i - mean)(b_i - mean)^T) / N
    小于阈值 (约 sin^2(5°) ≈ 0.0076) 时, 相机近似平行、不构成对物体的环绕,
    包围球假设失效 -> 返回 None (严格按"估计不出半径则失败")。

    返回 (sphere_center, radius), radius>0; 否则 None。
    '''
    n = centers.shape[0]
    if n < 2:
        return None

    mean_back = back_dirs.mean(dim=0)
    centered_back = back_dirs - mean_back[None, :]
    M_back = centered_back.T @ centered_back
    M_back = 0.5 * (M_back + M_back.T)
    try:
        eig_back = torch.linalg.eigvalsh(M_back)
    except Exception:
        return None
    spread = float(eig_back[-1].item()) / max(1.0, float(n))
    if spread <= spread_threshold:
        return None

    # 球心到所有相机位置的最大距离: 保证物体完整落在包围球内, 不误删内部点。
    cam_dists = torch.linalg.norm(centers - focus[None, :], dim=1)  # (N,)
    radius = float(cam_dists.max().item())

    if not np.isfinite(radius) or radius <= 1e-8:
        return None

    return focus, radius


def removeFloatGSGeneric(gs, **kwargs) -> bool:
    '''
    基于 "自适应八叉树占用率 + bbox 有限膨胀" 的主簇提取剔除漂浮高斯。

    与具体 GS 表达无关: 仅依赖 `get_xyz` 读取点坐标, 并通过模型自身的
    `prune_points` (optimizer 已初始化) 或 manualPruneGS (optimizer 尚未初始化)
    完成裁剪。所有超参通过 kwargs 透传给 searchMainClusterPointMask。
    '''
    if gs is None:
        print('[ERROR][filter::removeFloatGSGeneric]')
        print('\t gs is None!')
        return False

    if gs._xyz is None or gs._xyz.numel() == 0:
        print('[ERROR][filter::removeFloatGSGeneric]')
        print('\t gs has no points!')
        return False

    n = gs.get_xyz.shape[0]
    if n < 4:
        return True

    try:
        with torch.no_grad():
            xyz = gs.get_xyz.detach()
            keep_np = searchMainClusterPointMask(xyz, **kwargs)
    except ValueError as e:
        print('[ERROR][filter::removeFloatGSGeneric]')
        print('\t invalid hyperparameter:', e)
        return False

    if keep_np.size != n or np.all(keep_np):
        return True

    device = xyz.device
    valid_mask = torch.from_numpy(keep_np).to(device=device, dtype=torch.bool)

    _applyGSValidMask(gs, valid_mask)

    return True


def removeOutlierGSGeneric(
    gs,
    camera_list: List,
    radius_scale: float = 1.0,
    spread_threshold: float = 0.0076,
) -> bool:
    '''
    基于相机观测球裁掉球外漂浮高斯。

    第一性原理: 一组围绕物体拍摄的相机, 其视线在最小二乘意义下汇聚于一个关注中心
    (球心)。基于"人们围绕物体拍摄、物体被相机完全环绕在内侧"的假设, 取球心到所有
    相机位置的最大距离作为半径, 物体必然完整落在该包围球内, 故只会裁掉比最远相机
    更远的背景/漂浮点, 绝不会误删内部物体点。该操作与渲染贡献/梯度无关,
    对已收敛模型稳健。

    与具体 GS 表达无关: 仅依赖 `get_xyz` 读取点坐标, 并复用 _applyGSValidMask 完成裁剪。

    Args:
        gs: 高斯模型 (需有 get_xyz / _xyz)。
        camera_list: 相机列表 (camera_control.Camera 或 GSCamera)。
        radius_scale: 观测球半径的安全余量倍数 (>1 放宽, 默认 1.0)。
        spread_threshold: 后方向角散度阈值, 低于则视为相机近似平行、半径不可观测。

    Returns:
        True: 球估计成功 (即使无点被裁也返回 True);
        False: 输入非法 / 相机几何不可观测 / 近似平行无法估计半径 / 裁剪后无点可留。
    '''
    if gs is None:
        print('[ERROR][filter::removeOutlierGSGeneric]')
        print('\t gs is None!')
        return False

    if gs._xyz is None or gs._xyz.numel() == 0:
        print('[ERROR][filter::removeOutlierGSGeneric]')
        print('\t gs has no points!')
        return False

    if camera_list is None or len(camera_list) < 2:
        print('[ERROR][filter::removeOutlierGSGeneric]')
        print('\t need at least 2 cameras to estimate observation sphere!')
        return False

    if radius_scale <= 0:
        print('[ERROR][filter::removeOutlierGSGeneric]')
        print('\t radius_scale must be positive, got:', radius_scale)
        return False

    with torch.no_grad():
        xyz = gs.get_xyz.detach()
        device = xyz.device
        dtype = xyz.dtype

        geom = _collectCameraOrbitGeometry(camera_list, device, dtype)
        if geom is None:
            print('[ERROR][filter::removeOutlierGSGeneric]')
            print('\t failed to extract camera orbit geometry!')
            return False
        centers, back_dirs = geom
        forward_dirs = -back_dirs

        focus = _estimateCameraFocus(centers, forward_dirs)
        if focus is None:
            print('[WARN][filter::removeOutlierGSGeneric]')
            print('\t cameras nearly parallel, focus not observable; skip.')
            return False

        sphere = _estimateObservationSphere(
            centers, back_dirs, focus, spread_threshold=spread_threshold,
        )
        if sphere is None:
            print('[WARN][filter::removeOutlierGSGeneric]')
            print('\t cameras nearly parallel, sphere radius not observable; skip.')
            return False

        sphere_center, sphere_radius = sphere
        keep_radius = sphere_radius * float(radius_scale)

        dist = torch.linalg.norm(xyz - sphere_center[None, :], dim=1)
        valid_mask = dist <= keep_radius

        if bool(torch.all(valid_mask).item()):
            return True

        if int(valid_mask.sum().item()) == 0:
            print('[WARN][filter::removeOutlierGSGeneric]')
            print('\t all GS would be pruned; skip to avoid emptying model.')
            return False

        _applyGSValidMask(gs, valid_mask)

    return True
