import torch
import numpy as np

from torch import nn

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
    prune_mask = ~valid_mask

    if getattr(gs, 'optimizer', None) is None:
        manualPruneGS(gs, valid_mask)
    else:
        if not hasattr(gs, 'tmp_radii'):
            gs.tmp_radii = None
        gs.prune_points(prune_mask)

    return True
