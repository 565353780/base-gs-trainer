import os

from typing import Type

from base_gs_trainer.Method.path import removeFile, createFileFolder


def loadGSByClass(
    model_cls: Type,
    gs_ply_file_path: str,
    sh_degree: int = 3,
):
    '''
    通用 GS ply 读取: 接受任意暴露 `__init__(sh_degree=...)` 与 `load_ply(path)`
    的 GaussianModel 类, 由各方法包传入自己的模型类型 (fast_gs / twod_gs / ...)。

    路径不存在时打印错误并返回 None, 与各包原 loadGS 行为一致。
    '''
    if not os.path.exists(gs_ply_file_path):
        print('[ERROR][io::loadGSByClass]')
        print('\t gs ply file not exist!')
        print('\t gs_ply_file_path:', gs_ply_file_path)
        return None

    gaussians = model_cls(sh_degree=sh_degree)
    gaussians.load_ply(gs_ply_file_path)
    return gaussians


def saveGS(
    gaussians,
    save_gs_ply_file_path: str,
    overwrite: bool = False,
) -> bool:
    '''
    通用 GS ply 写出: 只依赖 `save_ply(path)`, 与具体表达无关。

    目标已存在且 overwrite=False 时直接返回 True (跳过写入), 与各包原 saveGS 行为一致。
    '''
    if gaussians is None:
        print('[ERROR][io::saveGS]')
        print('\t gaussians is None!')
        return False

    if os.path.exists(save_gs_ply_file_path):
        if not overwrite:
            return True
        removeFile(save_gs_ply_file_path)

    createFileFolder(save_gs_ply_file_path)
    gaussians.save_ply(save_gs_ply_file_path)
    return True
