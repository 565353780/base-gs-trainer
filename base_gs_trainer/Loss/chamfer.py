import torch


def chamferLossFn(device: str='cuda:0'):
    if torch.cuda.is_available() and device != "cpu":
        from base_gs_trainer.Lib.chamfer3D.dist_chamfer_3D import chamfer_3DDist
        return chamfer_3DDist()

    from base_gs_trainer.Lib.chamfer3D.chamfer_python import distChamfer
    return distChamfer
