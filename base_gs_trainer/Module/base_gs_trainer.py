import torch

from abc import ABC, abstractmethod

from fused_ssim import fused_ssim

from base_trainer.Module.logger import Logger
from base_trainer.Module.base_trainer import BaseTrainer

from base_gs_trainer.Dataset.gs_cameras import GSCameras
from base_gs_trainer.Lib.lpipsPyTorch import lpips
from base_gs_trainer.Loss.l1 import l1_loss
from base_gs_trainer.Metric.psnr import psnr
from base_gs_trainer.Method.colmap_io import readColmapPcd
from base_gs_trainer.Method.general_utils import safe_state, colormap


class BaseGSTrainer(ABC):
    def __init__(
        self,
        colmap_data_folder_path: str='',
        device: str='cuda:0',
        save_result_folder_path: str='./output/',
        save_log_folder_path: str='./logs/',
        test_freq: int=10000,
        save_freq: int=10000,
    ) -> None:
        self.colmap_data_folder_path = colmap_data_folder_path

        # 优先设定默认 dtype，避免其他脚本设为 bfloat16 导致类型不匹配
        torch.set_default_dtype(torch.float32)
        self.device = device

        self.save_result_folder_path = save_result_folder_path
        self.save_log_folder_path = save_log_folder_path

        self.test_freq = test_freq
        self.save_freq = save_freq

        safe_state(silent=False)

        torch.autograd.set_detect_anomaly(False)

        self.scene = GSCameras(self.colmap_data_folder_path)

        pcd = readColmapPcd(self.colmap_data_folder_path)
        self.gaussians.create_from_pcd(pcd, self.scene.cameras_extent)

        self.gaussians.training_setup(self.opt)

        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')

        self.logger = Logger()

        BaseTrainer.initRecords(self)

        self.is_gt_logged = False
        return

    @abstractmethod
    def renderImage(self, viewpoint_cam) -> dict:
        '''
        return render_fastgs(viewpoint_cam, self.gaussians, self.pipe, self.background, self.opt.mult)
        '''
        pass

    @torch.no_grad
    def logStep(self, iteration: int, loss_dict: dict, render_image_num: int=5) -> bool:
        for key, value in loss_dict.items():
            self.logger.addScalar('Loss/' + key, value, iteration)

        self.logger.addScalar('Gaussian/total_points', self.gaussians.get_xyz.shape[0], iteration)
        self.logger.addScalar('Gaussian/scale', torch.mean(self.gaussians.get_scaling).detach().clone().cpu().numpy(), iteration)
        self.logger.addScalar('Gaussian/opacity', torch.mean(self.gaussians.get_opacity).detach().clone().cpu().numpy(), iteration)

        # Report test and samples of training set
        if iteration % self.test_freq == 0:
            torch.cuda.empty_cache()
            cameras = [self.scene[idx % len(self.scene)] for idx in range(5, 30, 5)]

            l1_test = 0.0
            psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0
            for idx, viewpoint in enumerate(cameras):
                render_pkg = self.renderImage(viewpoint)
                image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                if self.logger.isValid() and (idx < render_image_num):
                    self.logger.summary_writer.add_images("view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                    try:
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        self.logger.summary_writer.add_images("view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                    except:
                        pass

                    try:
                        rend_alpha = render_pkg['rend_alpha']
                        rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                        surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                        self.logger.summary_writer.add_images("view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                        self.logger.summary_writer.add_images("view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                        self.logger.summary_writer.add_images("view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                        rend_dist = render_pkg["rend_dist"]
                        rend_dist = colormap(rend_dist.cpu().numpy()[0])
                        self.logger.summary_writer.add_images("view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                    except:
                        pass


                    if not self.is_gt_logged:
                        self.logger.summary_writer.add_images("view_{}/GT".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                l1_test += l1_loss(image, gt_image).mean().double()
                psnr_test += psnr(image, gt_image).mean().double()
                ssim_test += fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()
                lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()

            psnr_test /= len(cameras)
            ssim_test /= len(cameras)
            lpips_test /= len(cameras)
            l1_test /= len(cameras)
            print("\n[ITER {}] Evaluating: L1 {} PSNR {}".format(iteration, l1_test, psnr_test))
            self.logger.addScalar('Eval/l1', l1_test, iteration)
            self.logger.addScalar('Eval/psnr', psnr_test, iteration)
            self.logger.addScalar('Eval/ssim', ssim_test, iteration)
            self.logger.addScalar('Eval/lpips', lpips_test, iteration)

            if not self.is_gt_logged:
                self.is_gt_logged = True

            torch.cuda.empty_cache()
        return True
