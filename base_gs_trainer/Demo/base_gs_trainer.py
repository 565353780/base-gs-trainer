import sys
sys.path.append('../base-trainer')

from base_gs_trainer.Module.base_gs_trainer import BaseGSTrainer


class Trainer(BaseGSTrainer):
    def __init__(
        self,
        colmap_data_folder_path: str='',
        device: str='cuda:0',
        save_result_folder_path: str='./output/',
        save_log_folder_path: str='./logs/',
        test_freq: int=10000,
        save_freq: int=10000,
    ) -> None:
        # Set up command line argument parser
        parser = ArgumentParser(description="Training script parameters")
        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)
        args = parser.parse_args(sys.argv[1:])

        args.source_path = colmap_data_folder_path
        args.model_path = save_result_folder_path

        print("Optimizing " + args.model_path)

        self.dataset = lp.extract(args)
        self.opt = op.extract(args)
        self.pipe = pp.extract(args)

        self.gaussians = GaussianModel(self.dataset.sh_degree)

        BaseGSTrainer.__init__(
            self,
            colmap_data_folder_path=colmap_data_folder_path,
            device=device,
            save_result_folder_path=save_result_folder_path,
            save_log_folder_path=save_log_folder_path,
            test_freq=test_freq,
            save_freq=save_freq,
        )
        return

    def renderImage(self, viewpoint_cam) -> dict:
        return render_fastgs(viewpoint_cam, self.gaussians, self.pipe, self.background, self.opt.mult)

def demo():
    trainer = Trainer()
    return True
