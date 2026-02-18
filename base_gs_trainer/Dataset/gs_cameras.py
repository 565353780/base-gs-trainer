import random

from tqdm import tqdm
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from camera_control.Module.camera_convertor import CameraConvertor

from base_gs_trainer.Data.gs_camera import GSCamera
from base_gs_trainer.Method.camera_utils import cameras_extent_from_list


class GSCameras:
    def __init__(
        self,
        colmap_data_folder_path: str,
        device: str='cuda:0',
        shuffle: bool=True,
    ) -> None:
        colmap_cameras = CameraConvertor.loadColmapDataFolder(colmap_data_folder_path)
        if len(colmap_cameras) == 0:
            print('[ERROR][Scene::__init__]')
            print('\t loadColmapDataFolder failed!')
            return

        n_cams = len(colmap_cameras)
        self.train_cameras: List[Optional[GSCamera]] = [None] * n_cams
        print('[INFO][Scene::__init__]')
        print('\t start loading gs cameras...')
        with ThreadPoolExecutor(max_workers=min(8, n_cams)) as executor:
            futures = {
                executor.submit(GSCamera, c, device=device,): i
                for i, c in enumerate(colmap_cameras)
            }
            with tqdm(total=n_cams, desc="Loading GSCamera") as pbar:
                for fut in as_completed(futures):
                    idx = futures[fut]
                    self.train_cameras[idx] = fut.result()
                    pbar.update(1)

        self.cameras_extent = cameras_extent_from_list(colmap_cameras)

        for i in range(n_cams):
            self.train_cameras[i].uid = i

        if shuffle:
            random.shuffle(self.train_cameras)
        return

    def __len__(self) -> int:
        return len(self.train_cameras)

    def __getitem__(self, idx: int):
        valid_idx = idx % len(self.train_cameras)
        return self.train_cameras[valid_idx]
