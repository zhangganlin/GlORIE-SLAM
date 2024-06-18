import sys
import os
import random
import argparse
import numpy as np
import torch
import open3d as o3d
from torch.utils.data import Dataset, DataLoader
sys.path.append('.')
from src import config
from src.utils.common import update_cam
from src.utils.eval_traj import align_kf_traj
from src.utils.datasets import get_dataset
from src.utils.Printer import FontColor,TrivialPrinter
from tqdm import tqdm

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DepthImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.depth_files = sorted([os.path.join(root_dir, f) for f in os.listdir(
            root_dir) if f.startswith('depth_')])
        self.image_files = sorted([os.path.join(root_dir, f) for f in os.listdir(
            root_dir) if f.startswith('color_')])

        indices = []
        for depth_file in self.depth_files:
            base, ext = os.path.splitext(depth_file)
            index = int(base[-5:])
            indices.append(index)
        self.indices = indices

    def __len__(self):
        return len(self.depth_files)

    def __getitem__(self, idx):
        depth = np.load(self.depth_files[idx])
        image = np.load(self.image_files[idx])

        if self.transform:
            depth = self.transform(depth)
            image = self.transform(image)

        return depth, image


def generate_mesh_kf(config_path,rendered_path="rendered_every_keyframe",mesh_name_suffix="kf",printer=None):
    cfg = config.load_config(config_path, "configs/mono_point_slam.yaml")
    # define variables for dynamic query radius computation
    output = f"{cfg['data']['output']}/{cfg['setting']}/{cfg['scene']}"
    device = cfg['device']
    offline_video = f"{output}/video.npz"
    dataset = DepthImageDataset(root_dir=f'{output}/{rendered_path}')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    warmup = cfg['tracking']['warmup']
    scene_name = cfg["scene"]
    mesh_name = f'{scene_name}_{mesh_name_suffix}.ply'
    mesh_out_file = f'{output}/mesh/{mesh_name}'

    H, W, fx, fy, cx, cy = update_cam(cfg)
    scale = 1.0
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=5.0 * scale / 512.0,
        sdf_trunc=0.04 * scale,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    printer.print('Starting to integrate the mesh...',FontColor.MESH)
    # address the misalignment in open3d marching cubes
    compensate_vector = (-0.0 * scale / 512.0, 2.5 *
                         scale / 512.0, -2.5 * scale / 512.0)

    os.makedirs(f'{output}/mesh/mid_mesh', exist_ok=True)
    frame_reader = get_dataset(cfg, device=device)
    _,_,scene_scale,traj_est,traj_ref = align_kf_traj(offline_video,frame_reader,return_full_est_traj=True,printer=printer)
    traj = traj_est
    video = np.load(offline_video)

    v_idx_offset = 0
    for i, (depth, color) in tqdm(enumerate(dataloader),total=len(dataset)):
        index = dataset.indices[i]
        video_idx = i+warmup-1 + v_idx_offset
        while index!=int(video['timestamps'][video_idx]):
            assert(int(video['timestamps'][video_idx])<index)
            printer.print(f"Skip frame {int(video['timestamps'][video_idx])} (v_idx:{video_idx}) because rendered image and depth are not found.",
                          FontColor.MESH)
            v_idx_offset += 1
            video_idx = i+warmup-1 + v_idx_offset
            
        depth = depth[0].cpu().numpy() * scene_scale
        color = color[0].cpu().numpy()
        
        c2w = traj.poses_se3[video_idx]
        w2c = np.linalg.inv(c2w)

        depth = o3d.geometry.Image(depth.astype(np.float32))
        color = o3d.geometry.Image(
            np.array((np.clip(color, 0.0, 1.0)*255.0).astype(np.uint8)))

        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=1.0,
            depth_trunc=30,
            convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intrinsic, w2c)

    o3d_mesh = volume.extract_triangle_mesh()
    np.save(os.path.join(f'{output}/mesh',
            f'vertices_pos_{mesh_name_suffix}.npy'), np.asarray(o3d_mesh.vertices))
    o3d_mesh = o3d_mesh.translate(compensate_vector)

    o3d.io.write_triangle_mesh(mesh_out_file, o3d_mesh)
    printer.print(f"Final mesh file is saved: {mesh_out_file}",FontColor.INFO)
    printer.print('🕹️ Meshing finished.',FontColor.MESH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Configs for GlORIE-SLAM."
    )
    parser.add_argument(
        "config", type=str, help="Path to config file.",
    )
    args = parser.parse_args()
    generate_mesh_kf(args.config,printer=TrivialPrinter())
