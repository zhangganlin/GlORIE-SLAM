import os
import time
import cv2
import numpy as np
import open3d as o3d
import torch

from colorama import Fore, Style
from multiprocessing.connection import Connection
from src.utils.common import (get_samples, get_samples_with_pixel_grad,
                        random_select, project_point3d_to_image_batch,)
from src.utils.datasets import get_dataset, load_mono_depth
from src.utils.Visualizer import Visualizer
from src.utils.Renderer import Renderer
from src.utils.eval_render import eval_kf_imgs, eval_imgs
from src.utils.Printer import Printer, FontColor
from src.neural_point import NeuralPointCloud, update_points_pos, get_proxy_render_depth
from src.depth_video import DepthVideo
from src.modules import conv_onet

import torchvision.transforms as Transforms
transform = Transforms.ToPILImage()
from skimage.color import rgb2gray
from skimage import filters
from scipy.interpolate import interp1d
from tqdm import tqdm

import wandb
from wandb import sdk as wandb_sdk
import functools
# print = functools.partial(print,flush=True)
print = tqdm.write


class Mapper(object):
    """
    Mapper thread.

    """
    def __init__(self, slam, pipe:Connection):
        self.cfg = slam.cfg
        self.printer:Printer = slam.printer
        if self.cfg['only_tracking']:
            return
        self.pipe = pipe
        self.output = slam.output
        self.verbose = slam.verbose
        self.renderer:Renderer = Renderer(self.cfg,slam)
        self.video:DepthVideo = slam.video
        self.npc = NeuralPointCloud(self.cfg, self.video)
        self.low_gpu_mem = True
        self.device = self.cfg['device']
        self.decoders:conv_onet.models.decoder.POINT = conv_onet.config.get_model(self.cfg).to(self.device)
        self.load_pretrained()

        self.logger:wandb_sdk.wandb_run.Run = slam.logger
        self.bind_npc_with_pose = self.cfg['pointcloud']['bind_npc_with_pose']

        self.use_dynamic_radius = self.cfg['pointcloud']['use_dynamic_radius']
        self.dynamic_r_add, self.dynamic_r_query = None, None
        self.radius_add_max = self.cfg['pointcloud']['radius_add_max']
        self.radius_add_min = self.cfg['pointcloud']['radius_add_min']
        self.radius_query_ratio = self.cfg['pointcloud']['radius_query_ratio']
        self.color_grad_threshold = self.cfg['pointcloud']['color_grad_threshold']
        self.fix_geo_decoder = self.cfg['mapping']['fix_geo_decoder']
        self.fix_color_decoder = self.cfg['mapping']['fix_color_decoder']
        self.mapping_pixels = self.cfg['mapping']['pixels']
        self.pixels_adding = self.cfg['mapping']['pixels_adding']
        self.pixels_based_on_color_grad = self.cfg['mapping']['pixels_based_on_color_grad']
        self.num_joint_iters = self.cfg['mapping']['iters']
        self.geo_iter_first = self.cfg['mapping']['geo_iter_first']
        self.iters_first = self.cfg['mapping']['iters_first']
        
        self.geo_iter_ratio = self.cfg['mapping']['geo_iter_ratio']
        self.mapping_window_size = self.cfg['mapping']['mapping_window_size']
        self.frustum_feature_selection = self.cfg['mapping']['frustum_feature_selection']
        self.keyframe_selection_method = self.cfg['mapping']['keyframe_selection_method']
        self.frustum_edge = self.cfg['mapping']['frustum_edge']
        self.save_rendered_image = self.cfg['mapping']['save_rendered_image']
        self.min_iter_ratio = self.cfg['mapping']['min_iter_ratio']

        self.pix_warping = self.cfg['mapping']['pix_warping']

        self.w_color_loss = self.cfg['mapping']['w_color_loss']
        self.w_pix_warp_loss = self.cfg['mapping']['w_pix_warp_loss']
        self.w_geo_loss = self.cfg['mapping']['w_geo_loss']

        self.keyframe_dict = []
        self.keyframe_list = []
        self.frame_reader = get_dataset(
            self.cfg, device=self.device)
        self.n_img = len(self.frame_reader)
        self.visualizer = Visualizer(vis_dir=os.path.join(self.output, 'mapping_vis'), renderer=self.renderer,
                                     verbose=self.verbose, device=self.device, logger=self.logger,
                                     total_iters=self.num_joint_iters,
                                     img_dir=os.path.join(self.output, 'rendered_image'))
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy
        self.render_depth_type = self.cfg['mapping']['render_depth']
        self.use_mono_to_complete = self.cfg['mapping']['use_mono_to_complete']
        self.init_idx = 0

        if self.cfg["mapping"]["save_depth"]:
            os.makedirs(f'{self.output}/semi_dense_depth/droid', exist_ok=True)
            os.makedirs(f'{self.output}/semi_dense_depth/project', exist_ok=True)
    def load_pretrained(self):
        convo_pretrained = self.cfg['mapping']['pretrained']
        convo_ckpt = torch.load(convo_pretrained,
                          map_location=self.device)
        middle_dict = {}
        fine_dict = {}
        for key, val in convo_ckpt['model'].items():
            if ('decoder' in key) and ('encoder' not in key):
                if 'coarse' in key:
                    key = key[8+7:]
                    middle_dict[key] = val
                elif 'fine' in key:
                    key = key[8+5:]
                    fine_dict[key] = val
        self.decoders.geo_decoder.load_state_dict(
            middle_dict, strict=False)
        self.printer.print(f'Load ConvONet pretrained checkpiont from {convo_pretrained}!',FontColor.INFO)

    def set_pipe(self, pipe):
        self.pipe = pipe

    def get_mask_from_c2w(self, c2w:torch.Tensor, depth_np:np.array):
        """
        Frustum feature selection based on current camera pose and depth image.
        Args:
            c2w (tensor): camera pose of current frame.
            depth_np (numpy.array): depth image of current frame. for each (x,y)<->(width,height)

        Returns:
            mask (tensor): mask for selected optimizable feature.
        """
        H, W, fx, fy, cx, cy, = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        points = self.npc.cloud_pos().cpu().numpy().reshape(-1, 3)

        c2w = c2w.cpu().numpy()
        w2c = np.linalg.inv(c2w)
        ones = np.ones_like(points[:, 0]).reshape(-1, 1)
        homo_vertices = np.concatenate(
            [points, ones], axis=1).reshape(-1, 4, 1)
        cam_cord_homo = w2c@homo_vertices
        cam_cord = cam_cord_homo[:, :3]
        K = np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)
        # flip the x-axis such that the pixel space is u from the left to right, v top to bottom.
        # without the flipping of the x-axis, the image is assumed to be flipped horizontally.
        cam_cord[:, 0] *= -1
        uv = K@cam_cord
        z = uv[:, -1:]+1e-5
        uv = uv[:, :2]/z
        uv = uv.astype(np.float32)

        remap_chunk = int(3e4)
        depths = []
        for i in range(0, uv.shape[0], remap_chunk):
            depths += [cv2.remap(depth_np,
                                 uv[i:i+remap_chunk, 0],
                                 uv[i:i+remap_chunk, 1],
                                 interpolation=cv2.INTER_LINEAR)[:, 0].reshape(-1, 1)]
        depths = np.concatenate(depths, axis=0)

        edge = self.frustum_edge  # crop here on width and height
        mask = (uv[:, 0] < W-edge)*(uv[:, 0] > edge) * \
            (uv[:, 1] < H-edge)*(uv[:, 1] > edge)

        zero_mask = (depths == 0)
        depths[zero_mask] = np.max(depths)

        mask = mask & (0 <= -z[:, :, 0]) & (-z[:, :, 0] <= depths+0.5)
        mask = mask.reshape(-1)

        return np.where(mask)[0].tolist()

    def keyframe_selection_overlap(self, gt_color:torch.Tensor, mono_depth:torch.Tensor, 
                                   c2w:torch.Tensor, keyframe_dict:list, k:int, N_samples=8, pixels=200):
        """
        Select overlapping keyframes to the current camera observation.

        Args:
            gt_color (tensor): ground truth color image of the current frame.
            mono_depth (tensor): ground truth depth image of the current frame.
            c2w (tensor): camera to world matrix (3*4 or 4*4 both fine).
            keyframe_dict (list): a list containing info for each keyframe.
            k (int): number of overlapping keyframes to select.
            N_samples (int, optional): number of samples/points per ray. Defaults to 16.
            pixels (int, optional): number of pixels to sparsely sample 
                from the image of the current camera. Defaults to 100.
        Returns:
            selected_keyframe_list (list): list of selected keyframe id.
        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        rays_o, rays_d, mono_depth, gt_color = get_samples(
            0, H, 0, W, pixels,
            fx, fy, cx, cy, c2w, mono_depth, gt_color, self.device, depth_filter=True)

        mono_depth = mono_depth.reshape(-1, 1)
        mono_depth = mono_depth.repeat(1, N_samples)
        t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
        near = mono_depth*0.8
        far = mono_depth+0.5
        z_vals = near * (1.-t_vals) + far * (t_vals)
        pts = rays_o[..., None, :] + \
            rays_d[..., None, :] * z_vals[..., :, None]
        vertices = pts.reshape(-1, 3).cpu().numpy()
        list_keyframe = []
        for keyframeid, keyframe in enumerate(keyframe_dict):
            video_idx = keyframe['video_idx']
            c2w = self.video.get_pose(video_idx,'cpu').numpy()
            c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)
            ones = np.ones_like(vertices[:, 0]).reshape(-1, 1)
            homo_vertices = np.concatenate(
                [vertices, ones], axis=1).reshape(-1, 4, 1)
            cam_cord_homo = w2c@homo_vertices
            cam_cord = cam_cord_homo[:, :3]
            K = np.array([[fx, .0, cx], [.0, fy, cy],
                         [.0, .0, 1.0]]).reshape(3, 3)
            # flip the x-axis such that the pixel space is u from the left to right, v top to bottom.
            # without the flipping of the x-axis, the image is assumed to be flipped horizontally.
            cam_cord[:, 0] *= -1
            uv = K@cam_cord
            z = uv[:, -1:]+1e-5
            uv = uv[:, :2]/z
            uv = uv.astype(np.float32)
            edge = 20
            mask = (uv[:, 0] < W-edge)*(uv[:, 0] > edge) * \
                (uv[:, 1] < H-edge)*(uv[:, 1] > edge)
            mask = mask & (z[:, :, 0] < 0)
            mask = mask.reshape(-1)
            percent_inside = mask.sum()/uv.shape[0]
            list_keyframe.append(
                {'id': keyframeid, 'percent_inside': percent_inside})

        list_keyframe = sorted(
            list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
        selected_keyframe_list = [dic['id']
                                  for dic in list_keyframe if dic['percent_inside'] > 0.00]
        selected_keyframe_list = list(np.random.permutation(
            np.array(selected_keyframe_list))[:k])
        return selected_keyframe_list

    def get_c2w_and_depth(self,video_idx:int, idx:int, mono_depth:torch.Tensor, print_info:bool=False):
        """
        Given the index of keyframe in depth_video, fetch the estimated pose and depth

        Args:
            video_idx (int): index of the keyframe in depth_video
            idx (int): index (timestamp) of the keyframe in the dataset
            mono_depth (tensor, [H,W]): monocular depth prior
            print_info (bool): whether to print out the number and ratio of valid depth pixels 

        Returns:
            c2w (tensor, [4,4]): estimated pose of the selected keyframe
            mono_depth_wq (tensor, [H,W]): mono depth prior after scale and shift alignment
            est_droid_depth (tensor, [H,W]): esitmated depth map from the tracker
        """

        est_droid_depth, valid_depth_mask, c2w = self.video.get_depth_and_pose(video_idx,self.device)
        if print_info:
            total_number = (valid_depth_mask.shape[0]*valid_depth_mask.shape[1])
            valid_number = valid_depth_mask.sum().item()
            self.printer.print(f"Number of pixels with valid droid depth: {valid_number}/{total_number} ({100*valid_number/total_number:0.2f}%)",
                               FontColor.MAPPER)
        if valid_depth_mask.sum() < 100:
            self.printer.print(f"Skip mapping frame {idx} because the number of valid depth is not enough: ({valid_depth_mask.sum().itme()}).",
                               FontColor.MAPPER)                
            return None, None, None
        est_droid_depth[~valid_depth_mask] = 0
        c2w[:3, 1:3] *= -1
        mono_valid_mask = mono_depth < (mono_depth.mean()*3)
        valid_mask = mono_valid_mask*valid_depth_mask
        cur_wq = self.video.get_depth_scale_and_shift(video_idx,mono_depth,est_droid_depth, valid_mask)
        c2w = c2w.to(self.device)
        mono_depth_wq = mono_depth * cur_wq[0] + cur_wq[1]
        return c2w, mono_depth_wq, est_droid_depth
    
    def anchor_points(self, anchor_depth:torch.Tensor, cur_gt_color:torch.Tensor, 
                      cur_c2w:torch.Tensor, cur_video_idx:int):
        """
        Anchor neural points to the neural pointcloud

        Args:
            anchor_depth (tensor, [N]): depth of the pixels which would be anchored
            cur_gt_color (tensor, [N,3]): color of the pixels which would be anchored
            cur_c2w (tensor, [4,4]): pose of the frame which those pixels from
            cur_video_idx (int): the index of the frame in depth_video

        Returns:
            frame_pts_add (int): total number of points added.
        """
        edge = 0
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        anchor_mask = anchor_depth>0
        gt_color = cur_gt_color.to(self.device)
        add_pts_num = self.pixels_adding

        batch_rays_o, batch_rays_d, batch_anchor_depth, batch_gt_color, i, j = get_samples(
            edge, H-edge, edge, W-edge, add_pts_num,
            fx, fy, cx, cy, cur_c2w, anchor_depth, gt_color, self.device, depth_filter=True, return_index=True,mask=anchor_mask)

        frame_pts_add = 0
        _ = self.npc.add_points(cur_video_idx)
        pts_add = self.npc.add_neural_points(batch_rays_o, batch_rays_d, batch_anchor_depth, batch_gt_color,
                                        cur_video_idx, i,j,
                                        dynamic_radius=self.dynamic_r_add[j, i] if self.use_dynamic_radius else None)
        frame_pts_add += pts_add

        if self.pixels_based_on_color_grad > 0:

            batch_rays_o, batch_rays_d, batch_anchor_depth, batch_gt_color, i, j = get_samples_with_pixel_grad(
                edge, H-edge, edge, W-edge, self.pixels_based_on_color_grad,
                H, W, fx, fy, cx, cy, cur_c2w, anchor_depth, gt_color, self.device,
                anchor_mask,
                depth_filter=True, return_index=True)
            pts_color_add = self.npc.add_neural_points(batch_rays_o, batch_rays_d, batch_anchor_depth, batch_gt_color,
                                            cur_video_idx, i,j,
                                            is_pts_grad=True, dynamic_radius=self.dynamic_r_add[j, i] if self.use_dynamic_radius else None)
            frame_pts_add += pts_color_add
        self.printer.print(f'{frame_pts_add} locations to add points.',FontColor.PCL)
        return frame_pts_add

    def pix_warping_loss(self,batch_rays_o,batch_rays_d,depth,c2ws,
                         fx,fy,cx,cy,W,H,
                         frame_indices,indices_tensor,
                         img_gt_colors,batch_gt_color):
        """
        pixel warping loss in the paper

        Args:
            batch_rays_o (tensor, [N,3]): the origin of unprojected rays of sampled pixels 
            batch_rays_d (tensor, [N,3]): the direction of unprojected rays of sampled pixels 
            depth (tensor, [N]): the depth of sampled pixels 
            c2ws (tensor, [M,4,4]): the poses of frames where the pixels would be warped to
            fx,fy,cx,cy (float): camera intrinsic params
            W,H (int): camera width and height
            frame_indices (tensor, [M]): indices of frames where the pixels would be warped to
            indices_tensor (tensor, [N]): indices of frames where the sampled pixels are from
            img_gt_colors (tensor, [M,H,W]): colors of frames where the pixels would be warped to
            batch_gt_color (tensor, [N,3]): the color of sampled pixels 

        Returns:
            frame_pts_add (int): total number of points added.
        """
        pix_warping_edge = 5
        pixel_3d_pts = (batch_rays_o + batch_rays_d * depth[:,None])
        pixel_3d_pts = pixel_3d_pts.float()
        uv, z = project_point3d_to_image_batch(
            c2ws, pixel_3d_pts.view(-1, 3, 1), fx, fy, cx, cy, self.device)

        uv = uv.view(1, pixel_3d_pts.shape[0], c2ws.shape[0], 2)  # [1, pn, Cn, 2]

        mask = (
            (uv[0,:, :, 0] < W - pix_warping_edge)
            * (uv[0,:, :, 0] > pix_warping_edge)
            * (uv[0,:, :, 1] < H - pix_warping_edge)
            * (uv[0,:, :, 1] > pix_warping_edge)
        )  # [Pn, Cn]

        mask = mask & (z.view(pixel_3d_pts.shape[0], c2ws.shape[0], 1)[:, :, 0] < 0)
        mask = mask & (frame_indices[None, :] != indices_tensor[:, None])

        # Only enable pixel warping loss when the pixels appears in at least 4 keyframes
        mask[mask.sum(dim=1) < 4] = False 

        windows_reproj_idx = uv.permute(2, 1, 0, 3)  # Cn, pn, 1, 2
        windows_reproj_idx[..., 0] = windows_reproj_idx[..., 0] / W * 2.0 - 1.0
        windows_reproj_idx[..., 1] = windows_reproj_idx[..., 1] / H * 2.0 - 1.0


        # img_gt_colors [cn,height,width,3]    
        windows_reproj_gt_color = torch.nn.functional.grid_sample(
            img_gt_colors.permute(0, 3, 1, 2).float(),              
            windows_reproj_idx,
            padding_mode="border",
            align_corners=False
        ).permute(2, 0, 3, 1)  # [Pn, cn, 1, 3]


        tmp_windows_reproj_gt_color = windows_reproj_gt_color[:,:,0,:]
        tmp_batch_gt_color = batch_gt_color

        warp_loss = torch.nn.functional.smooth_l1_loss(tmp_windows_reproj_gt_color[mask],
                    tmp_batch_gt_color.unsqueeze(1).repeat(1,c2ws.shape[0],1)[mask], beta=0.1) * 1.0
        return warp_loss

    def optimizer_update_one_step(self,optimizer:torch.optim.Adam,
                                  cur_stage,cur_sub_stage,
                                  optimize_frame_dict,pixs_per_image,
                                  cur_r_query,keyframe_dict,npc_geo_feats,npc_col_feats):
        """
        optimize one step for the given Adam optimizer by back-propogation

        Args:
            cur_stage: "init" | "stage"
            cur_sub_stage: "geometry" | "color"
            optimize_frame_dict (list(dict)): list of dict of frame which needs to be optimized, 
                                              each frame dict stores frame-related data
            pixs_per_image (int): number of sampled data for each image
            cur_r_query (tensor [H,W]): query radius for each pixel
            keyframe_dict (list(dict)): list of dict of all previous keyframes
            npc_geo_feats, npc_col_feats (tensor): geometry and color features of the neural point cloud 
        Returns:
            geo_loss,color_loss,pix_warp_loss: the three type of losses
            depth_mask: where the rendered depth has valid number
        """


        optimizer.param_groups[0]['lr'] = self.cfg['mapping'][cur_stage][cur_sub_stage]['decoders_lr']
        optimizer.param_groups[1]['lr'] = self.cfg['mapping'][cur_stage][cur_sub_stage]['geometry_lr']
        optimizer.param_groups[2]['lr'] = self.cfg['mapping'][cur_stage][cur_sub_stage]['color_lr']

        optimizer.zero_grad()
        batch_rays_d_list, batch_rays_o_list, batch_render_depth_list, batch_gt_color_list, batch_r_query_list = [],[],[],[],[]
        c2w_list, img_gt_color_list, indices_tensor, frame_indices = [],[],[],[]

        for frame_dict in optimize_frame_dict:
            frame = frame_dict["frame"]
            render_depth = frame_dict["render_depth"]
            render_mask = frame_dict["render_mask"]
            c2w = frame_dict["c2w"]
            gt_color = frame_dict["gt_color"]
            edge = 0
            batch_rays_o, batch_rays_d, batch_render_depth, batch_gt_color, i, j = get_samples(
                edge, self.H-edge, edge, self.W-edge, pixs_per_image, 
                self.fx, self.fy, self.cx, self.cy, c2w, render_depth, gt_color, self.device, 
                depth_filter=True, 
                return_index=True, mask=render_mask)
            batch_rays_o_list.append(batch_rays_o.float())
            batch_rays_d_list.append(batch_rays_d.float())
            batch_render_depth_list.append(batch_render_depth.float())
            batch_gt_color_list.append(batch_gt_color.float())
                
            if self.pix_warping:
                if c2w.shape[0]==4:
                    c2w_list.append(c2w)
                elif c2w.shape[0]==3:
                    bottom = torch.tensor([0, 0, 0, 1.0], device=self.device).reshape(1, 4) 
                    c2w_homo = torch.cat([c2w, bottom], dim=0)
                    c2w_list.append(c2w_homo)
                else:
                    raise NotImplementedError
        
                img_gt_color_list.append(gt_color)

            if self.use_dynamic_radius:
                if frame == -1:
                    batch_r_query_list.append(cur_r_query[j, i])
                else:
                    r_query = keyframe_dict[frame]['dynamic_r_query']/3.0*render_depth
                    batch_r_query_list.append(r_query[j, i])
            
            # log frame idx of pixels
            frame_indices_tensor = torch.full(
                (i.shape[0],), frame, dtype=torch.long, device=self.device)
            indices_tensor.append(frame_indices_tensor)
            frame_indices.append(frame)

        batch_rays_d = torch.cat(batch_rays_d_list)
        batch_rays_o = torch.cat(batch_rays_o_list)
        batch_render_depth = torch.cat(batch_render_depth_list)
        batch_gt_color = torch.cat(batch_gt_color_list)

        if self.pix_warping:
            img_gt_colors = torch.stack(img_gt_color_list).to(self.device)
            c2ws = torch.stack(c2w_list, dim=0)

        r_query_list = torch.cat(
            batch_r_query_list) if self.use_dynamic_radius else None

        with torch.no_grad():
            inside_mask = batch_render_depth <= torch.minimum(
                10*batch_render_depth.median(), 1.2*torch.max(batch_render_depth))

        batch_rays_d, batch_rays_o = batch_rays_d[inside_mask], batch_rays_o[inside_mask]
        batch_render_depth, batch_gt_color = batch_render_depth[inside_mask], batch_gt_color[inside_mask]

        if self.use_dynamic_radius:
            r_query_list = r_query_list[inside_mask]
        ret = self.renderer.render_batch_ray(self.npc, self.decoders, batch_rays_d, batch_rays_o, self.device, self.stage,
                                                gt_depth=batch_render_depth, npc_geo_feats=npc_geo_feats,
                                                npc_col_feats=npc_col_feats,
                                                is_tracker=False,
                                                cloud_pos=self.cloud_pos_tensor,
                                                dynamic_r_query=r_query_list)
        depth, uncertainty, color, valid_ray_mask, valid_ray_count = ret

        depth_mask = (batch_render_depth > 0) & (~torch.isnan(depth))
        geo_loss = torch.abs(
            batch_render_depth[depth_mask]-depth[depth_mask]).sum()
        loss = geo_loss*self.w_geo_loss
        
        indices_tensor = torch.cat(indices_tensor, dim=0)[inside_mask]
        frame_indices = torch.tensor(frame_indices).long().to(self.device)

        color_loss = None
        if self.stage == 'color':
            color_loss = torch.abs(batch_gt_color[depth_mask] - color[depth_mask]).sum()
            loss += self.w_color_loss*color_loss
        
        pix_warp_loss = None
        if self.pix_warping:
            pix_warp_loss = self.pix_warping_loss(batch_rays_o,batch_rays_d,depth,c2ws,
                                                        self.fx,self.fy,self.cx,self.cy,self.W,self.H,
                                                        frame_indices, indices_tensor,img_gt_colors,batch_gt_color)
            loss += self.w_pix_warp_loss*(pix_warp_loss.sum())

        loss.backward(retain_graph=False)
        optimizer.step()
        optimizer.zero_grad()

        return geo_loss,color_loss,pix_warp_loss,depth_mask

    def optimize_map(self, num_joint_iters, cur_idx, cur_depth, cur_gt_color, cur_gt_depth, cur_mono_depth,
                     cur_droid_depth,frame_pts_add,
                     keyframe_dict, keyframe_list, cur_c2w, init, color_refine=False):
        """
        Optimize the map given a new keyframe. Sample pixels from selected keyframes, then optimize scene representation.

        Args:
            num_joint_iters (int): number of mapping iterations.
            cur_idx (int): the index of current frame
            cur_depth (tensor): proxy depth of the current camera
            cur_gt_color (tensor): gt_color image of the current camera.
            cur_gt_depth, cur_mono_depth, cur_droid_depth (tensor): 
                three different depth maps of the current camera, for visualization only.
            frame_pts_add (int): number of new anchored points in the current frame, 
                use this to determine how many optimization iters needed
            keyframe_dict (list): list of keyframes info dictionary.
            keyframe_list (list): list of keyframe index.
            cur_c2w (tensor): the estimated camera to world matrix of current frame. 
            init (bool): whether the current frame is the first mapping frame.
            color_refine (bool): whether to do color refinement.
        """
        cur_r_query = self.dynamic_r_query/3.0*cur_depth
        cur_mask  = cur_depth > 0       
        if len(keyframe_dict) == 0:
            optimize_frame = []
        else:
            if self.keyframe_selection_method == 'global':
                num = self.mapping_window_size-2
                optimize_frame = random_select(len(self.keyframe_dict)-1, num)
            elif self.keyframe_selection_method == 'overlap':
                num = self.mapping_window_size-2
                optimize_frame = self.keyframe_selection_overlap(
                    cur_gt_color, cur_depth, cur_c2w, keyframe_dict[:-1], num)        

        # add the last keyframe and the current frame(use -1 to denote)
        if len(keyframe_list) > 0:
            optimize_frame = optimize_frame + [len(keyframe_list)-1]
        optimize_frame += [-1]
        optimize_frame_dict = []
        self.printer.print("Projecting pointcloud to keyframes ...",FontColor.PCL)
        for frame in optimize_frame:
            if frame != -1:
                mono_depth = keyframe_dict[frame]['mono_depth'].to(self.device)
                gt_color = keyframe_dict[frame]['color'].to(self.device)
                video_idx = keyframe_dict[frame]['video_idx']
                idx = keyframe_dict[frame]['idx']
                c2w,mono_depth, droid_depth = self.get_c2w_and_depth(video_idx,idx,mono_depth)
                if c2w is None:
                    continue
                if self.render_depth_type == "proxy":
                    render_depth = get_proxy_render_depth(self.npc, self.cfg, c2w.clone(),
                                                           droid_depth, mono_depth, 
                                                           self.device,use_mono_to_complete=self.use_mono_to_complete)
                    render_mask = render_depth > 0
                elif self.render_depth_type == "mono":
                    render_depth = mono_depth
                    render_mask = torch.ones_like(mono_depth,dtype=torch.bool,device=mono_depth.device)
            else:
                if color_refine:
                    continue
                render_depth = cur_depth
                render_mask = cur_mask
                gt_color = cur_gt_color.to(self.device)
                c2w = cur_c2w
            optimize_frame_dict.append({"frame":frame, "render_depth":render_depth, 
                                        "render_mask":render_mask, "gt_color":gt_color,"c2w":c2w})

        pixs_per_image = self.mapping_pixels//len(optimize_frame)

        # clone all point feature from shared npc, (N_points, c_dim)
        npc_geo_feats = self.npc.get_geo_feats()
        npc_col_feats = self.npc.get_col_feats()
        self.cloud_pos_tensor = self.npc.cloud_pos()
        
        if self.frustum_feature_selection:  # required if not color_refine
            pcl_indices = self.get_mask_from_c2w(cur_c2w, cur_depth.cpu().numpy())
            geo_pcl_grad = npc_geo_feats[pcl_indices].requires_grad_(True)
            color_pcl_grad = npc_col_feats[pcl_indices].requires_grad_(True)
        else:
            geo_pcl_grad = npc_geo_feats.requires_grad_(True)
            color_pcl_grad = npc_col_feats.requires_grad_(True)
            
        decoders_para_list = []
        if not self.fix_geo_decoder:
            decoders_para_list += list(
                self.decoders.geo_decoder.parameters())
        if not self.fix_color_decoder:
            decoders_para_list += list(
                self.decoders.color_decoder.parameters())

        optim_para_list = [{'params': decoders_para_list, 'lr': 0},
                           {'params': [geo_pcl_grad], 'lr': 0},
                           {'params': [color_pcl_grad], 'lr': 0}]

        optimizer = torch.optim.Adam(optim_para_list)

        if not init and not color_refine:
            num_joint_iters = np.clip(int(num_joint_iters*frame_pts_add/300), int(
                self.min_iter_ratio*num_joint_iters), 2*num_joint_iters)

        for joint_iter in range(num_joint_iters):
            
            if self.frustum_feature_selection:
                npc_geo_feats[pcl_indices] = geo_pcl_grad
                npc_col_feats[pcl_indices] = color_pcl_grad
            else:
                npc_geo_feats = geo_pcl_grad  # all feats
                npc_col_feats = color_pcl_grad

            if joint_iter <= (self.geo_iter_first if init else int(num_joint_iters*self.geo_iter_ratio)):
                self.stage = 'geometry'
            else:
                self.stage = 'color'
            cur_stage = 'init' if init else 'stage'
            cur_sub_stage = 'color' if color_refine else self.stage

            tic = time.perf_counter()

            geo_loss,color_loss,forward_reproj_loss,depth_mask = self.optimizer_update_one_step(
                                           optimizer,cur_stage,cur_sub_stage,optimize_frame_dict,
                                           pixs_per_image,cur_r_query,keyframe_dict,
                                           npc_geo_feats,npc_col_feats)

            # put selected and updated params back to npc
            npc_geo_feats, npc_col_feats = npc_geo_feats.detach(), npc_col_feats.detach()

            toc = time.perf_counter()
            if self.logger is None:
                if joint_iter % 100 == 0:
                    info = f"iter: {joint_iter}, geo_loss: {geo_loss.item():0.6f}"
                    if self.stage == 'color':
                        info += f", color_loss: {color_loss.item():0.6f}"
                    if self.pix_warping and forward_reproj_loss.sum().item() >= 0:
                        info += f", pix_warp_loss: {forward_reproj_loss.sum().item():0.6f}"
                    self.printer.print(info,FontColor.MAPPER)

            if joint_iter == num_joint_iters-1:
                self.printer.print(f'idx: {cur_idx}, time: {toc - tic:0.6f}, geo_loss_pixel: {(geo_loss.item()/depth_mask.sum().item()):0.6f}, color_loss_pixel: {(color_loss.item()/depth_mask.sum().item()):0.4f}',
                                   FontColor.MAPPER)
                if self.logger:
                    self.logger.log({'idx_map': int(cur_idx), 'time': float(f'{toc - tic:0.6f}'),
                                'geo_loss_pixel': float(f'{(geo_loss.item()/depth_mask.sum().item()):0.6f}'),
                                'color_loss_pixel': float(f'{(color_loss.item()/depth_mask.sum().item()):0.6f}'),
                                'pts_total': self.npc.index_ntotal(),
                                'num_joint_iters': num_joint_iters})


        if (not color_refine) and (not self.cfg['silence']):
            self.visualizer.vis(cur_idx, num_joint_iters-1, cur_gt_depth, 
                                cur_depth, cur_droid_depth, cur_mono_depth,
                                cur_gt_color, 
                                cur_c2w, self.npc, self.decoders,
                                npc_geo_feats, npc_col_feats, self.cfg, self.printer,
                                freq_override=True if init else False,
                                dynamic_r_query=cur_r_query,
                                cloud_pos=self.cloud_pos_tensor,
                                cur_total_iters=num_joint_iters, save_rendered_image=self.save_rendered_image)

        if self.frustum_feature_selection:
            self.npc.update_geo_feats(geo_pcl_grad, indices=pcl_indices)
            self.npc.update_col_feats(color_pcl_grad, indices=pcl_indices)
        else:
            self.npc.update_geo_feats(npc_geo_feats.detach().clone())
            self.npc.update_col_feats(npc_col_feats.detach().clone())

        self.printer.print('Mapper has updated point features.',FontColor.MAPPER)

        return

    def mapping_keyframe(self, idx, video_idx, mono_depth, 
                         outer_joint_iters, num_joint_iters,
                         gt_color, gt_depth, init=False, color_refine=False):
        """
        Mapping. 1. deform neural point cloud according the updated poses and depth.
                 2. get the proxy depth map
                 3. do optimization iteration                 
        Args:
            idx (int): the index of current keyframe in the dataset
            video_idx (int): the index of current keyframe in the depth_video
            mono_depth (tensor): mono depth prior
            outer_joint_iters, num_joint_iters (int): number of iters
            gt_color (tensor): gt_color image of the current camera.
            gt_depth (tensor): gt depth of the current camera, for visualization only.
            init (bool): whether the current frame is the first mapping frame.
            color_refine (bool): whether to do color refinement.
        Returns:
            False if cannot get valid estimated camera pose, otherwise True 
        """
        if self.bind_npc_with_pose:
            self.printer.print("Updating pointcloud position ...",FontColor.PCL)
            update_points_pos(self.npc, self.video)

        cur_c2w,depth_wq, droid_depth = self.get_c2w_and_depth(video_idx,idx,mono_depth,print_info=True)
        if cur_c2w is None:
            return False

        if self.render_depth_type == "proxy":
            anchor_depth = droid_depth.clone()
            anchor_depth_invalid = (anchor_depth==0)
            anchor_depth[anchor_depth_invalid] = depth_wq[anchor_depth_invalid]
        elif self.render_depth_type == "mono":
            anchor_depth = depth_wq.clone()
        self.dynamic_r_add = self.dynamic_r_add/3.0 * anchor_depth

        frame_pts_add = 0
        if not color_refine:
            frame_pts_add = self.anchor_points(anchor_depth, gt_color, cur_c2w, video_idx)
        
        if self.render_depth_type == "proxy":
            render_depth = get_proxy_render_depth(self.npc, self.cfg, cur_c2w.clone(), 
                                                  droid_depth, depth_wq, self.device, idx, 
                                                  use_mono_to_complete=self.use_mono_to_complete)
        elif self.render_depth_type == "mono":
            render_depth = depth_wq

        if color_refine:
            self.dynamic_r_query = torch.load(f'{self.output}/dynamic_r_frame/r_query_{idx:05d}.pt', map_location=self.device)

        for _ in range(outer_joint_iters):
            self.optimize_map(num_joint_iters, idx, render_depth, gt_color, gt_depth,
                              depth_wq, droid_depth, frame_pts_add,
                              self.keyframe_dict, self.keyframe_list, cur_c2w, init, 
                              color_refine=color_refine)
        return True
    
    def run(self):
        """
        Trigger mapping process, get estimated pose and depth from tracking process,
        send continue signal to tracking process when the mapping of the current frame finishes.  
        """
        cfg = self.cfg
        init = True
        while (1):
            frame_info = self.pipe.recv()
            idx = frame_info['timestamp']
            video_idx = frame_info['video_idx']
            is_finished = frame_info['end']
            if is_finished:
                break

            if self.verbose:
                self.printer.print(f"\nMapping Frame {idx} ...", FontColor.MAPPER)
            
            _, gt_color, gt_depth, _= self.frame_reader[idx]
            mono_depth_input = load_mono_depth(idx,self.cfg)
            
            gt_color = gt_color.to(self.device).squeeze(0).permute(1,2,0)
            gt_depth = gt_depth.to(self.device)
            mono_depth = mono_depth_input.to(self.device)

            if self.use_dynamic_radius:
                ratio = self.radius_query_ratio
                intensity = rgb2gray(gt_color.cpu().numpy())
                grad_y = filters.sobel_h(intensity)
                grad_x = filters.sobel_v(intensity)
                color_grad_mag = np.sqrt(grad_x**2 + grad_y**2)  # range 0~1
                color_grad_mag = np.clip(
                    color_grad_mag, 0.0, self.color_grad_threshold)  # range 0~1
                fn_map_r_add = interp1d([0, 0.01, self.color_grad_threshold], [
                                        self.radius_add_max, self.radius_add_max, self.radius_add_min])
                fn_map_r_query = interp1d([0, 0.01, self.color_grad_threshold], [
                                          ratio*self.radius_add_max, ratio*self.radius_add_max, ratio*self.radius_add_min])
                dynamic_r_add = fn_map_r_add(color_grad_mag)
                dynamic_r_query = fn_map_r_query(color_grad_mag)
                self.dynamic_r_add, self.dynamic_r_query = torch.from_numpy(dynamic_r_add).to(
                    self.device), torch.from_numpy(dynamic_r_query).to(self.device)
                torch.save(
                        self.dynamic_r_query, f'{self.output}/dynamic_r_frame/r_query_{idx:05d}.pt')

            outer_joint_iters = 1
            if not init:
                num_joint_iters = cfg['mapping']['iters']
                self.mapping_window_size = cfg['mapping']['mapping_window_size']*(
                    2 if self.n_img > 4000 else 1)
            else:
                self.init_idx = idx
                num_joint_iters = self.iters_first  # more iters on first run

            valid = self.mapping_keyframe(idx,video_idx,mono_depth,outer_joint_iters,num_joint_iters,
                                          gt_color,gt_depth,init,color_refine=False)
            torch.cuda.empty_cache()
              
            init = False
            if not valid:
                self.pipe.send("continue")
                continue

            self.keyframe_list.append(idx)
            dic_of_cur_frame = {'idx': idx, 'color': gt_color.detach().cpu(),
                                'video_idx': video_idx,
                                'mono_depth': mono_depth_input.detach().clone().cpu(),
                                'gt_depth': gt_depth.detach().cpu()}
            
            if self.use_dynamic_radius:
                dic_of_cur_frame.update(
                    {'dynamic_r_query': self.dynamic_r_query.detach()})
            self.keyframe_dict.append(dic_of_cur_frame)
            self.pipe.send("continue")

    def final_refine(self,save_final_pcl=True):
        """
        Final global refinement after mapping all the keyframes
        """
        video_idx = self.video.counter.value-1
        idx = int(self.video.timestamp[video_idx])
        num_joint_iters = self.cfg['mapping']['iters']
        self.mapping_window_size = self.video.counter.value-1
        outer_joint_iters = 5
        self.geo_iter_ratio = 0.0
        num_joint_iters *= 2
        self.fix_color_decoder = True
        self.frustum_feature_selection = False
        self.keyframe_selection_method = 'global'
        _, gt_color, gt_depth, _ = self.frame_reader[idx]
        mono_depth = load_mono_depth(idx,self.cfg)
        gt_color = gt_color.to(self.device).squeeze(0).permute(1,2,0)
        gt_depth = gt_depth.to(self.device)
        mono_depth = mono_depth.to(self.device)
        self.mapping_keyframe(idx,video_idx,mono_depth,outer_joint_iters,num_joint_iters,
                              gt_color,gt_depth,init=False,color_refine=True)

        if save_final_pcl:
            cloud_pos = self.npc.input_pos().cpu().numpy()
            cloud_rgb = self.npc.input_rgb().cpu().numpy()
            point_cloud = np.hstack((cloud_pos, cloud_rgb))
            npc_cloud = self.npc.cloud_pos().cpu().numpy()
            np.save(f'{self.output}/final_point_cloud', point_cloud)
            np.save(f'{self.output}/npc_cloud', npc_cloud)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cloud_pos)
            pcd.colors = o3d.utility.Vector3dVector(cloud_rgb/255.0)
            o3d.io.write_point_cloud(
                f'{self.output}/final_point_cloud.ply', pcd)
            self.printer.print('Saved point cloud and point normals.',FontColor.INFO)
            if self.logger:
                self.logger.log(
                    {f'Cloud/point_cloud_{idx:05d}': wandb.Object3D(point_cloud)})
        if self.low_gpu_mem:
            torch.cuda.empty_cache()


Mapper.eval_kf_imgs = eval_kf_imgs
Mapper.eval_imgs = eval_imgs