import os
import shutil
import torch
import cv2
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from src.utils.datasets import load_mono_depth
from src.neural_point import proj_depth_map, get_proxy_render_depth
from skimage.color import rgb2gray
from skimage import filters
from scipy.interpolate import interp1d
from pytorch_msssim import ms_ssim
from src.utils.common import align_scale_and_shift
from src.utils.Printer import FontColor
import traceback
import numpy as np
from tqdm import tqdm

def eval_kf_imgs(self):
    # re-render frames at the end for meshing
    self.printer.print('Starting re-rendering keyframes...',FontColor.EVAL)
    render_idx, frame_cnt, masked_psnr_sum, masked_ssim_sum, masked_lpips_sum = self.init_idx, 0, 0, 0, 0
    psnr_sum, ssim_sum, lpips_sum = 0,0,0
    if os.path.exists(f'{self.output}/rendered_every_keyframe'):
        shutil.rmtree(f'{self.output}/rendered_every_keyframe')
    os.makedirs(f'{self.output}/rendered_every_keyframe', exist_ok=True)
    os.makedirs(f'{self.output}/rerendered_keyframe_image', exist_ok=True)
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type='alex', normalize=True).to(self.device)
    try:
        for kf in tqdm(self.keyframe_dict):
            render_idx = kf['idx']
            render_video_idx = kf['video_idx']
            _, gt_color, gt_depth, _= self.frame_reader[render_idx]
            mono_depth = load_mono_depth(render_idx,self.cfg)
            gt_color = gt_color.to(self.device).squeeze(0).permute(1,2,0)
            gt_depth = gt_depth.to(self.device)
            mono_depth = mono_depth.to(self.device)
            cur_c2w, mono_depth_wq,droid_depth = self.get_c2w_and_depth(render_video_idx, render_idx, mono_depth)
            if self.cfg["mapping"]["render_depth"] == "proxy":
                render_depth = get_proxy_render_depth(self.npc, self.cfg, cur_c2w.clone(), 
                                                        droid_depth, mono_depth_wq, self.device,
                                                        use_mono_to_complete=self.use_mono_to_complete)
            elif self.cfg["mapping"]["render_depth"] == "mono":
                render_depth = mono_depth_wq

            r_query_frame = torch.load(f'{self.output}/dynamic_r_frame/r_query_{render_idx:05d}.pt', map_location=self.device) \
                if self.use_dynamic_radius else None
            if r_query_frame is not None:
                r_query_frame = r_query_frame/3.0 * render_depth

            cur_frame_depth, cur_frame_color, valid_mask, valid_ray_count = self.visualizer.vis_value_only(render_depth, cur_c2w, self.npc, self.decoders,
                                                                                self.npc.get_geo_feats(), self.npc.get_col_feats(), 
                                                                                dynamic_r_query=r_query_frame, cloud_pos=self.cloud_pos_tensor)
            img = cv2.cvtColor(
                cur_frame_color.cpu().numpy()*255, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(
                f'{self.output}/rerendered_keyframe_image', f'frame_{render_idx:05d}.png'), img)

            mse_loss = torch.nn.functional.mse_loss(
                gt_color, cur_frame_color)
            psnr_frame = -10. * torch.log10(mse_loss)
            ssim_value = ms_ssim(gt_color.transpose(0, 2).unsqueeze(0).float(), cur_frame_color.transpose(0, 2).unsqueeze(0).float(),
                                    data_range=1.0, size_average=True)
            lpips_value = cal_lpips(torch.clamp(gt_color.unsqueeze(0).permute(0, 3, 1, 2).float(), 0.0, 1.0),
                                    torch.clamp(cur_frame_color.unsqueeze(0).permute(0, 3, 1, 2).float(), 0.0, 1.0)).item()
            psnr_sum += psnr_frame
            ssim_sum += ssim_value
            lpips_sum += lpips_value


            mask = (valid_mask>0) * (render_depth > 0) * (gt_depth>0) # * (droid_depth>0)
            cur_frame_depth[~mask] = 0.0 
            gt_color[~mask] = 0.0
            cur_frame_color[~mask] = 0.0
            np.save(f'{self.output}/rendered_every_keyframe/depth_{render_idx:05d}',
                    cur_frame_depth.cpu().numpy())
            np.save(f'{self.output}/rendered_every_keyframe/color_{render_idx:05d}',
                    cur_frame_color.cpu().numpy())

            masked_mse_loss = torch.nn.functional.mse_loss(
                gt_color[mask], cur_frame_color[mask])
            masked_psnr_frame = -10. * torch.log10(masked_mse_loss)
            masked_ssim_value = ms_ssim(gt_color.transpose(0, 2).unsqueeze(0).float(), cur_frame_color.transpose(0, 2).unsqueeze(0).float(),
                                    data_range=1.0, size_average=True)
            masked_lpips_value = cal_lpips(torch.clamp(gt_color.unsqueeze(0).permute(0, 3, 1, 2).float(), 0.0, 1.0),
                                    torch.clamp(cur_frame_color.unsqueeze(0).permute(0, 3, 1, 2).float(), 0.0, 1.0)).item()
            masked_psnr_sum += masked_psnr_frame
            masked_ssim_sum += masked_ssim_value
            masked_lpips_sum += masked_lpips_value

            if self.logger:
                self.logger.log({'idx_frame': render_idx,
                            'psnr_frame': masked_psnr_frame})
            frame_cnt += 1

        avg_masked_psnr = masked_psnr_sum / frame_cnt
        avg_masked_ssim = masked_ssim_sum / frame_cnt
        avg_masked_lpips = masked_lpips_sum / frame_cnt
        avg_psnr = psnr_sum / frame_cnt
        avg_ssim = ssim_sum / frame_cnt
        avg_lpips = lpips_sum / frame_cnt
        self.printer.print(f'avg_masked_msssim: {avg_masked_ssim}',FontColor.EVAL)
        self.printer.print(f'avg_msssim: {avg_ssim}',FontColor.EVAL)
        self.printer.print(f'avg_masked_psnr: {avg_masked_psnr}',FontColor.EVAL)
        self.printer.print(f'avg_psnr: {avg_psnr}',FontColor.EVAL)
        self.printer.print(f'avg_masked_lpips: {avg_masked_lpips}',FontColor.EVAL)
        self.printer.print(f'avg_lpips: {avg_lpips}',FontColor.EVAL)
        if self.logger:
            self.logger.log({'avg_masked_ssim': avg_masked_ssim, 'avg_ssim': avg_ssim,
                                'avg_masked_psnr': avg_masked_psnr, 'avg_psnr': avg_psnr,
                                'avg_masked_lpips': avg_masked_lpips, 'avg_lpips': avg_lpips})
        out_path=f'{self.output}/logs/metrics_render_kf.txt'
        output_str = f"avg_masked_ssim: {avg_masked_ssim}\n"
        output_str += f"avg_masked_psnr: {avg_masked_psnr}\n"
        output_str += f"avg_masked_lpips: {avg_masked_lpips}\n###############\n"
        output_str += f"avg_ssim: {avg_ssim}\n"
        output_str += f"avg_psnr: {avg_psnr}\n"
        output_str += f"avg_lpips: {avg_lpips}\n###############\n"
        with open(out_path, 'w+') as fp:
            fp.write(output_str)
    except Exception as e:
        traceback.print_exception(e)
        self.printer.print('Rerendering frames failed.',FontColor.ERROR)
    self.printer.print(f'Finished rendering {frame_cnt} frames.',FontColor.EVAL)

def eval_imgs(self,est_c2ws):

    # re-render frames at the end for meshing
    self.printer.print('Starting re-rendering frames...',FontColor.EVAL)
    render_idx, frame_cnt, masked_psnr_sum, masked_ssim_sum, masked_lpips_sum = self.init_idx, 0, 0, 0, 0
    psnr_sum, ssim_sum, lpips_sum = 0,0,0
    if os.path.exists(f'{self.output}/rendered_every_frame'):
        shutil.rmtree(f'{self.output}/rendered_every_frame')
    os.makedirs(f'{self.output}/rendered_every_frame', exist_ok=True)
    os.makedirs(f'{self.output}/rerendered_image', exist_ok=True)
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type='alex', normalize=True).to(self.device)
    try:
        for idx, gt_color, gt_depth, _ in tqdm(self.frame_reader):
            every_frame = self.cfg['mapping']['every_frame']
            if idx % every_frame != 0:
                continue
            render_idx = idx
            mono_depth = load_mono_depth(render_idx,self.cfg)
            gt_color = gt_color.to(self.device).squeeze(0).permute(1,2,0)
            gt_depth = gt_depth.to(self.device)
            mono_depth = mono_depth.to(self.device)
            cur_c2w = torch.from_numpy(est_c2ws[idx].copy()).to(self.device)
            cur_c2w[:3, 1:3] *= -1
            proj_depth = proj_depth_map(cur_c2w, self.npc, self.device, self.cfg)
            proj_valid_mask = (proj_depth > 0)
            mono_scale, mono_shift,_ = align_scale_and_shift(mono_depth,proj_depth,proj_valid_mask)
            mono_depth_wq = mono_scale*mono_depth + mono_shift
            render_depth = proj_depth
            render_depth[~proj_valid_mask] = mono_depth_wq[~proj_valid_mask]


            if self.use_dynamic_radius:
                ratio = self.radius_query_ratio
                intensity = rgb2gray(gt_color.cpu().numpy())
                grad_y = filters.sobel_h(intensity)
                grad_x = filters.sobel_v(intensity)
                color_grad_mag = np.sqrt(grad_x**2 + grad_y**2)  # range 0~1
                color_grad_mag = np.clip(
                    color_grad_mag, 0.0, self.color_grad_threshold)  # range 0~1
                fn_map_r_query = interp1d([0, 0.01, self.color_grad_threshold], [
                                        ratio*self.radius_add_max, ratio*self.radius_add_max, ratio*self.radius_add_min])
                dynamic_r_query = fn_map_r_query(color_grad_mag)

            r_query_frame = torch.from_numpy(dynamic_r_query).to(self.device) \
                if self.use_dynamic_radius else None
            if r_query_frame is not None:
                r_query_frame = r_query_frame/3.0 * render_depth

            cur_frame_depth, cur_frame_color, valid_mask, valid_ray_count = self.visualizer.vis_value_only(render_depth, cur_c2w, self.npc, self.decoders,
                                                                                self.npc.get_geo_feats(), self.npc.get_col_feats(), 
                                                                                dynamic_r_query=r_query_frame, cloud_pos=self.cloud_pos_tensor)
            img = cv2.cvtColor(
                cur_frame_color.cpu().numpy()*255, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(
                f'{self.output}/rerendered_image', f'frame_{render_idx:05d}.png'), img)

            mse_loss = torch.nn.functional.mse_loss(
                gt_color, cur_frame_color)
            psnr_frame = -10. * torch.log10(mse_loss)
            ssim_value = ms_ssim(gt_color.transpose(0, 2).unsqueeze(0).float(), cur_frame_color.transpose(0, 2).unsqueeze(0).float(),
                                    data_range=1.0, size_average=True)
            lpips_value = cal_lpips(torch.clamp(gt_color.unsqueeze(0).permute(0, 3, 1, 2).float(), 0.0, 1.0),
                                    torch.clamp(cur_frame_color.unsqueeze(0).permute(0, 3, 1, 2).float(), 0.0, 1.0)).item()
            psnr_sum += psnr_frame
            ssim_sum += ssim_value
            lpips_sum += lpips_value

            mask = (valid_mask>0) * (render_depth > 0) * (gt_depth>0) # * (droid_depth>0)
            cur_frame_depth[~mask] = 0.0 
            gt_color[~mask] = 0.0
            cur_frame_color[~mask] = 0.0
            np.save(f'{self.output}/rendered_every_frame/depth_{render_idx:05d}',
                    cur_frame_depth.cpu().numpy())
            np.save(f'{self.output}/rendered_every_frame/color_{render_idx:05d}',
                    cur_frame_color.cpu().numpy())

            masked_mse_loss = torch.nn.functional.mse_loss(
                gt_color[mask], cur_frame_color[mask])
            masked_psnr_frame = -10. * torch.log10(masked_mse_loss)
            masked_ssim_value = ms_ssim(gt_color.transpose(0, 2).unsqueeze(0).float(), cur_frame_color.transpose(0, 2).unsqueeze(0).float(),
                                    data_range=1.0, size_average=True)
            masked_lpips_value = cal_lpips(torch.clamp(gt_color.unsqueeze(0).permute(0, 3, 1, 2).float(), 0.0, 1.0),
                                    torch.clamp(cur_frame_color.unsqueeze(0).permute(0, 3, 1, 2).float(), 0.0, 1.0)).item()
            masked_psnr_sum += masked_psnr_frame
            masked_ssim_sum += masked_ssim_value
            masked_lpips_sum += masked_lpips_value

            frame_cnt += 1

        avg_masked_psnr = masked_psnr_sum / frame_cnt
        avg_masked_ssim = masked_ssim_sum / frame_cnt
        avg_masked_lpips = masked_lpips_sum / frame_cnt
        avg_psnr = psnr_sum / frame_cnt
        avg_ssim = ssim_sum / frame_cnt
        avg_lpips = lpips_sum / frame_cnt

        self.printer.print(f'avg_masked_msssim: {avg_masked_ssim}',FontColor.EVAL)
        self.printer.print(f'avg_msssim: {avg_ssim}',FontColor.EVAL)
        self.printer.print(f'avg_masked_psnr: {avg_masked_psnr}',FontColor.EVAL)
        self.printer.print(f'avg_psnr: {avg_psnr}',FontColor.EVAL)
        self.printer.print(f'avg_masked_lpips: {avg_masked_lpips}',FontColor.EVAL)
        self.printer.print(f'avg_lpips: {avg_lpips}',FontColor.EVAL)


        if self.logger:
            self.logger.log({'avg_masked_ssim_every': avg_masked_ssim, 'avg_ssim_every': avg_ssim,
                                'avg_masked_psnr_every': avg_masked_psnr, 'avg_psnr_every': avg_psnr,
                                'avg_masked_lpips_every': avg_masked_lpips, 'avg_lpips_every': avg_lpips})
        out_path=f'{self.output}/logs/metrics_render_every.txt'
        output_str = f"avg_masked_msssim: {avg_masked_ssim}\n"
        output_str += f"avg_masked_psnr: {avg_masked_psnr}\n"
        output_str += f"avg_masked_lpips: {avg_masked_lpips}\n###############\n"
        output_str += f"avg_msssim: {avg_ssim}\n"
        output_str += f"avg_psnr: {avg_psnr}\n"
        output_str += f"avg_lpips: {avg_lpips}\n###############\n"
        with open(out_path, 'w+') as fp:
            fp.write(output_str)
    except Exception as e:
        traceback.print_exception(e)
        self.printer.print('Rerendering frames failed.',FontColor.ERROR)
    self.printer.print(f'Finished rendering {frame_cnt} frames.',FontColor.EVAL)
