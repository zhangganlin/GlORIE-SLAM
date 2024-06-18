import torch
import numpy as np
import os
import faiss
import faiss.contrib.torch_utils
from src.utils.common import setup_seed, get_rays_from_uv, update_cam
from src.utils.datasets import load_mono_depth
import droid_backends
from lietorch import SE3

def get_scale(depth_prev, depth_curr):
    # prev*scale = curr
    depth_prev_2 = torch.sum(depth_prev * depth_prev)
    depth_prev_curr = torch.sum(depth_prev * depth_curr)
    scale  = depth_prev_curr / depth_prev_2
    return scale

class NeuralPointCloud(object):
    def __init__(self, cfg, video):
        self.cfg = cfg
        self.c_dim = cfg['model']['c_dim']
        self.device = cfg['device']
        self.cuda_id = 0
        self.use_dynamic_radius = cfg['pointcloud']['use_dynamic_radius']
        self.nn_num = cfg['pointcloud']['nn_num']

        self.nlist = cfg['pointcloud']['nlist']
        self.radius_add = cfg['pointcloud']['radius_add']
        self.radius_min = cfg['pointcloud']['radius_min']
        self.radius_query = cfg['pointcloud']['radius_query']
        self.fix_interval_when_add_along_ray = cfg['pointcloud']['fix_interval_when_add_along_ray']

        self.N_add = cfg['pointcloud']['N_add']
        self.near_end_surface = cfg['pointcloud']['near_end_surface']
        self.far_end_surface = cfg['pointcloud']['far_end_surface']

        self._cloud_pos = None       # (input_pos) * N_add
        self._input_pos = None       # to save locations of the depth input
        self._input_rgb = None       # to save locations of the rgb input at the depth input
        self._input_video_idx = None# to save video index of the depth input
        self._input_j = None # pixel location of input, depth[j,i]
        self._input_i = None # pixel location of input, depth[j,i]
        self._input_depth = None


        self._pts_num = 0          # number of points in neural point cloud
        self.geo_feats = None
        self.col_feats = None

        H,W,fx,fy,cx,cy = update_cam(self.cfg)
        buffer_size = self.cfg["tracking"]["buffer"]
        self._full_pcl = torch.zeros(buffer_size,H,W,3, device="cuda", dtype=torch.float)
        self._full_mask = torch.zeros(buffer_size,H,W, device="cuda", dtype=torch.bool)


        self.resource = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(self.resource,
                                            self.cuda_id,
                                            faiss.IndexIVFFlat(faiss.IndexFlatL2(3), 3, self.nlist, faiss.METRIC_L2))
        self.index.nprobe = cfg['pointcloud']['nprobe']

        self.video = video

        setup_seed(cfg["setup_seed"])
    def get_device(self):
        return self.device


    def get_N_add(self):
        return self.N_add
    def get_near_end_surface(self):
        return self.near_end_surface
    def get_far_end_surface(self):
        return self.far_end_surface
    def is_fix_interval_when_add_along_ray(self):
        return self.fix_interval_when_add_along_ray

    def cloud_pos(self, index=None):
        if index is None:
            return self._cloud_pos
        return self._cloud_pos[index]

    def input_pos(self):
        return self._input_pos

    def full_pcl(self):
        return self._full_pcl
    def full_mask(self):
        return self._full_mask

    def input_rgb(self):
        return self._input_rgb
    
    def input_i(self):
        return self._input_i
    def input_j(self):
        return self._input_j
    def input_video_idx(self):
        return self._input_video_idx

    def pts_num(self):
        return self._pts_num

    def index_train(self, xb):
        assert torch.is_tensor(xb), 'use tensor to train FAISS index'
        self.index.train(xb)
        return self.index.is_trained
    def index_reset(self):
        self.index.reset()
    def index_add(self,xb):
        assert torch.is_tensor(xb)
        self.index.add(xb)


    def index_ntotal(self):
        return self.index.ntotal

    def get_radius_query(self):
        return self.radius_query

    def get_geo_feats(self):
        return self.geo_feats

    def get_col_feats(self):
        return self.col_feats

    def update_geo_feats(self, feats, indices=None):
        assert torch.is_tensor(feats), 'use tensor to update features'
        if indices is not None:
            self.geo_feats[indices] = feats.detach().clone()
        else:
            assert feats.shape[0] == self.geo_feats.shape[0], 'feature shape[0] mismatch'
            self.geo_feats = feats.detach().clone()
        self.geo_feats = self.geo_feats.detach()

    def update_col_feats(self, feats, indices=None):
        assert torch.is_tensor(feats), 'use tensor to update features'
        if indices is not None:
            self.col_feats[indices] = feats.detach().clone()
        else:
            assert feats.shape[0] == self.col_feats.shape[0], 'feature shape[0] mismatch'
            self.col_feats = feats.detach().clone()
        self.col_feats = self.col_feats.detach()
    
    def add_points(self, video_idxs):
        """
        Given some keyframes in depth_video, unproject their depth to 3D points and save as point cloud
        """
        if isinstance(video_idxs, int):
            video_idxs = torch.tensor([video_idxs],dtype=torch.long,device=self.device)
        intrinsic = self.video.intrinsics[0].detach() * float(self.video.down_scale)
        depth_masks = torch.index_select(self.video.valid_depth_mask.detach(), dim=0, index=video_idxs)
        disps = torch.index_select(self.video.disps_up.detach(), dim=0, index=video_idxs)
        poses = torch.index_select(self.video.poses.detach(), dim=0, index=video_idxs)        
        
        pts_gt = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsic) # [b, h, w 3]
        mask = depth_masks

        self._full_pcl[video_idxs] = pts_gt
        self._full_mask[video_idxs] = mask
   
        return torch.sum(mask)    
    

    def add_neural_points(self, batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color,
                          video_idx, i,j, train=False, is_pts_grad=False, dynamic_radius=None):
        """
        Add multiple neural points, will use depth filter when getting these samples.

        Args:
            batch_rays_o (tensor): ray origins (N,3)
            batch_rays_d (tensor): ray directions (N,3)
            batch_gt_depth (tensor): sensor depth (N,)
            batch_gt_color (tensor): sensor color (N,3)
            train (bool): whether to update the FAISS index
            is_pts_grad (bool): the points are chosen based on color gradient
            dynamic_radius (tensor): choose every radius differently based on its color gradient

        """

        if batch_rays_o.shape[0]:
            mask = batch_gt_depth > 0

            depth_mask = (batch_gt_depth < batch_gt_depth.quantile(0.8)*2.0)
            mask = mask * depth_mask

            batch_gt_color = batch_gt_color*255
            batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = \
                batch_rays_o[mask], batch_rays_d[mask], batch_gt_depth[mask], batch_gt_color[mask]
            i, j = i[mask], j[mask]
            if dynamic_radius is not None:
                dynamic_radius = dynamic_radius[mask]

            pts_gt = batch_rays_o[..., None, :] + batch_rays_d[...,
                                                               None, :] * batch_gt_depth[..., None, None]
            mask = torch.ones(pts_gt.shape[0], device=self.device).bool()
            pts_gt = pts_gt.reshape(-1, 3)
            if self.index.is_trained:
                _, _, neighbor_num_gt = self.find_neighbors_faiss(
                    pts_gt, step='add', is_pts_grad=is_pts_grad, dynamic_radius=dynamic_radius)
                mask = (neighbor_num_gt == 0)

            if self._input_pos is None:
                self._input_pos = pts_gt[mask].detach().clone().requires_grad_(False)
                self._input_rgb = batch_gt_color[mask].detach().clone().requires_grad_(False)
                self._input_depth = batch_gt_depth[mask].detach().clone().requires_grad_(False)
                self._input_video_idx = video_idx*torch.ones_like(mask,dtype=torch.long,device=self.device).requires_grad_(False)
                self._input_i = i[mask].detach().clone().requires_grad_(False)
                self._input_j = j[mask].detach().clone().requires_grad_(False)
            else:
                self._input_pos = torch.cat([self._input_pos, pts_gt[mask].detach().clone().requires_grad_(False)])
                self._input_rgb = torch.cat([self._input_rgb, batch_gt_color[mask].detach().clone().requires_grad_(False)])
                self._input_depth = torch.cat([self._input_depth, batch_gt_depth[mask].detach().clone().requires_grad_(False)])
                self._input_video_idx = torch.cat([self._input_video_idx, video_idx*torch.ones_like(i[mask],dtype=torch.long,device=self.device).requires_grad_(False)])
                self._input_i = torch.cat([self._input_i, i[mask].detach().clone().requires_grad_(False)])
                self._input_j = torch.cat([self._input_j, j[mask].detach().clone().requires_grad_(False)])

            gt_depth_surface = batch_gt_depth.unsqueeze(
                -1).repeat(1, self.N_add)
            t_vals_surface = torch.linspace(
                0.0, 1.0, steps=self.N_add, device=self.device)

            if self.fix_interval_when_add_along_ray:
                # add along ray, interval unrelated to depth
                intervals = torch.linspace(-0.04, 0.04, steps=self.N_add,
                                           device=self.device).unsqueeze(0)
                z_vals = gt_depth_surface + intervals
            else:  # add along ray, interval related to depth
                z_vals_surface = self.near_end_surface*gt_depth_surface * (1.-t_vals_surface) + \
                    self.far_end_surface * \
                    gt_depth_surface * (t_vals_surface)
                z_vals = z_vals_surface

            pts = batch_rays_o[..., None, :] + \
                batch_rays_d[..., None, :] * z_vals[..., :, None]
            pts = pts[mask]  # use mask from pts_gt for auxiliary points
            pts = pts.reshape(-1, 3)

            self._pts_num += pts.shape[0]

            if self.geo_feats is None:
                self._cloud_pos = pts.detach().clone().requires_grad_(False)
                self.geo_feats = torch.zeros(
                    [self._pts_num, self.c_dim], device=self.device).normal_(mean=0, std=0.1).requires_grad_(False)
                self.col_feats = torch.zeros(
                    [self._pts_num, self.c_dim], device=self.device).normal_(mean=0, std=0.1).requires_grad_(False)
            else:
                self._cloud_pos = torch.cat([self._cloud_pos, pts.detach().clone().requires_grad_(False)])
                self.geo_feats = torch.cat([self.geo_feats,
                                            torch.zeros([pts.shape[0], self.c_dim], device=self.device).normal_(mean=0, std=0.1)], 0).requires_grad_(False)
                self.col_feats = torch.cat([self.col_feats,
                                            torch.zeros([pts.shape[0], self.c_dim], device=self.device).normal_(mean=0, std=0.1)], 0).requires_grad_(False)

            if train or not self.index.is_trained:
                self.index.train(pts)
            # self.index.reset()
            self.index.train(self._cloud_pos)
            # self.index.add(torch.tensor(self._cloud_pos, device=self.device))
            self.index.add(pts)
            return torch.sum(mask)
        else:
            return 0

    def find_neighbors_faiss(self, pos, step='add', retrain=False, is_pts_grad=False, dynamic_radius=None):
        """
        Query neighbors using faiss.

        Args:
            pos (tensor): points to find neighbors
            step (str): 'add'|'query'
            retrain (bool, optional): if to retrain the index cluster of IVF
            is_pts_grad: whether it's the points chosen based on color grad, will use smaller radius when looking for neighbors
            dynamic_radius (tensor, optional): choose every radius differently based on its color gradient

        Returns:
            D: distances to neighbors for the positions in pos
            I: indices of neighbors for the positions in pos
            neighbor_num: number of neighbors for the positions in pos
        """
        if (not self.index.is_trained) or retrain:
            self.index.train(self._cloud_pos)

        assert step in ['add', 'query']
        split_pos = torch.split(pos, 65000, dim=0)
        D_list = []
        I_list = []

        nn_num = self.nn_num

        for split_p in split_pos:
            D, I = self.index.search(split_p.float(), nn_num)
            D_list.append(D)
            I_list.append(I)
        D = torch.cat(D_list, dim=0)
        I = torch.cat(I_list, dim=0)

        if step == 'query':  # used if dynamic_radius is None
            radius = self.radius_query
        else:  # step == 'add', used if dynamic_radius is None
            if not is_pts_grad:
                radius = self.radius_add
            else:
                radius = self.radius_min

        # faiss returns "D" in the form of squared distances. Thus we compare D to the squared radius
        if dynamic_radius is not None:
            assert pos.shape[0] == dynamic_radius.shape[0], 'shape mis-match for input points and dynamic radius'
            neighbor_num = (D < dynamic_radius.reshape(-1, 1)
                            ** 2).sum(axis=-1).int()
        else:
            neighbor_num = (D < radius**2).sum(axis=-1).int()

        return D, I, neighbor_num

    def sample_near_pcl(self, rays_o, rays_d, near, far, num):
        """
        For pixels with 0 depth readings, preferably sample near point cloud.

        Args:
            rays_o (tensor): rays origin
            rays_d (tensor): rays direction
            near : near end for sampling along this ray
            far: far end
            num (int): sampling num between near and far

        Returns:
            z_vals (tensor): z values for zero valued depth pixels
            invalid_mask (bool): mask for zero valued depth pixels that are not close to neural point cloud
        """
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        n_rays = rays_d.shape[0]
        intervals = 25
        z_vals = torch.linspace(near, far, steps=intervals, device=self.device)
        pts = rays_o[..., None, :] + \
            rays_d[..., None, :] * z_vals[..., :, None]
        pts = pts.reshape(-1, 3)

        if torch.is_tensor(far):
            far = far.item()
        z_vals_section = np.linspace(near, far, intervals)
        z_vals_np = np.linspace(near, far, num)
        z_vals_total = np.tile(z_vals_np, (n_rays, 1))

        pts_split = torch.split(pts, 65000)  # limited by faiss bug
        Ds, Is, neighbor_nums = [], [], []
        for pts_batch in pts_split:
            D, I, neighbor_num = self.find_neighbors_faiss(
                pts_batch, step='query')
            D, I, neighbor_num = D.cpu().numpy(), I.cpu().numpy(), neighbor_num.cpu().numpy()
            Ds.append(D)
            Is.append(I)
            neighbor_nums.append(neighbor_num)
        D = np.concatenate(Ds, axis=0)
        I = np.concatenate(Is, axis=0)
        neighbor_num = np.concatenate(neighbor_nums, axis=0)

        neighbor_num = neighbor_num.reshape((n_rays, -1))
        # a point is True if it has at least one neighbor
        neighbor_num_bool = neighbor_num.astype(bool)
        # a ray is invalid if it has less than two True points along the ray
        invalid = neighbor_num_bool.sum(axis=-1) < 2

        if invalid.sum(axis=-1) < n_rays:
            # select, for the valid rays, a subset of the 25 points along the ray (num points = 5)
            # that are close to the surface.
            r, c = np.where(neighbor_num[~invalid].astype(bool))
            idx = np.concatenate(
                ([0], np.flatnonzero(r[1:] != r[:-1])+1, [r.size]))
            out = [c[idx[i]:idx[i+1]] for i in range(len(idx)-1)]
            z_vals_valid = np.asarray([np.linspace(
                z_vals_section[item[0]], z_vals_section[item[1]], num=num) for item in out])
            z_vals_total[~invalid] = z_vals_valid

        invalid_mask = torch.from_numpy(invalid).to(self.device)
        return torch.from_numpy(z_vals_total).float().to(self.device), invalid_mask

    @torch.no_grad()
    def update_points_pos(self,v_idx,depth,c2w,cfg):        
        '''
        update the positions of points which are unprojected by a certain keyframe

        Args:
            v_idx (int): the index of the keyframe in depth_video
            depth (tensor): depth map of that keyframe
            c2w (tensor) : camera pose of that keyframe
            cfg: config
        '''
        depth = depth.to(self.device)
        input_video_idx = self._input_video_idx
        input_j = self._input_j
        input_i = self._input_i
        input_depth_prev = self._input_depth
        frame_mask = (input_video_idx==v_idx)
        if frame_mask.sum() == 0:
            return
        points_j = input_j[frame_mask]
        points_i = input_i[frame_mask]
        points_depth_prev = input_depth_prev[frame_mask]
        points_depth = depth[points_j, points_i]  
        mask_invalid_depth = (points_depth==0.0)
        if mask_invalid_depth.sum() > 0:    
            scale = get_scale(points_depth_prev[~mask_invalid_depth], points_depth[~mask_invalid_depth])
            points_depth[mask_invalid_depth] = scale * points_depth_prev[mask_invalid_depth]
        
        _,_,fx,fy,cx,cy = update_cam(cfg)

        rays_o, rays_d = get_rays_from_uv(points_i, points_j, c2w,
                                          fx, fy, cx, cy,
                                          self.device)
        pts_input = rays_o[..., None, :] + rays_d[...,None, :]\
            * points_depth[..., None, None]
        pts_input = pts_input.reshape(-1, 3)
        
        self._input_pos[frame_mask] = pts_input.clone()
        self._input_depth[frame_mask] = points_depth.clone()

        points_depth_surface = points_depth.unsqueeze(-1).repeat(1, self.N_add)
        cloud_frame_mask = frame_mask.unsqueeze(-1).repeat(1, self.N_add)

        t_vals_surface = torch.linspace(
            0.0, 1.0, steps=self.N_add, device=self.device)

        if self.fix_interval_when_add_along_ray:
            # add along ray, interval unrelated to depth
            intervals = torch.linspace(-0.04, 0.04, steps=self.N_add,
                                       device=self.device).unsqueeze(0)
            z_vals = points_depth_surface + intervals
        else:  # add along ray, interval related to depth
            z_vals_surface = self.near_end_surface*points_depth_surface * (1.-t_vals_surface) + \
                self.far_end_surface * \
                points_depth_surface * (t_vals_surface)
            z_vals = z_vals_surface

        pts = rays_o[..., None, :] + \
            rays_d[..., None, :] * z_vals[..., :, None]
        pts = pts.reshape(-1, 3)
        cloud_frame_mask = cloud_frame_mask.reshape(-1)
        self._cloud_pos[cloud_frame_mask] = pts.clone()      
    

    def retrain_updated_points(self):
        self.index.reset()
        self.index.train(self._cloud_pos)
        self.index.add(self._cloud_pos)

def proj_depth_map(c2w, npc, device, cfg, neural_pcl=False):
    '''
    project the pointcloud to a camera pose to get a projected depth map,
    if there are multiple points projected into the same pixel, 
        choose the closest one to calculate depth.

    Args:
        c2w (tensor): the given camera pose
        npc (NeuralPointCloud): the pointcloud object
        cfg: config
        neural_pcl (bool): if True, use the neural point cloud, 
                           if False, use the point cloud 
                              which is unprojected by previous keyframes' depth maps 
    Returns:
        depth_map (tensor): the projected depth map
    '''
    H,W,fx,fy,cx,cy = update_cam(cfg)
    if neural_pcl:
        points = npc.cloud_pos()
    else:
        points = npc.full_pcl()[npc.full_mask()]

    w2c = c2w.inverse()
    ones = torch.ones_like(points[:,0]).reshape(-1,1)
    homo_vertices = torch.cat([points,ones],axis=1).reshape(-1,4,1)
    cam_cord_homo = w2c@homo_vertices
    cam_cord = cam_cord_homo[:, :3]
    K = torch.tensor([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]],device=device).reshape(3, 3)

    # flip the x-axis such that the pixel space is u from the left to right, v top to bottom.
    # without the flipping of the x-axis, the image is assumed to be flipped horizontally.
    cam_cord[:, 0] *= -1
    uv = K@cam_cord
    uv = uv.squeeze(-1)
    z = uv[:, -1]+1e-6
    uv = uv[:, :2]/z[:,None]
    mask = (uv[:, 0] < W)*(uv[:, 0] >= 0) * \
        (uv[:, 1] < H)*(uv[:, 1] >= 0)*((-z)>0)
    uv = (uv[mask,:]).long()
    z = -z[mask].float()

    depth_map = torch.full((H, W), 0.0, device=device, dtype=torch.float)

    z_sorted, idx_sorted = torch.sort(z)
    uv_sorted = uv[idx_sorted,:]
    unique_uv, idx_uv, counts_uv = torch.unique(uv_sorted, dim=0, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx_uv, stable=True)
    cum_sum = counts_uv.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0],device=device), cum_sum[:-1]))
    uv_first_indicies = ind_sorted[cum_sum]

    depth_map[unique_uv[:,1],unique_uv[:,0]] = z_sorted[uv_first_indicies]
    return depth_map


def update_points_pos(npc:NeuralPointCloud, video):
    '''
    deform the pointcloud according to the estiamted poses
        and depth maps from the depth_video

    Args:
        npc (NeuralPointCloud): the pointcloud object
        video (DepthVideo): the depth_video object
    '''
    with video.get_lock():
        video_idx, = torch.where(video.npc_dirty.clone())
    if (len(video_idx) == 0) or (npc.pts_num() == 0):
        return
    video.npc_dirty[video_idx] = False
    device = npc.get_device()
    for i in range(len(video_idx)):
        v_idx = video_idx[i]
        est_depth, est_depth_mask, c2w = video.get_depth_and_pose(v_idx,device)
        est_depth[~est_depth_mask] = 0
        c2w[:3, 1:3] *= -1
        if video.cfg["mapping"]["render_depth"] == "proxy":
            render_depth = est_depth
        elif video.cfg["mapping"]["render_depth"] == "mono":
            dataset_idx = int(video.timestamp[v_idx])
            mono_depth = load_mono_depth(dataset_idx,video.cfg).to(device)
            depth_mask = est_depth > 0
            scale_shift = video.get_depth_scale_and_shift(v_idx,mono_depth,
                                                            est_depth, depth_mask)
            render_depth = mono_depth*scale_shift[0] + scale_shift[1]
        npc.update_points_pos(v_idx,render_depth.clone(),c2w.clone(),video.cfg)
    npc.add_points(video_idx)
    
    npc.retrain_updated_points()

def get_proxy_render_depth(npc,cfg,c2w, droid_depth, mono_depth, device, 
                           idx=None, use_mono_to_complete=True):
    '''
    get proxy depth map in the paper, 
    by combining droid_depth (the one from depth_video), 
                 projected_depth and mono_depth

    Args:
        npc (NeuralPointCloud): the pointcloud object
        cfg: config
        c2w (tensor): camera pose
        droid_depth (tensor): depth map estimated from the tracker, the one stored in depth_video
        mono_depth (tensor): mono depth prior
        idx (int): index of the keyframe in the dataset
        use_mono_to_complete (bool): whether to use mono_depth to complete proxy depth map
    Returns:
        proxy_depth (tensor): proxy depth map
    '''
    proxy_depth = droid_depth.clone()
    droid_valid_mask = droid_depth > 0.0
    proj_depth = proj_depth_map(c2w, npc, device, cfg)
    proj_valid_mask = (proj_depth > 0.0)
    proj_mask = (~droid_valid_mask) * proj_valid_mask
    proxy_depth[proj_mask] = proj_depth[proj_mask]

    if cfg["mapping"]["save_depth"] and (idx is not None):
        output_path_droid = f"{cfg['data']['output']}/{cfg['setting']}/{cfg['scene']}/semi_dense_depth/droid/{idx:05d}.npy"
        output_path_proj = f"{cfg['data']['output']}/{cfg['setting']}/{cfg['scene']}/semi_dense_depth/project/{idx:05d}.npy"
        if not os.path.isfile(output_path_droid):
            droid_depth_np = droid_depth.detach().cpu().float().numpy()
            proj_depth_np = proxy_depth.detach().cpu().float().numpy()
            np.save(output_path_droid,droid_depth_np)
            np.save(output_path_proj,proj_depth_np)
    if use_mono_to_complete:
        render_invalid_mask = (proxy_depth == 0)
        proxy_depth[render_invalid_mask] = mono_depth[render_invalid_mask]

    return proxy_depth
