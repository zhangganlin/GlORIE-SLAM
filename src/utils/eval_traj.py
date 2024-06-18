import numpy as np
from lietorch import SE3

def align_kf_traj(npz_path,stream,return_full_est_traj=False):
    offline_video = dict(np.load(npz_path))
    traj_ref = []
    traj_est = []
    video_traj = offline_video['poses']
    video_timestamps = offline_video['timestamps']
    timestamps = []

    for i in range(video_timestamps.shape[0]):
        timestamp = int(video_timestamps[i])
        val = stream.poses[timestamp].sum()
        if np.isnan(val) or np.isinf(val):
            print(f'Nan or Inf found in gt poses, skipping {i}th pose!')
            continue
        traj_est.append(video_traj[i])
        traj_ref.append(stream.poses[timestamp])
        timestamps.append(video_timestamps[i])

    from evo.core.trajectory import PoseTrajectory3D

    traj_est =PoseTrajectory3D(poses_se3=traj_est,timestamps=timestamps)
    traj_ref =PoseTrajectory3D(poses_se3=traj_ref,timestamps=timestamps)

    from evo.core import sync

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    r_a, t_a, s = traj_est.align(traj_ref, correct_scale=True)

    if return_full_est_traj:
        from evo.core import lie_algebra as lie
        traj_est_full = PoseTrajectory3D(poses_se3=video_traj,timestamps=video_timestamps)
        traj_est_full.scale(s)
        traj_est_full.transform(lie.se3(r_a, t_a))
        traj_est = traj_est_full

    return r_a, t_a, s, traj_est, traj_ref    

def align_full_traj(traj_est_full,stream):

    timestamps = []
    traj_ref = []
    traj_est = []
    for i in range(len(stream.poses)):
        val = stream.poses[i].sum()
        if np.isnan(val) or np.isinf(val):
            print(f'Nan or Inf found in gt poses, skipping {i}th pose!')
            continue
        traj_est.append(traj_est_full[i])
        traj_ref.append(stream.poses[i])
        timestamps.append(float(i))
    
    from evo.core.trajectory import PoseTrajectory3D

    traj_est =PoseTrajectory3D(poses_se3=traj_est,timestamps=timestamps)
    traj_ref =PoseTrajectory3D(poses_se3=traj_ref,timestamps=timestamps)

    from evo.core import sync

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    r_a, t_a, s = traj_est.align(traj_ref, correct_scale=True)
    return r_a, t_a, s, traj_est, traj_ref    


def traj_eval_and_plot(traj_est, traj_ref, plot_parent_dir, plot_name):
    import os
    from evo.core import metrics
    from evo.tools import plot
    import matplotlib.pyplot as plt
    if not os.path.exists(plot_parent_dir):
        os.makedirs(plot_parent_dir)
    print("calculating APE ...")
    data = (traj_ref, traj_est)
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data(data)
    ape_statistics = ape_metric.get_all_statistics()

    print("plotting ...")

    plot_collection = plot.PlotCollection("kf factor graph")
    # metric values
    fig_1 = plt.figure(figsize=(8, 8))
    plot_mode = plot.PlotMode.xy
    ax = plot.prepare_axis(fig_1, plot_mode)
    plot.traj(ax, plot_mode, traj_ref, '--', 'gray', 'reference')
    plot.traj_colormap(
    ax, traj_est, ape_metric.error, plot_mode, min_map=ape_statistics["min"],
    max_map=ape_statistics["max"], title="APE mapped onto trajectory")
    plot_collection.add_figure("2d", fig_1)
    plot_collection.export(f"{plot_parent_dir}/{plot_name}.png", False)

    return ape_statistics


def kf_traj_eval(npz_path, plot_parent_dir,plot_name, stream, logger):
    r_a, t_a, s, traj_est, traj_ref = align_kf_traj(npz_path, stream)

    offline_video = dict(np.load(npz_path))
    
    import os
    if not os.path.exists(plot_parent_dir):
        os.makedirs(plot_parent_dir)

    ape_statistics = traj_eval_and_plot(traj_est,traj_ref,plot_parent_dir,plot_name)

    output_str = "#"*10+"Keyframes traj"+"#"*10+"\n"
    output_str += f"scale: {s}\n"
    output_str += f"rotation:\n{r_a}\n"
    output_str += f"translation:{t_a}\n"
    output_str += f"statistics:\n{ape_statistics}\n"
    output_str += "#"*34+"\n"

    print(output_str)
    out_path=f'{plot_parent_dir}/metrics_kf_traj.txt'
    with open(out_path, 'w+') as fp:
        fp.write(output_str)
    if logger is not None:
        logger.log({'kf_ate_rmse':ape_statistics['rmse'],'pose_scale':s})

    offline_video["scale"]=np.array(s)
    np.savez(npz_path,**offline_video)

    from src.utils.Visualizer import CameraPoseVisualizer
    est_camera_vis = CameraPoseVisualizer()
    est_camera_vis.add_traj(traj_est.poses_se3)
    est_camera_vis.save(f"{plot_parent_dir}/{plot_name}_3d.png")

    ref_camera_vis = CameraPoseVisualizer()
    ref_camera_vis.add_traj(traj_ref.poses_se3)
    ref_camera_vis.save(f"{plot_parent_dir}/ref_3d.png")

    return ape_statistics, s, r_a, t_a


def full_traj_eval(traj_filler, plot_parent_dir, plot_name, stream,logger):

    traj_est_inv = traj_filler(stream)
    traj_est_lietorch = traj_est_inv.inv()
    traj_est = traj_est_lietorch.matrix().data.cpu().numpy()
    kf_num = traj_filler.video.counter.value
    kf_timestamps = traj_filler.video.timestamp[:kf_num].cpu().int().numpy()
    kf_poses = SE3(traj_filler.video.poses[:kf_num].clone()).inv().matrix().data.cpu().numpy()
    traj_est[kf_timestamps] = kf_poses
    traj_est_not_align = traj_est.copy()

    r_a, t_a, s, traj_est, traj_ref = align_full_traj(traj_est, stream)    

    import os
    if not os.path.exists(plot_parent_dir):
        os.makedirs(plot_parent_dir)

    ape_statistics = traj_eval_and_plot(traj_est,traj_ref,plot_parent_dir,plot_name)
    
    output_str = "#"*10+"Full traj"+"#"*10+"\n"
    output_str += f"scale: {s}\n"
    output_str += f"rotation:\n{r_a}\n"
    output_str += f"translation:{t_a}\n"
    output_str += f"statistics:\n{ape_statistics}\n"
    output_str += "#"*29+"\n"

    print(output_str)
    out_path=f'{plot_parent_dir}/metrics_full_traj.txt'
    with open(out_path, 'w+') as fp:
        fp.write(output_str)
    if logger is not None:
        logger.log({'full_ate_rmse':ape_statistics['rmse']})
    return traj_est_not_align, traj_est, traj_ref
