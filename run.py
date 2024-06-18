import numpy as np
import torch
import argparse
import os

from src import config
from src.slam import SLAM
from src.utils.datasets import get_dataset
from time import gmtime, strftime
from colorama import Fore,Style

import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument("--only_tracking", action="store_true", help="Only tracking is triggered")
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    cfg = config.load_config(
        args.config, './configs/mono_point_slam.yaml'
    )
    setup_seed(cfg['setup_seed'])

    if args.only_tracking:
        cfg['only_tracking'] = True
        cfg['wandb'] = False
        cfg['mono_prior']['predict_online'] = True

    output_dir = cfg['data']['output']
    output_dir = output_dir+f"/{cfg['setting']}/{cfg['scene']}"

    start_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    start_info = "-"*30+Fore.LIGHTRED_EX+\
                 f"\nStart GlORIE-SLAM at {start_time},\n"+Style.RESET_ALL+ \
                 f"   scene: {cfg['dataset']}-{cfg['scene']},\n" \
                 f"   only_tracking: {cfg['only_tracking']},\n" \
                 f"   output: {output_dir}\n"+ \
                 "-"*30
    print(start_info)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config.save_config(cfg, f'{output_dir}/cfg.yaml')

    dataset = get_dataset(cfg)

    slam = SLAM(cfg,dataset)
    slam.run()

    end_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    print("-"*30+Fore.LIGHTRED_EX+f"\nGlORIE-SLAM finishes!\n+Style.RESET_ALL+{end_time}\n"+"-"*30)

