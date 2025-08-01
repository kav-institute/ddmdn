import os
import sys
import json
import argparse
import logging
import torch
import numpy as np
import seaborn as sns
import shutil
from termcolor import colored
from box import Box

sys.path.append("/workspace/repos/framework")


class Config_Setup:
    """ Class loading parameters and and paths from external config file
    """
    
    def __init__(self, args, mode, dir):
        """ Init and setup 
        """
        
        # Load config for training
        if mode == 'train':
            
            self.config_name = args.cfg.split('/')[-1]
            self.config_path = os.path.join(dir, 'configs', args.cfg)
            print(colored(f"Loading config data: {self.config_path}", 'green'))
            with open(self.config_path) as f:
                cfg = json.load(f)
        
        # For testing
        elif mode == 'test':
            
            self.config_name = args.cfg.split('/')[-1]
            self.config_path = os.path.join(dir, args.cfg.split('.')[0], self.config_name)
            print(colored(f"Loading config data: {self.config_path}", 'green'))
            with open(self.config_path) as f:
                cfg = json.load(f)
        
        # Variables
        self.mode = mode
        self.arch = dir.split('/')[-1]
        self.target = args.cfg.split('.')[0]
        self.with_print = args.print
        self.tqdm_bar = args.bar
        self.epoch = 0
        
        # Parameters
        self.common = Box(cfg['common'])
        self.grid = Box(cfg['grid'])
        self.model = Box(cfg['model'])
        self.train = Box(cfg['train'])
        self.test = Box(cfg['test'])
        self.vis = Box(cfg['vis'])
        
        self.gpuid = self.common.gpuid
        torch.cuda.set_device(self.gpuid)
        self.target_device = torch.device(f"cuda:{self.gpuid}" if torch.cuda.is_available() else 'cpu')
        
        self.dataset = self.common.dataset
        self.input_horizon = self.common.input_horizon 
        self.output_horizons = self.common.output_horizons
        self.plot_output_horizons = self.vis.plot_output_horizons
        self.len_forecast_horizon = len(self.output_horizons)
        self.len_plot_forecast_horizon = len(self.plot_output_horizons)
        self.num_modes = self.common.num_modes
        self.num_hypos = self.common.num_hypos
        self.num_top_hypos = self.common.num_top_hypos
        self.min_k_per_mode  = self.common.min_k_per_mode
        self.with_discrete = self.model.with_discrete_pipeline
        self.with_spatial = self.model.with_spatial_encoder
        self.with_discrete_social_encoder = self.model.with_discrete_social_encoder
        self.with_dynamic_modes = self.model.with_dynamic_modes
        
        
        self.temporal_reweight = torch.tensor(self.common.temporal_reweight).to(self.target_device)
        self.wta_tau = self.train.wta_tau
        self.prune_delta = self.train.prune_delta
        self.prune_tau = self.train.prune_tau
        self.top_p_trunc = self.train.top_p_truncation
        
        # Get mixture dim
        if self.model.mdn_mixture_typ == 'MDN_MultivariateNormal': self.mixture_dim = self.num_modes * 5
        elif self.model.mdn_mixture_typ == 'MDN_LowRankMultivariateNormal': self.mixture_dim = self.num_modes * 6
        elif self.model.mdn_mixture_typ == 'MDN_Laplace': self.mixture_dim = self.num_modes * 4
        elif self.model.mdn_mixture_typ == 'MDN_Normal': self.mixture_dim = self.num_modes * 4
        
        if mode == 'train':
            
            self.train_batch_size = self.train.train_batch_size
            self.train_mini_batch_size = self.train.train_mini_batch_size
            self.eval_batch_size = self.train.eval_batch_size
            self.eval_mini_batch_size = self.train.eval_mini_batch_size
            self.kappas = self.train.confidence_levels
            
            self.train_with_dynamic_input_horizon = self.train.train_with_dynamic_input_horizon
            self.train_min_input_horizon = self.train.train_min_input_horizon
            self.train_max_input_horizon = self.train.train_max_input_horizon
            
            self.eval_with_dynamic_input_horizon = self.train.eval_with_dynamic_input_horizon
            self.eval_min_input_horizon = self.train.eval_min_input_horizon
            self.eval_max_input_horizon = self.train.eval_max_input_horizon
            
            self.plot_reliability = self.vis.plot_reliability
            self.plot_reliability_step = self.vis.plot_reliability_step
            
        elif mode == 'test':
            
            self.test_batch_size = self.test.test_batch_size
            self.test_mini_batch_size = self.test.test_mini_batch_size
            self.kappas = self.test.confidence_levels
            self.test_with_dynamic_input_horizon = self.test.test_with_dynamic_input_horizon
            self.test_min_input_horizon = self.test.test_min_input_horizon
            self.test_max_input_horizon = self.test.test_max_input_horizon
            
            self.plot_reliability = self.test.plot_reliability
            self.plot_reliability_step = 1
        
        # Grid variables
        self.full_grid_resolution = (2 * self.grid.size_meter) / self.grid.full_cell_size
        self.full_grid_cell_size = self.grid.full_cell_size
        self.grid_size_meter = self.grid.size_meter
        self.context_scale = self.common.context_scale
        self.context_radius = self.common.context_radius
        
        # Eval and plotting
        self.reliability_bins = [k for k in np.arange(0.0, 1.01, 0.01)]
        self.confidence_colors_rgb = np.array(sns.color_palette(palette='colorblind', n_colors=self.len_forecast_horizon))[::-1,:]
        self.hypo_colors_rgb = np.array(sns.color_palette(palette='viridis_r', n_colors=self.num_hypos*3))
        self.top_hypo_colors_rgb = np.array(sns.color_palette(palette='viridis', n_colors=self.num_top_hypos*5))
        
        # Paths
        self.data_dir = os.path.join(cfg['paths']['data_path'])
        self.topview_path = cfg['paths']['topview_path']
        data_list = os.listdir(self.data_dir)
        
        # Weather to load full or dev data
        file_association = "_data.h5"
        
        self.train_data_path = os.path.join(self.data_dir, [file for file in data_list if "train"+file_association in file][0])
        self.eval_data_path = os.path.join(self.data_dir, [file for file in data_list if "eval"+file_association in file][0])
        self.test_data_path = os.path.join(self.data_dir, [file for file in data_list if "test"+file_association in file][0])
        
        # Define result directory structure
        self.result_dir = os.path.join(cfg['paths']['result_path'], self.target)
        self.checkpoint_path = os.path.join(self.result_dir, 'checkpoints')
        self.train_path = os.path.join(self.result_dir, 'train')
        self.train_examples_path = os.path.join(self.result_dir, 'train', 'examples')
        self.train_reliability_path = os.path.join(self.result_dir, 'train', 'reliability')
        self.eval_path = os.path.join(self.result_dir, 'eval')
        self.eval_examples_path = os.path.join(self.result_dir, 'eval', 'examples')
        self.eval_reliability_path = os.path.join(self.result_dir, 'eval', 'reliability')
        self.test_path = os.path.join(self.result_dir, 'test')
        self.test_examples_path = os.path.join(self.result_dir, 'test', 'examples')
        self.test_reliability_path = os.path.join(self.result_dir, 'test', 'reliability')
        self.vis_path = os.path.join(self.result_dir, 'vis')
        
        # Create result directory structure
        if not os.path.exists(self.result_dir): os.makedirs(self.result_dir) 
        if not os.path.exists(self.checkpoint_path): os.makedirs(self.checkpoint_path)
        if not os.path.exists(self.train_path): os.makedirs(self.train_path)
        if not os.path.exists(self.train_examples_path): os.makedirs(self.train_examples_path)
        if not os.path.exists(self.train_reliability_path): os.makedirs(self.train_reliability_path)
        if not os.path.exists(self.eval_path): os.makedirs(self.eval_path)
        if not os.path.exists(self.eval_examples_path): os.makedirs(self.eval_examples_path)
        if not os.path.exists(self.eval_reliability_path): os.makedirs(self.eval_reliability_path)
        if not os.path.exists(self.test_path): os.makedirs(self.test_path)
        if not os.path.exists(self.test_examples_path): os.makedirs(self.test_examples_path)
        if not os.path.exists(self.test_reliability_path): os.makedirs(self.test_reliability_path)
        
        # Save config file to dest dir
        if mode == 'train': 
            shutil.copyfile(src=self.config_path, dst=os.path.join(self.result_dir, self.config_name))
            shutil.copyfile(src=os.path.join(dir, "model.py"), dst=os.path.join(self.result_dir, "model.py"))
        
        # Logging
        log_file_path = os.path.join(self.result_dir, f'{self.mode}.log')
        os.remove(log_file_path) if os.path.exists(log_file_path) else None        
        self.logger = logging.getLogger(f'{self.mode}')
        self.logger.setLevel(logging.INFO)
        self.log_file_handler = logging.FileHandler(log_file_path)
        self.log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(self.log_file_handler)
        
        print(colored(f"Loading config setup completed", 'green'))
        return


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='imptc_default.json', help='Config file to use')
    parser.add_argument('-d', '--dir', type=str, default='', help='Path to the test directory')
    parser.add_argument('-g', '--gpu', type=str, default='0', help='GPU ID to use')
    parser.add_argument('-p', '--print', action='store_true')
    parser.add_argument('-b', '--bar', action='store_true')
    return parser.parse_args()