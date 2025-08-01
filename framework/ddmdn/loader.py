import numpy as np
import torch
import random
import sys
import h5py
import pickle

sys.path.append("/workspace/repos/framework")
sys.path.append("/workspace/repos/framework/datasets")

from torch.utils.data import Dataset
from termcolor import colored
from common import *
from misc import *


class HDF5_Dataset(Dataset):
    """
    A PyTorch-compatible Dataset that reads from an HDF5 file of pickled samples.
    Each sample is stored as a pickled dictionary, so we unpickle it in __getitem__.
    """
    def __init__(self, cfg, path, typ, size):
        """
        :param cfg:    config object (not used heavily here, but kept for consistency)
        :param path:   path to the .h5 file created via convert_pickle_to_hdf5.py
        :param typ:    'train' or 'eval' or similar
        :param size:   fraction of the data (0 < size <= 1.0) or integer count of samples
        """
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.typ = typ
        
        print(colored(f"Opening {typ} dataset @ {path}", 'green'))
        
        # We'll open the file handle lazily (below) so that each DataLoader worker
        # process has its own HDF5 handle. For that, store the file path here.
        self._file_handle = None
        
        # We need to figure out how many total samples are in the h5 file.
        with h5py.File(self.path, 'r') as f:
            n_samples = len(f['data'])
            
        print(colored(f"Total {typ} samples available: {n_samples}", 'green'))
        
        # If size < 1.0, interpret as fraction of total; if size >= 1, interpret as exact number of samples
        if 0 < size < 1.0:
            
            subset_size = int(size * n_samples)
            
        else:
            
            subset_size = min(int(size), n_samples) if size > 1 else n_samples
            
        # Now pick a subset of indices at random if subset_size < n_samples
        if self.typ == "test":
            
            # self.indices = list(range(n_samples))
            full_indices = list(range(n_samples))
            random.shuffle(full_indices)
            self.indices = full_indices[:subset_size]
            print(colored(f"Using {len(self.indices)} samples out of {n_samples} total for {typ}", 'green'))
            
        else:
            
            full_indices = list(range(n_samples))
            random.shuffle(full_indices)
            self.indices = full_indices[:subset_size]
            print(colored(f"Using {len(self.indices)} samples out of {n_samples} total for {typ}", 'green'))
        
        return
    
    
    def _lazy_init(self):
        """ Lazily open the HDF5 file if not already open.
            Ensures that each worker process gets its own file handle.
        """
        
        if self._file_handle is None:
            self._file_handle = h5py.File(self.path, 'r', swmr=True)
            self.dataset = self._file_handle['data']
        
        return
    
    
    def __len__(self):
        
        return len(self.indices)
    
    
    def __getitem__(self, idx):
        
        # Make sure HDF5 file is open in this worker.
        self._lazy_init()
        real_idx = self.indices[idx]
        pickled_bytes = self.dataset[real_idx]
        raw_bytes = pickled_bytes.tobytes()
        sample_dict = pickle.loads(raw_bytes)
        
        return sample_dict
    
    
def collate_fn(batch, cfg, typ, cord_convs_batched):
    
    obs_len = cfg.input_horizon
    
    # With dynamic input horizon scaling per batch
    if typ == "train" and cfg.train_with_dynamic_input_horizon:
        
        # Randomly choose the current input horizon based on user defined limits
        obs_len_delta = abs(np.random.choice(range(obs_len - cfg.train_max_input_horizon, obs_len - cfg.train_min_input_horizon + 1, 1)))
        
    elif typ == "eval" and cfg.eval_with_dynamic_input_horizon:
        
        # Randomly choose the current input horizon based on user defined limits
        obs_len_delta = abs(np.random.choice(range(obs_len - cfg.eval_max_input_horizon, obs_len - cfg.eval_min_input_horizon + 1, 1)))
        
    elif typ == "test" and cfg.test_with_dynamic_input_horizon:
        
        # Randomly choose the current input horizon based on user defined limits
        obs_len_delta = abs(np.random.choice(range(obs_len - cfg.test_max_input_horizon, obs_len - cfg.test_min_input_horizon + 1, 1)))
    
    # Without dynamic input horizon scaling per batch
    else:
        
        # No dynamic scaling always use the maximum input horizon
        obs_len_delta = 0
    
    # Get shapes
    output_horizons = cfg.output_horizons
    plot_output_horizons = cfg.plot_output_horizons
    B = len(batch)
    A = np.array([len(scene['other_agents'])+1 for scene in batch]).max()
    T = cfg.len_forecast_horizon
    V = cfg.len_plot_forecast_horizon
    I = obs_len - obs_len_delta
    
    # Collect scene info
    scene_info = [{
        "motion_state": obj['target_motion'], 
        "rotation": obj['target_rotation'], 
        "translation": obj['target_translation'], 
        "lsa": obj['lsa']['status'] if obj['lsa'] is not None else None, 
        "source": obj['source'],
        "flipped": obj['flipped'],
        "input_horizon": I} 
        for obj in batch]
    
    # Cost grid
    # Add coord-convs
    target_agent_grid = torch.tensor(np.array([obj['target_grid'] for obj in batch]), dtype=torch.float32).unsqueeze(1)
    target_agent_grid = torch.cat([target_agent_grid, cord_convs_batched], dim=1)
    
    # Agent track data and masking
    agent_past_traj = torch.zeros((B, A, I, 6), dtype=torch.float32)
    agent_future_traj = torch.zeros((B, A, T, 4), dtype=torch.float32)
    agent_plot_future_traj = torch.zeros((B, A, V, 2), dtype=torch.float32)
    
    # Initialize the agent mask with True (agent does not exist)
    agent_mask = torch.ones((B, A), dtype=torch.bool)
    
    # Fill the tensor with the "tracks"
    for i, element in enumerate(batch):
        
        # Set to False for existing agents
        agent_mask[i, :len(element['other_agents'])+1] = False
        
        # Target agent
        agent_past_traj[i, 0] = torch.nan_to_num(torch.tensor(element['target_agent']['ego_track'][obs_len_delta:obs_len]), nan=0.0)
        agent_future_traj[i, 0] = torch.tensor(element['target_agent']['ego_track'][[x + obs_len for x in output_horizons], :4])
        agent_plot_future_traj[i, 0] = torch.tensor(element['target_agent']['ego_track'][[x + obs_len for x in plot_output_horizons], :2])
        
        # Other agents
        for j, agent in enumerate(element['other_agents']):
            
            agent_past_traj[i, j+1] = torch.nan_to_num(torch.tensor(agent['ego_track'][obs_len_delta:obs_len]), nan=0.0)
            agent_future_traj[i, j+1] = torch.tensor(agent['ego_track'][[x + obs_len for x in output_horizons], :4])
            agent_plot_future_traj[i, j+1] = torch.tensor(agent['ego_track'][[x + obs_len for x in plot_output_horizons], :2])
    
    return agent_past_traj, agent_future_traj, agent_plot_future_traj, target_agent_grid, agent_mask, scene_info