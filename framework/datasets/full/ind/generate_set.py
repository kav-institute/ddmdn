import numpy as np
import random
import os
import h5py
import sys
import pickle
from typing import List, Dict, Any, Union

sys.path.append('/workspace/repos/framework/datasets')
sys.path.append('/workspace/repos/framework')

from common import *


MOTION_WEIGHTS = {
    'standing': 1.0,
    'starting': 1.0,
    'stopping': 1.0,
    'straight': 1.0,
    'light_left': 1.0,
    'light_right': 1.0,
    'strong_left': 1.0,
    'strong_right': 1.0
}

MAX_ELEMENTS = 7.0e4


def sample_from_list(d):
    
    if len(d) <= MAX_ELEMENTS:
        
        random.shuffle(d)
        return d
    
    else:
        
        return random.sample(d, int(MAX_ELEMENTS))
    

def convert_samples(final_samples, start_timestep=0, scene_gap=100, time_step=10):
    """
    Convert a list of scene‐dicts into a flat list of [timestep, agent_id, x, y].
    Args:
        final_samples: list of dicts, each with keys 'target_agent' and 'other_agents',
                        and where each agent dict has a 'track' ndarray of shape [T,2].
        start_timestep: the timestep for the first frame of scene 0.
        scene_gap: how much to add to the base timestep when you move to the next scene.
        time_step: increment per frame within a scene.
    Returns:
        rows: list of [timestep, agent_id, x, y], sorted by scene then time.
    """
    
    rows: List[List[Union[int, float]]] = []
    
    if not final_samples:
        return rows
    
    # frames per scene
    T = final_samples[0]['target_agent']['track'].shape[0]
    scene_stride = T * time_step + scene_gap
    
    # offset for assigning unique agent IDs
    agent_id_offset = 0
    
    for scene_idx, scene in enumerate(final_samples):
        
        base_time = start_timestep + scene_idx * scene_stride
        
        # build a list of all agents in this scene
        # target first, then other_agents in order
        agents = [scene['target_agent']] + scene['other_agents']
        num_agents = len(agents)
        
        # assign each agent a unique global ID
        # local IDs would be 1..num_agents, so global = offset + local
        for frame_idx in range(T):
            
            t = base_time + frame_idx * time_step
            
            for local_id, agent in enumerate(agents, start=1):
                
                global_id = agent_id_offset + local_id
                x, y = agent['track'][frame_idx]
                rows.append([t, float(global_id), float(x), float(y)])
                
        # increment offset for the next scene
        agent_id_offset += num_agents
        
    return rows


def generate_set(work_dir, typ, dataset):
    
    print(f"{typ}-set starting...")
    database = f"database"
    
    src_dir_path = f"{work_dir}/{dataset}/{database}/{typ}"
    src_file_list = [os.path.join(src_dir_path, fn) for fn in os.listdir(src_dir_path)]
    
    output_path = f"{work_dir}/{dataset}/{database}/{dataset}_{typ}_data.h5"
    all_samples = {v: [] for _, v in MOTION_DICT.items()}
    final_samples = []
    all_motion_cnt = {v: 0 for _, v in MOTION_DICT.items()}
    final_motion_cnt = {v: 0 for _, v in MOTION_DICT.items()}
    
    for src_idx, src_file in enumerate(src_file_list):
        
        # Open file
        file_handle = h5py.File(src_file, 'r', swmr=True)
        raw_data = file_handle['data']
        n_samples = len(raw_data)
        motion_bins = {v: [] for _, v in MOTION_DICT.items()}
        
        # Load all samples
        for idx in range(n_samples):
            
            sample_data = pickle.loads(raw_data[idx].tobytes())
            motion_bins[sample_data['target_motion']].append(sample_data)
            
        # Apply weight reduction
        for k, v in motion_bins.items():
            
            w = MOTION_WEIGHTS[k]
            
            if w < 1.0:
                
                n = int(w * len(v))
                subset = random.sample(v, k=n)
            
            else:
                
                subset = v
            
            # Collection buffer
            all_motion_cnt[k] += len(subset)
            all_samples[k].extend(subset)
        
        # Feedback
        print(f"[{src_idx+1}/{len(src_file_list)}] Processing: {src_file}: Sample Buffer Size: {sum(all_motion_cnt.values())}")
        
    # Shuffle data
    for k, v in all_samples.items():
        
        subset = sample_from_list(d=v)
        final_motion_cnt[k] += len(subset)
        final_samples.extend(subset)
        
    # Feedback
    print(f"Motions: {final_motion_cnt}")
    print(f"Final Buffer Size: {sum(final_motion_cnt.values())}")
    
    # Save sample buffer to file
    print(f"Saving data...")
    
    with h5py.File(f"{output_path}", 'w') as hf:
        
        # We define a 'special_dtype' that can hold variable-length arrays of type uint8
        dt = h5py.special_dtype(vlen=np.dtype('uint8'))
        
        # Create a dataset of length = number of samples, each entry is a variable-length array of bytes
        dset = hf.create_dataset('data', shape=(len(final_samples),), dtype=dt)
        
        for i, sample in enumerate(final_samples):
            
            # Pickle the Python dict
            pickled = pickle.dumps(sample)
            # Convert pickled bytes -> a uint8 array we can store
            dset[i] = np.frombuffer(pickled, dtype='uint8')
            
    print(f"{typ}-set completed...")
    
    
if __name__ == "__main__":
    
    dataset = 'ind'
    
    #--- Procedure:
    # 1.) Extract from raw source data with: extract_from_dataset.py
    # 2.) Generate Database using: generate_database.py
    # 3.) Generate final dataset using: generate_set.py

    #--- Target settings:
    # Input: 1.0s
    # Output: 4.0s
    # Downsampling: to 10.0 Hz
    generate_set(work_dir='/workspace/data/full/', typ='eval', dataset=dataset)
    generate_set(work_dir='/workspace/data/full/', typ='train', dataset=dataset)
    generate_set(work_dir='/workspace/data/full/', typ='test', dataset=dataset)