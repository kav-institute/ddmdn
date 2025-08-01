import numpy as np
import random
import os
import h5py
import sys
import pickle

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

MAX_ELEMENTS = 2.5e4


def sample_from_list(d):
    
    if len(d) <= MAX_ELEMENTS:
        
        random.shuffle(d)
        return d
    
    else:
        
        return random.sample(d, int(MAX_ELEMENTS))


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
    with h5py.File(output_path, 'w') as hf:
        
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
    
    #--- Procedure:
    # 1.) Create .pkl files from raw data like Social-LSTM and Social-GAN with: preprocess.py
    # 2.) Extract from raw source data with: extract_from_dataset.py
    # 3.) Generate Database using: generate_database.py
    # 4.) Generate final dataset using: generate_set.py
    
    #--- Target settings:
    # Defined by Social-LSTM and Social-GAN
    # Input: 3.2s -> 8 steps
    # Output: 4.8s -> 12 steps
    # SampleRate: to 2.5 Hz
    # Only Pedestrians
    
    set_names = ['eth', 'hotel', 'univ', 'zara01', 'zara02']
    
    for scene in set_names:
    
        print(f"--- Processing set: {scene} ---")
        generate_set(work_dir='/workspace/data/benchmarks/', typ='eval', dataset=scene)
        generate_set(work_dir='/workspace/data/benchmarks/', typ='train', dataset=scene)
        generate_set(work_dir='/workspace/data/benchmarks/', typ='test', dataset=scene)
        print(f"\n\n")