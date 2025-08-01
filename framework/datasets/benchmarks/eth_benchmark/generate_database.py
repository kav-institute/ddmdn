import numpy as np
import os
import cv2
import sys
import copy
import math
import pickle
import h5py

sys.path.append('/workspace/repos/framework/datasets')
sys.path.append('/workspace/repos/framework')

from torch.utils.data import Dataset
from joblib import Parallel, delayed
from eth.utils import *
from common import *
from transformation import *


GRID_SIZE_PX = 128
CONTEXT_RADIUS = 3
GRID_RADIUS = 10

SPECIAL_STATES = {'light_left', 'light_right', 'strong_left', 'strong_right', 'starting', 'stopping'}


def filter_unique(sequences):
    
    seen = set()
    unique_sequences = []
    
    for seq in sequences:
        
        signature = tuple(sorted(seq))
        
        if signature not in seen:
            seen.add(signature)
            unique_sequences.append(seq)
            
    return unique_sequences


def filter_objects(target, other, obs_len, r):
    
    tp = target['track'][obs_len-1]
    
    # Check if object is within context radius to target
    filtered = []
    
    for idx, _ in enumerate(other):
        
        cp = other[idx]['track'][obs_len-1]
        
        # Yes within radius keep it
        if  math.sqrt((cp[0] - tp[0]) ** 2 + (cp[1] - tp[1]) ** 2) < r: 
            filtered.append(other[idx])
            
    return filtered


def transform_segmentation(seg_map):
    
    new_seg_map = np.zeros_like(seg_map, dtype=np.uint8)
    new_seg_map[seg_map == ETH_SEG_DICT['road']] = TARGET_SEG_DICT['road']
    new_seg_map[seg_map == ETH_SEG_DICT['terrain_1']] = TARGET_SEG_DICT['road']
    new_seg_map[seg_map == ETH_SEG_DICT['terrain_2']] = TARGET_SEG_DICT['terrain']
    new_seg_map[seg_map == ETH_SEG_DICT['pavement']] = TARGET_SEG_DICT['sidewalk']
    new_seg_map[seg_map == ETH_SEG_DICT['blocking']] = TARGET_SEG_DICT['blocking']
    
    return new_seg_map[:,:,0]


def filter_substrings(list_a, list_b, case_sensitive=True):
    
    if not case_sensitive:
        
        # Lowercase all strings for comparison
        list_a = [a.lower() for a in list_a]
        list_b = [b.lower() for b in list_b]
        
    return [a for a in list_a if any(a in b for b in list_b)]


class Sample_Generator(Dataset):
    
    def __init__(self, path, dataset, typ, obs_len=None, future_len=None, sample_rate=None, norm=False, augment=False, num_workers=8):
        
        super(Sample_Generator, self).__init__()
        
        self.delta_t = sample_rate
        self.dataset = dataset
        self.norm = norm
        self.typ = typ
        self.path = path
        self.obs_len = obs_len
        self.future_len = future_len
        self.seq_len = obs_len + future_len
        self.work_dir = os.path.join(path, 'database', typ)
        if not os.path.exists(self.work_dir): os.makedirs(self.work_dir)
        self.invalid_cnt = 0
        self.augment = augment
        self.num_workers = num_workers
        
        # Load the data
        file_path = os.path.join(path, f"{self.dataset}_{self.typ}_extracted.pickle")
        raw_data = np.load(file=file_path, allow_pickle=True)
        
        # Get sequence related masks
        self.seg_maps = {}
        self.homographs = {}
        masks_path = os.path.join(self.path, 'masks')
        
        mask_files = os.listdir(masks_path)
        source_names = filter_substrings(list_a=ETH_SCENES, list_b=mask_files)
        
        for name in list(source_names):
            
            sm = cv2.cvtColor(cv2.imread(os.path.join(self.path, 'masks', f"{name}_mask.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            sm = transform_segmentation(seg_map=sm)
            self.seg_maps[name] = sm
            self.homographs[name] = np.loadtxt(os.path.join(self.path, 'homography', f"{name}_H.txt"))
        
        # Parallel processing over sources
        job_list = sorted(list(raw_data.keys()))
        print(f"--- Start {typ} data processing of {len(job_list)} sources ---")
        results = Parallel(n_jobs=self.num_workers)(delayed(self.process_data)(
            src=src, 
            data_package=raw_data[src],
            idx=f"[{k+1}/{len(job_list)}]")
            for k, src in enumerate(job_list)
        )
        
        print(f"--- End {typ} data processing of {len(job_list)} sources: Extracted {np.array(results).sum()} samples ---")
        return
    
    
    # Each item in raw_data is a list representing a time specific scene within a sequence
    # Each scene contains a variable number of agents (objects)
    # Function to process a full single sequence
    def process_data(self, src, data_package, idx):
        
        motion_bins = {v: [] for _, v in MOTION_DICT.items()}
        
        # Iterate timestamps of sequence
        for idx, scene_data in enumerate(data_package):
            
            # Get current scene data with sequence and the indexes of complete pedestrians
            agent_data = [agent for agent in scene_data if agent.get("typ") == "complete"]
            
            # Create a ego scene representation derived form every full pedestrian
            for adx, _ in enumerate(agent_data):
                
                transformed_agents, target_grid = self.generate_ego_data(
                    all_agents=copy.deepcopy(agent_data),
                    adx=adx,
                    src=src
                )
                
                if not transformed_agents:
                    continue
                
                # Add data
                processed_data = {
                    "target_agent": transformed_agents[0],
                    "other_agents": transformed_agents[1:],
                    "vehicles": None,
                    "target_grid": target_grid,
                    "target_translation": transformed_agents[0]["translation"],
                    "target_rotation": transformed_agents[0]["rotation"],
                    "target_motion": transformed_agents[0]["motion"],
                    "lsa": None,
                    "source": f"{src}_{str(idx).zfill(5)}_{str(adx).zfill(3)}",
                    "flipped": False,
                    "num_agents": len(transformed_agents)
                }
                
                # Store at corresponding motion state bin
                motion_bins[processed_data['target_motion']].append(copy.deepcopy(processed_data))
                
                
        # Augment
        if self.augment:
            motion_bins = self.augment_data(d=motion_bins)
        
        final_data = []
        for _, tracks in motion_bins.items():
            
            for t in tracks:
            
                final_data.append(t)
            
        # Save data 
        with h5py.File(os.path.join(self.work_dir, f"{src}.h5"), 'w') as hf:
            
            # Variable-length arrays of type uint8
            dt = h5py.special_dtype(vlen=np.dtype('uint8'))
            
            # Dataset of length = number of samples, each entry is a variable-length array of bytes
            dset = hf.create_dataset('data', shape=(len(final_data),), dtype=dt)
            
            for i, sample in enumerate(final_data):
                
                # Pickle the Python dict
                pickled = pickle.dumps(sample)
                
                # Convert pickled bytes -> a uint8 array we can store
                dset[i] = np.frombuffer(pickled, dtype='uint8')
                
        print(f"{idx}: Processed set: {src} - Motion analysis: total: {len(final_data)} - is standing: {len(motion_bins['standing'])} - starting: {len(motion_bins['starting'])} - stopping: {len(motion_bins['stopping'])} - straight: {len(motion_bins['straight'])} - light right: {len(motion_bins['light_right'])} - strong right: {len(motion_bins['strong_right'])} - light left: {len(motion_bins['light_left'])} - strong left: {len(motion_bins['strong_left'])}")
        return len(final_data)
    
    
    def generate_ego_data(self, all_agents, adx, src):
        
        target_agent = all_agents[adx]
        other_agents = [element for i, element in enumerate(all_agents) if i != adx]
        t = target_agent['track'][self.obs_len-1]
        
        # Setup cost map generator
        occupancy_grid = Occupancy_Grid_ETH(seg_map=copy.deepcopy(self.seg_maps[src]), grid_size_px=GRID_SIZE_PX, H=self.homographs[src])
        
        # # Validate target track and map
        # if not occupancy_grid.validate_target(target=target_agent): 
        #     return [], None
        
        # Validate other tracks and create occupancy cost grid 
        occupancy_grid.create_static_layer()
        
        # Get cost map
        target_ego_grid = occupancy_grid.create_ego_grid(
            rotation_angle=target_agent['rotation'],
            translation=target_agent['translation'],
            radius=GRID_RADIUS
        )
        
        # Setup coordinate transformer
        ego_mapper = Ego_Transformer(translation_vector=target_agent['translation'], rotation_angle=target_agent['rotation'])
        
        # Filter all other agents and vehicles
        filtered_other_agents = filter_objects(target=target_agent, other=other_agents, obs_len=self.obs_len, r=CONTEXT_RADIUS)
        
        transformed_agents = []
        
        for k, obj in enumerate([target_agent] + filtered_other_agents):
        
            # Transform track data to target ego, handle nan´s correctly
            track_non_nan_mask = ~np.isnan(obj['track'])
            masked_track= obj['track'][track_non_nan_mask].reshape(-1,2)
            
            # Apply targets ego transformation
            masked_track = ego_mapper.apply(X=masked_track)
            
            # Apply normalization if enabled
            if self.norm:
                masked_track = ego_mapper.norm(X=masked_track, size=CONTEXT_RADIUS)
            
            # Determine the x and y velocities and accelerations
            v = np.diff(masked_track, axis=0) / self.delta_t
            velos = np.vstack([v[0], v])
            a = np.diff(velos, axis=0) / self.delta_t
            accel = np.vstack([a[0], a])
            masked_track = np.concatenate([masked_track, velos, accel], axis=1)
            
            # Expand observation to handle velocities and accelerations
            ego_track = np.zeros((obj['track'].shape[0], 6))
            track_non_nan_mask = np.concatenate([track_non_nan_mask, track_non_nan_mask, track_non_nan_mask], axis=1)
            ego_track[track_non_nan_mask] = masked_track.reshape(-1)
            obj['ego_track'] = ego_track
            transformed_agents.append(obj)
        
        return transformed_agents, target_ego_grid
    
    
    def augment_data(self, d):
        
        flipped_data = {}
        
        for k, agent_scene in d.items():
            
            flipped_motion_state = k
            
            if k == 'starting' or k == 'stopping' or k == 'strong_left' or k == 'strong_right':
                
                if k == 'strong_left' or k == 'strong_right':
                    
                    flipped_motion_state = k.replace("left", "TEMP").replace("right", "left").replace("TEMP", "right")
                
                agent_scene_flipped = copy.deepcopy(agent_scene)
                
                for scene in agent_scene_flipped:
                    
                    # target agent
                    scene['target_agent']['track'][:, [0]] *= -1
                    scene['target_agent']['ego_track'][:, [0,2,4]] *= -1
                    scene['target_agent']['motion'] = flipped_motion_state
                    
                    # other agents
                    for other in scene['other_agents']:
                        
                        other['track'][:, [0]] *= -1
                        other['ego_track'][:, [0,2,4]] *= -1
                    
                    # misc
                    scene['flipped'] = True
                    scene['target_grid'] = np.flip(scene['target_grid'], axis=1)
                    scene['motion'] = flipped_motion_state
                    
                flipped_data[flipped_motion_state] = agent_scene_flipped
                
            else:
                
                flipped_data[flipped_motion_state] = []
                
        combined_data = {key: d[key] + flipped_data[key] for key in d if key in flipped_data}
                
        return combined_data
    
    
    
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
    path = f"/workspace/data/benchmarks/{scene}"
    data_generator = Sample_Generator(path=path, dataset=scene, typ='train', obs_len=8, future_len=12, sample_rate=0.4, norm=False, augment=True, num_workers=12)
    data_generator = Sample_Generator(path=path, dataset=scene, typ='eval', obs_len=8, future_len=12, sample_rate=0.4, norm=False, augment=False, num_workers=12)
    data_generator = Sample_Generator(path=path, dataset=scene, typ='test', obs_len=8, future_len=12, sample_rate=0.4, norm=False, augment=False, num_workers=12)
    print(f"\n\n")
    
sys.exit()