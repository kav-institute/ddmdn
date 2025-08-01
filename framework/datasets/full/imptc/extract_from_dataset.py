import numpy as np
import os
import pickle
import sys
import functools
import copy

sys.path.append('/workspace/repos/framework')
sys.path.append('/workspace/repos/framework/datasets')

from joblib import Parallel, delayed
from imptc import *
from common import *
from transformation import *

# Definitions
IMPTC_CLASS_DICT = {'pedestrian': 0, 'bicycle': 2, 'motorbike': 3, 'scooter': 4, 'stroller': 5, 'wheelchair': 6, 'car': 7, 'truck_bus': 8}

# Data spits
IMPTC_TRAIN = ["set_01","set_03"]
IMPTC_EVAL = ["set_02", "set_04"]
IMPTC_TEST = ["set_05"]

# Velocity limits
DT=0.04
MAX_VELO = 2.5 # 100 px/s
LIMIT = 0.1
STANDING_THRESHOLD = 0.375

def extract(obs_len, seq_len, step, min_peds, original_rate, resample_rate, work_path, set, seq, typ, index):
    
    scene_name =f"{seq}_{set}"
    scene_data = {}
    
    ped_complete_track_count = 0
    ped_incomplete_track_count = 0
    other_track_count = 0
    seq_count = 0
    ped_invalid_count = 0
    
    if resample_rate == None: standing_threshold = STANDING_THRESHOLD * original_rate
    else: standing_threshold = STANDING_THRESHOLD * resample_rate
    
    seq_loader = IMPTC_SequenceLoader(src_path=os.path.join(work_path, seq, set))
    motion_classifier = Motion_Classifier(standing_threshold=standing_threshold)
    
    # scene_np: [frame id, object id, class_id, x_pos, y_pos, x_velo, y_velo, cuboid, timestamp]
    scene_np, lsa_data = seq_loader.load_data()
    
    data_list = []
    data_storage = []
    timestamp_list = check_timestamps(timestamps=np.unique(scene_np[:, 0]).tolist())
    for stamps in timestamp_list: data_list.append([scene_np[ts == scene_np[:, 0], :] for ts in stamps])
    
    for frame_data in data_list:
            
        timestamps = np.unique(np.vstack(frame_data)[:, 0])
        data_dict = {f"{timestamp}": [] for timestamp in timestamps}
        
        for idx in range(0, len(timestamps), step):
            
            # end of usable scene data
            if idx >= len(timestamps) - obs_len: continue
            
            # check sample length
            if len(frame_data[idx:idx + seq_len]) < 1: continue
            timestamp_id = timestamps[idx]
            
            # get best fitting lsa status for track split
            ts = frame_data[idx+obs_len][0,8]
            closest = min(lsa_data.keys(), key=lambda k: abs(float(k) - ts))
            lsa_status = lsa_data[closest]
            
            # extract all object data at current timestamp
            all_seq_obs = np.concatenate(frame_data[idx:idx + seq_len], axis=0)
            
            # get all pedestrians
            all_seq_peds = all_seq_obs[all_seq_obs[:, 2] == IMPTC_CLASS_DICT['pedestrian'], :]
            
            # check if at least one pedestrian exists at current timestamp, else skip
            all_peds_ids = np.unique(all_seq_peds[:, 1])
            if len(all_peds_ids) < 1: continue
            
            # get all other objects
            all_seq_diff = all_seq_obs[all_seq_obs[:, 2] != IMPTC_CLASS_DICT['pedestrian'], :]
            all_other_ids = np.unique(all_seq_diff[:, 1])
            
            # split pedestrians into complete (exist during the whole sequence) and in-complete
            all_seq_incomplete_peds = []
            all_seq_full_peds = []
            all_seq_others = []
            
            # check for all pedestrians
            for ped_id in all_peds_ids:
                
                # get single pedestrian
                curr_ped_seq = all_seq_peds[all_seq_peds[:, 1] == ped_id, :]
                if len(curr_ped_seq) != seq_len: all_seq_incomplete_peds.append(curr_ped_seq)
                else: all_seq_full_peds.append(curr_ped_seq)
                
            for other_id in all_other_ids:
            
                # get single other object
                curr_other_seq = all_seq_diff[all_seq_diff[:, 1] == other_id, :]
                all_seq_others.append(curr_other_seq)
                
            # check if we have at least one complete predestrian, else skip
            if len(all_seq_full_peds) < min_peds: continue
            else: seq_count += 1
            
            # check if complete pedestrian has a reasonable velocity
            all_seq_full_peds, filter_cnt = check_velocity_exceedance(tracks=all_seq_full_peds, dt=DT, max_velo=MAX_VELO, exceeding_limit=LIMIT)
            ped_invalid_count += filter_cnt
            
            for d in all_seq_full_peds:
                
                # data structure: [frame id, object id, class_id, x_pos, y_pos, x_velo, y_velo, cuboid, timestamp]
                dataset_class_name = get_key(d=IMPTC_CLASS_DICT, v=d[0,2])
                target_class_id = TARGET_CLASS_DICT[dataset_class_name]
                target_class_name = get_key(d=TARGET_CLASS_DICT, v=target_class_id)
                trajectory = d[:,3:5].astype(float).copy()
                observation_length = copy.deepcopy(obs_len)
                
                # Resample
                if resample_rate is not None:
                    trajectory, observation_length = resample_tracks(trajectories=[trajectory], obs_len=obs_len, seq_len=seq_len, original_rate=original_rate, resample_rate=resample_rate)
                
                motion_state, translation_vector, rotation_angle = motion_classifier.classify(T=trajectory, obs_len=observation_length)
                
                obj_data = {'typ': 'complete',
                            'class_id': target_class_id,
                            'class_name': target_class_name,
                            'track_id': d[0,1],
                            'track': trajectory,
                            'motion': motion_state,
                            'translation': translation_vector,
                            'rotation': rotation_angle
                            }
                
                data_dict[f"{timestamp_id}"].append(obj_data)
                ped_complete_track_count += 1
                
            # incomplete pedestrian handling
            for curr_ped_seq in all_seq_incomplete_peds:
                    
                d = np.zeros((seq_len, scene_np.shape[1]), dtype=object)
                d.fill(np.nan)
                start = int(curr_ped_seq[0,0]) - timestamps[idx]
                end = start + curr_ped_seq.shape[0]
                d[start:end,:] = curr_ped_seq
                
                p = np.where(~np.isnan(d[...,0:5].astype(float)).any(axis=1))[0]
                if p.any():
                    new_start = p[0]
                    new_end = p[-1] + 1
                    
                else: 
                    ped_invalid_count += 1
                    continue
                
                # check incompleteness - usability check
                if not incomplete_usability_check(track=d[:,3:5].astype(float), obs_len=obs_len): 
                    ped_invalid_count += 1
                    continue
                
                dataset_class_name = get_key(d=IMPTC_CLASS_DICT, v=d[new_start,2])
                target_class_id = TARGET_CLASS_DICT[dataset_class_name]
                target_class_name = get_key(d=TARGET_CLASS_DICT, v=target_class_id)
                trajectory = d[:,3:5].astype(float).copy()
                
                # Resample
                if resample_rate is not None:
                    trajectory, _ = resample_tracks(trajectories=[trajectory], obs_len=obs_len, seq_len=seq_len, original_rate=original_rate, resample_rate=resample_rate)
                
                obj_data = {'typ': 'incomplete',
                            'class_id': target_class_id,
                            'class_name': target_class_name,
                            'track_id': d[new_start,1],
                            'track': trajectory
                            }
                
                data_dict[f"{timestamp_id}"].append(obj_data)
                ped_incomplete_track_count += 1
                
            # other object handling
            for curr_other_seq in all_seq_others:
                
                d = np.zeros((seq_len, scene_np.shape[1]), dtype=object)
                d.fill(np.nan)
                start = int(curr_other_seq[0,0]) - timestamps[idx]
                end = start + curr_other_seq.shape[0]
                d[start:end,:] = curr_other_seq
                
                p = np.where(~np.isnan(d[...,0:5].astype(float)).any(axis=1))[0]
                if p.any(): 
                    new_start = p[0]
                    new_end = p[-1] + 1
                    
                else: continue
                
                # check incompleteness - usability check
                if not incomplete_usability_check(track=d[:,3:5].astype(float), obs_len=obs_len): 
                    continue
                
                dataset_class_name = get_key(d=IMPTC_CLASS_DICT, v=d[new_start,2])
                target_class_id = TARGET_CLASS_DICT[dataset_class_name]
                target_class_name = get_key(d=TARGET_CLASS_DICT, v=target_class_id)
                trajectory = d[:,3:5].astype(float).copy()
                
                # Resample
                if resample_rate is not None:
                    trajectory, _ = resample_tracks(trajectories=[trajectory], obs_len=obs_len, seq_len=seq_len, original_rate=original_rate, resample_rate=resample_rate)
                
                # Cuboid handling
                if d[obs_len-1,7] is not None:
                    cuboid = np.array(d[obs_len-1,7])
                else:
                    cuboid = None
                
                obj_data = {'typ': 'other',
                            'class_id': target_class_id,
                            'class_name': target_class_name,
                            'track_id': d[new_start,1],
                            'track': trajectory,
                            'cuboid': cuboid
                            }
                
                data_dict[f"{timestamp_id}"].append(obj_data)
                other_track_count += 1
                
            data_dict[f"{timestamp_id}"].append({'typ': 'lsa', 'status': lsa_status})
                
        data_storage.append(data_dict)
            
    # merge single dicts
    temp = functools.reduce(lambda x, y: {**x, **y}, data_storage)
    
    # remove empty timestamps
    scene_data[scene_name] = {k: v for k, v in temp.items() if v != []}
    
    # console feedback
    print(f'{index}: {scene_name}: extracted sequence samples: {len(scene_data[scene_name])}')
    return [scene_data, ped_complete_track_count, ped_incomplete_track_count, other_track_count, seq_count, ped_invalid_count]


def extract_imptc_data(path, min_peds, obs_len, pred_len, step, typ, original_rate, resample_rate, num_workers):
    
    work_path = os.path.join(path, 'raw')
    dest_path = os.path.join(path,)
    seq_len = obs_len + pred_len
    set_list = []
    seq_list = []
    if not os.path.exists(dest_path): os.makedirs(dest_path)
    
    if typ == 'train': sets = IMPTC_TRAIN
    elif typ == 'test': sets = IMPTC_TEST
    elif typ == 'eval': sets = IMPTC_EVAL
    
    print(f"\nprocessing imptc: {typ} set")
    
    # iterate imptc sets
    for set in sets:
        
        set_path = os.path.join(work_path, set)
        set_list.extend([entry.name for entry in os.scandir(set_path) if entry.is_dir()])
        seq_list.extend([set for entry in os.scandir(set_path) if entry.is_dir()])
        
    ped_complete_track_count = 0
    ped_incomplete_track_count = 0
    other_track_count = 0
    seq_count = 0
    ped_invalid_count = 0
    scene_data_list = []
    
    # Parallel processing over sources
    results = Parallel(n_jobs=num_workers)(delayed(extract)(
        obs_len=obs_len,
        seq_len=seq_len, 
        step=step, 
        min_peds=min_peds,
        original_rate=original_rate,
        resample_rate=resample_rate,
        work_path=work_path, 
        set=set_list[k],
        seq=seq_list[k],
        typ=typ,
        index=f"[{k+1}/{len(set_list)}]")
        for k, _ in enumerate(set_list)
    )
    
    for res in results:
        
        scene_data_list.append(res[0])
        ped_complete_track_count += res[1]
        ped_incomplete_track_count += res[2]
        other_track_count += res[3]
        seq_count += res[4]
        ped_invalid_count += res[5]
        
    scene_data = functools.reduce(lambda x, y: {**x, **y}, scene_data_list)
    
    # console feedback
    print(f"\nset: {typ} - total sequence samples: {seq_count} - num complete tracks: {ped_complete_track_count} - num incomplete tracks: {ped_incomplete_track_count} - num other tracks: {other_track_count} - num invalid tracks: {ped_invalid_count}")
    
    # save data and meta data to files
    with open(os.path.join(dest_path, f"imptc_{typ}_extracted.pickle"), 'wb') as f:
        pickle.dump(scene_data, f)
    
    return


#--- IMPTC is 25.0 Hz native:
path = '/workspace/data/full/imptc/'

#--- Procedure:
# 1.) Extract from raw source data with: extract_from_dataset.py
# 2.) Generate Database using: generate_database.py
# 3.) Generate final dataset using: generate_set.py

#--- Target settings:
# Input: 1.0s
# Output: 4.0s
# Downsampling: to 10.0 Hz
extract_imptc_data(path, min_peds=1, obs_len=25, pred_len=100, step=1, typ='eval', original_rate= 0.04, resample_rate=0.1, num_workers=16)
extract_imptc_data(path, min_peds=1, obs_len=25, pred_len=100, step=1, typ='test', original_rate= 0.04, resample_rate=0.1, num_workers=16)
extract_imptc_data(path, min_peds=1, obs_len=25, pred_len=100, step=1, typ='train', original_rate= 0.04, resample_rate=0.1, num_workers=16)
sys.exit()