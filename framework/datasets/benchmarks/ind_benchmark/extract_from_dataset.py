import numpy as np
import pandas as pd
import os
import pickle
import sys
import copy
import functools

sys.path.append('/workspace/repos/framework/datasets')

from joblib import Parallel, delayed
from utils import *
from common import *
from ind.utils import get_ind_rotated_bbox

# Definitions
IND_CLASS_DICT = {'pedestrian': 0, 'bicycle': 1, 'car': 6, 'truck_bus': 7}

# Ind specidic raw data structure
IND_COLS = ['recordingId', 'trackId', 'frame', 'trackLifetime', 'xCenter', 'yCenter', 'heading', 'width', 'length', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration', 'lonVelocity', 'latVelocity', 'lonAcceleration', 'latAcceleration']
IND_META_COLS = ['recordingId', 'trackId', 'initialFrame', 'finalFrame', 'numFrames', 'width', 'length', 'class']


# Split definition from Y-Net an DiLong
IND_TEST = [str(i).zfill(2) for i in [0,1,2,3,4,5,6]]
IND_EVAL = [str(i).zfill(2) for i in [0,1,2,3,4,5,6]]
IND_TRAIN = [str(i).zfill(2) for i in [7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]]

# Velocity limits
DT=0.04
MAX_VELO = 5.0
LIMIT = 0.1
STANDING_THRESHOLD = 0.375


def extract(obs_len, seq_len, step, min_peds, original_rate, resample_rate, work_path, seq_id, typ):
    
    scene_name = f"{typ}_{seq_id}"
    scene_data = {}
    print(f"\n - processing scene {scene_name}:")
    
    ped_complete_track_count = 0
    ped_incomplete_track_count = 0
    other_track_count = 0
    seq_count = 0
    ped_invalid_count = 0
    
    if resample_rate == None: standing_threshold = STANDING_THRESHOLD * original_rate
    else: standing_threshold = STANDING_THRESHOLD * resample_rate
    
    motion_classifier = Motion_Classifier(standing_threshold=standing_threshold)
    
    scene_df = pd.read_csv(os.path.join(work_path, f"{seq_id}_tracks.csv" ), header=0, names=IND_COLS, delimiter=',')
    meta_df = pd.read_csv(os.path.join(work_path, f"{seq_id}_tracksMeta.csv" ), header=0, names=IND_META_COLS, delimiter=',')
    
    # delete unnecessary columns
    # scene_df: ['trackId', 'frame', 'xCenter', 'yCenter', heading, 'width', 'length', 'xVelocity', 'yVelocity', 'class']
    scene_df = scene_df.drop(columns=['recordingId',
                                    'trackLifetime',
                                    'xAcceleration', 
                                    'yAcceleration', 
                                    'lonVelocity', 
                                    'latVelocity',
                                    'lonAcceleration', 
                                    'latAcceleration'])
    
    # add class id to data structure
    #scene_df['class'] = np.zeros_like(scene_df['trackId']).astype(int)
    scene_df['yCenter'] = -scene_df['yCenter']
    
    # meta_df: ['trackId', 'numFrames', 'class']
    meta_df = meta_df.drop(columns=['recordingId', 'initialFrame', 'finalFrame', 'width', 'length'])
    
    # Merge classes from meta file
    scene_df = scene_df.merge(
        meta_df[['trackId', 'class']], 
        on='trackId', 
        how='left', 
        validate='many_to_one'
    )
    
    # Map class names to class IDs using IND_CLASS_DICT
    scene_df['class_id'] = scene_df['class'].map(IND_CLASS_DICT)
    scene_df = scene_df.drop(columns=['class'], errors='ignore') 
    
    # Add placeholders and re-arrange
    scene_df['cuboid'] = None
    scene_df['timestamp'] = None
    scene_df = scene_df.reindex(columns=['frame', 'trackId', 'class_id', 'xCenter', 'yCenter', 'xVelocity', 'yVelocity', 'cuboid', 'timestamp' , 'heading', 'width', 'length'])
    
    # convert from pandas to numpy
    # scene_np: [frame id, object id, class_id, x_pos, y_pos, x_velo, y_velo, cuboid, timestamp, heading, obj_width, obj_length]
    scene_np = np.swapaxes(np.stack([scene_df[col].values for col in scene_df.columns],dtype=object), 1, 0)
    
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
            
            # extract all object data at current timestamp
            all_seq_obs = np.concatenate(frame_data[idx:idx + seq_len], axis=0)
            
            # get all pedestrians
            all_seq_peds = all_seq_obs[all_seq_obs[:, 2] == IND_CLASS_DICT['pedestrian'], :]
            
            # check if at least one pedestrian exists at current timestamp, else skip
            all_peds_ids = np.unique(all_seq_peds[:, 1])
            if len(all_peds_ids) < 1: continue
            
            # get all other objects
            all_seq_diff = all_seq_obs[all_seq_obs[:, 2] != IND_CLASS_DICT['pedestrian'], :]
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
            
            # data structure: [frame id, object id, class_id, x_pos, y_pos, x_velo, y_velo, cuboid, timestamp, heading, obj_width, obj_length]
            for d in all_seq_full_peds:
                
                dataset_class_name = get_key(d=IND_CLASS_DICT, v=d[0,2])
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
                
                dataset_class_name = get_key(d=IND_CLASS_DICT, v=d[new_start,2])
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
                
                dataset_class_name = get_key(d=IND_CLASS_DICT, v=d[new_start,2])
                target_class_id = TARGET_CLASS_DICT[dataset_class_name]
                target_class_name = get_key(d=TARGET_CLASS_DICT, v=target_class_id)
                trajectory = d[:,3:5].astype(float).copy()
                
                # Resample
                if resample_rate is not None:
                    trajectory, _ = resample_tracks(trajectories=[trajectory], obs_len=obs_len, seq_len=seq_len, original_rate=original_rate, resample_rate=resample_rate)
                
                # Cuboid handling
                if target_class_id == TARGET_CLASS_DICT['car'] or target_class_id == TARGET_CLASS_DICT['truck_bus']:
                    
                    cuboid = get_ind_rotated_bbox(center=np.array([d[obs_len-1,3], d[obs_len-1,4]]), length=d[obs_len-1,11], width=d[obs_len-1,10], heading=d[obs_len-1,9])
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
                    
        data_storage.append(data_dict)
    
    # merge single dicts
    temp = functools.reduce(lambda x, y: {**x, **y}, data_storage)
    
    # remove empty timestamps
    scene_data[scene_name] = {k: v for k, v in temp.items() if v != []}
    
    # console feedback
    print(f' - {scene_name}: extracted sequence samples: {len(scene_data[scene_name])}')
    return [scene_data, ped_complete_track_count, ped_incomplete_track_count, other_track_count, seq_count, ped_invalid_count]


def extract_ind_data(path, min_peds, obs_len, pred_len, step, typ, original_rate, resample_rate, num_workers):
    """load raw data splits, sort and process in timestamp order
    
    Args:
        path (str): raw dataset dir
        min_peds (int, optional): number of minimal available pedestrians. Defaults to 1.
        obs_len (int, optional): _description_. Defaults to 8.
        pred_len (int, optional): _description_. Defaults to 12.
        delta_t (float, optional): _description_. Defaults to 0.4.
        step (int, optional): _description_. Defaults to 1.
        typ (str, optional): _description_. Defaults to 'train'.
    """
    
    work_path = os.path.join(path, 'raw')
    dest_path = os.path.join(path,)
    seq_len = obs_len + pred_len
    if not os.path.exists(dest_path): os.makedirs(dest_path)
    
    if typ == 'train': ids = IND_TRAIN
    elif typ == 'test': ids = IND_TEST
    elif typ == 'eval': ids = IND_EVAL
    
    print(f"\nprocessing ind: {typ} set")
    
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
        seq_id=id, 
        typ=typ)
        for id in ids
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
    with open(os.path.join(dest_path, f"ind_{typ}_extracted.pickle"), 'wb') as f:
        pickle.dump(scene_data, f)
    
    return


#--- inD is 25.0 Hz native:
path = '/workspace/data/benchmarks/ind'

#--- Procedure:
# 1.) Extract from raw source data with: extract_from_dataset.py
# 2.) Generate Database using: generate_database.py
# 3.) Generate final dataset using: generate_set.py

#--- Target settings:
# Defined by Y-Net and DiLong: 
# Input: 3.2s
# Output: 4.8s 
# Downsampling: to 2.5 Hz
# Sliding window step: 1s -> 25 steps
# Only Train and Test Splits, Eval as copy (dummy) of Test
# Only Pedestrians

extract_ind_data(path, min_peds=1, obs_len=80, pred_len=120, step=25, typ='test', original_rate= 0.04, resample_rate=0.4, num_workers=16)
extract_ind_data(path, min_peds=1, obs_len=80, pred_len=120, step=25, typ='train', original_rate= 0.04, resample_rate=0.4, num_workers=16)

sys.exit() 