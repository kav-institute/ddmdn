import os
import sys
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import defaultdict
import torch
import pickle

sys.path.append('/workspace/repos/framework')
sys.path.append('/workspace/repos/framework/datasets')

from common import *


def seq_collate(batch):
    
    peds_num_list = []
    pre_motion_3D_list = []
    fut_motion_3D_list = []
    traj_scale_list = []
    source_list = []
    
    for idx, sample in enumerate(batch):
        
        (peds_num, pre_motion_3D, fut_motion_3D, traj_scale, source) = sample
        
        peds_num_list.append(peds_num)
        pre_motion_3D_list.append(pre_motion_3D)
        fut_motion_3D_list.append(fut_motion_3D)
        traj_scale_list.append(traj_scale)
        source_list.append(source)
        
    peds_num = torch.Tensor(peds_num_list).reshape(-1)
    pre_motion_3D = torch.cat(pre_motion_3D_list, dim=0)
    fut_motion_3D = torch.cat(fut_motion_3D_list, dim=0)
    
    data = {
        'peds_num_per_scene': peds_num,
        'pre_motion_3D': pre_motion_3D,
        'fut_motion_3D': fut_motion_3D,
        'traj_scale': torch.cat(traj_scale_list, dim=0),
        'sources': source_list,
        'seq': 'sdd',
    }
    
    return data

class TrajDataset(Dataset):
    """Dataloder for the SDD dataset"""
    
    def __init__(self, dir, obs_len=8, pred_len=12, set_name='sdd', typ='train'):
        
        super(TrajDataset, self).__init__()
        
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        
        data_root = os.path.join(dir,set_name, 'pkl', f"{set_name}_{typ}.pkl")
        print("Loading dataset: ", data_root)
        
        with open(data_root, "rb") as f:
            self.raw_data = pickle.load(f)
        
        self.data = self.raw_data[0]                   
        self.numPeds_in_sequence = self.raw_data[1]    # sum = data.shape[0]
        self.numSeqs_in_scene = self.raw_data[2]       # sum = len(numPeds in each sequence)
        self.sourceSeqs_in_scene = self.raw_data[4] 
        self.data_len = sum(self.numSeqs_in_scene)
        self.traj_abs = torch.from_numpy(self.data).type(torch.float)     
        self.traj_scales = torch.from_numpy(self.raw_data[3]).type(torch.float)
        
        
    def __len__(self):
        
        return self.data_len
    
    
    def __getitem__(self, index):
        
        # index in self.numPeds_in_sequence: number of peds in current scene
        peds_num = self.numPeds_in_sequence[index]
        srt_idx = sum(self.numPeds_in_sequence[:index])
        end_idx = sum(self.numPeds_in_sequence[:index+1])
        source = self.sourceSeqs_in_scene[index]
        curr_traj = self.traj_abs[srt_idx:end_idx, :, :]
        
        pre_motion_3D = curr_traj[:, :self.obs_len, :]
        fut_motion_3D = curr_traj[:, self.obs_len:, :]
        
        out = [torch.Tensor([peds_num]), pre_motion_3D, fut_motion_3D, self.traj_scales[srt_idx:end_idx], source]
        return out
    

if __name__ == "__main__":

    #--- Procedure:
    # 1.) Load raw data and save as .pkl files like Social-LSTM and Social-GAN with: preprocess.py
    # 2.) Extract from raw sources with: extract_from_dataset.py
    # 3.) Generate Database using: generate_database.py
    # 4.) Generate final dataset using: generate_set.py
    
    #--- Target settings:
    # Defined by Social-LSTM and Social-GAN
    # Input: 3.2s -> 8 steps
    # Output: 4.8s -> 12 steps
    # SampleRate: to 2.5 Hz
    # Only Pedestrians
    
    obs_len = 8
    pred_len = 12
    sample_rate = 0.4
    standing_threshold = 0.375 * sample_rate
    
    set_names = ['sdd']
    splits = ['train', 'test']
    root_dir = '/workspace/data/benchmarks'
    
    motion_classifier = Motion_Classifier(standing_threshold=standing_threshold)
    
    
    for set_name in set_names:
        
        work_dir = os.path.join(root_dir, set_name)
        
        for split in splits:
            
            timestamp = 0
            track_id = 0
            data_dict = {}
            sources = []
            
            dset = TrajDataset(
                dir=root_dir,
                obs_len=obs_len,
                pred_len=pred_len,
                set_name=set_name,
                typ=split)
            
            train_loader = DataLoader(
                dset,
                batch_size=128,
                shuffle=False,
                num_workers=1,
                collate_fn=seq_collate,
                pin_memory=False,
                drop_last=False)
            
            for data in train_loader:
                
                scene_start = 0
                sources.append(data['sources'])
                
                for scene_cnt in data['peds_num_per_scene']:
                    
                    scene_end = scene_start + int(scene_cnt.item())
                    
                    in_data = data['pre_motion_3D'][scene_start:scene_end, ...]
                    out_data = data['fut_motion_3D'][scene_start:scene_end, ...]
                    all_data = torch.cat([in_data, out_data], dim=1)
                    
                    for a, track in enumerate(all_data):
                        
                        motion_state, translation_vector, rotation_angle = motion_classifier.classify(T=track.numpy(), obs_len=obs_len)
                        
                        obj_data = {'typ': 'complete',
                            'class_id': 0,
                            'class_name': 'pedestrian',
                            'track_id': track_id,
                            'track': track.numpy(),
                            'motion': motion_state,
                            'translation': translation_vector,
                            'rotation': rotation_angle
                            }
                        
                        track_id += 1
                        
                        if a == 0: data_dict[timestamp] = [obj_data]
                        else: data_dict[timestamp].append(obj_data)
                        
                    timestamp += 1
                    scene_start = scene_start + int(scene_cnt.item())
                    
            # console feedback
            print(f"{set_name}_{split}: - total sequences: {len(data_dict)} - num total tracks: {track_id}\n")
            sources_flat = [item for sublist in sources for item in sublist]
            
            clusters = defaultdict(list)
            for idx, item in data_dict.items():
                cls = sources_flat[idx]
                clusters[cls].append(item)
            
            clusters = dict(clusters)
            
            # save data and meta data to files
            with open(os.path.join(work_dir, f"{set_name}_{split}_extracted.pickle"), 'wb') as f:
                pickle.dump(clusters, f)
