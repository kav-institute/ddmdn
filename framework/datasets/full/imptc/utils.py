import os
import json
import math
import cv2
import numpy as np
import imutils

from skimage.draw import polygon
from common import *

# Segmentation
IMPTC_SEG_DICT = {'unlabeled': (0,0,0),
                'blocking': (70,70,71),
                'terrain': (81,0,81),
                'road': (128,64,128),
                'sidewalk': (244,35,232),
                'bikelane': (98,160,234),
                'crosswalk': (229,165,10),
                'vegetation': (107,142,35)
                }


IMPTC_SEG_LSA_POLYS = {
    "f1": [[574, 782], [783, 629], [804, 651], [828, 668], [855, 679], [878, 684], [903, 686], [610, 901], [609, 884], [607, 863], [600, 835], [592, 814], [579, 790]],
    "f2": [[983, 675], [1005, 663], [1087, 598], [1389, 989], [1306, 1056], [1292, 1074]],
    "f3": [[985, 1412], [1251, 1202], [1253, 1222], [1258, 1244], [1270, 1268], [1283, 1287], [1301, 1309], [1076, 1486], [1042, 1447], [1026, 1431], [1006, 1418]]
}

IMPTC_MAP_LSA_POLYS = { 
    "f1": [[374, 582], [583, 429], [604, 451], [628, 468], [655, 479], [678, 484], [703, 486], [410, 701], [409, 684], [407, 663], [400, 635], [392, 614], [379, 590]],
    "f2": [[783, 475], [805, 463], [887, 398], [1189, 789], [1106, 856], [1092, 874]],
    "f3": [[785, 1212], [1051, 1002], [1053, 1022], [1058, 1044], [1070, 1068], [1083, 1087], [1101, 1109], [876, 1286], [842, 1247], [826, 1231], [806, 1218]]
        }


class Occupancy_Grid_IMPTC(Occupancy_Grid):
    
    def __init__(self, seg_map, scale, radius, grid_size_px): 
        
        super().__init__(seg_map)
        
        self.lsa_red_value = 0.9
        
        # Seg map transformation parameters
        self.scale = scale
        self.radius = radius * scale
        self.gamma = 0.57 * math.pi
        self.grid_size_px = grid_size_px
        
        # translation vector
        self.t = np.matrix([[960], [647]])
        
        # Rotation matrix
        self.R = np.matrix([[math.cos(self.gamma), -math.sin(self.gamma)], [math.sin(self.gamma), math.cos(self.gamma)]])
        return
    
    
    def transform_point(self, p):
        """ Transform a 3d point from imptc world coords into 2d map coords
        
        Args:
            p (np.array): 3d ind position, only x and y
        Returns:
            np.array: transformed u,v map pixel coordinates
        """
        
        # scale, rotate, translate
        p = np.matrix([p[0], -p[1]])
        p *= self.scale
        p = self.R * p.transpose()
        p += self.t
        
        return np.array(p).reshape(-1).astype(int)
    
    
    def create_static_layer(self, lsa):
        
        # convert segmentation map to cost probabilities
        for seg_value, prob_value in self.probability_mapping.items():
            self.static_layer[self.map == seg_value] = prob_value
            
        # check pedestrian crosswalks signal status
        for f in ['f1', 'f2', 'f3']:
            
            # if red, mark as heavy blocking area
            if lsa['status'][f] == 10:
                
                # get the indices of the pixels inside the area polygon
                area = IMPTC_SEG_LSA_POLYS.get(f)
                cols, rows = zip(*area)
                rr, cc = polygon(rows, cols)
                
                # fill the polygon area with the specified value
                self.static_layer[rr, cc] = self.lsa_red_value
        
        return
    
    
    def create_object_layer(self, others):
        
        for obj in others:
            
            # Car or Truck
            if obj['class_id'] == TARGET_CLASS_DICT['car'] or obj['class_id'] == TARGET_CLASS_DICT['truck_bus']:
                
                cuboid = [self.transform_point(p=p) for p in obj['cuboid']]
                cols, rows = zip(*[cuboid[0], cuboid[1], cuboid[2], cuboid[3]])
                rr, cc = polygon(rows, cols)
                
                # Create a mask for valid coordinates
                valid_mask = (rr >= 0) & (rr < self.h) & (cc >= 0) & (cc < self.w)
                rr = rr[valid_mask]
                cc = cc[valid_mask]
                
                self.object_layer[rr, cc] = 1.0
        
        return
    
    
    def create_ego_grid(self, rotation_angle, translation, apply_rot=False):
        
        roi, _ = self.create_grid_roi(translation=translation)
        
        # Rotate
        if apply_rot: 
            
            roi = imutils.rotate(image=roi, angle=math.degrees(self.gamma) + math.degrees(rotation_angle))
            
        # Resize to target grid size
        ego_grid = cv2.resize(roi, (self.grid_size_px, self.grid_size_px), interpolation=cv2.INTER_NEAREST)
        return ego_grid



class IMPTC_SequenceLoader(object):
    
    def __init__(self, src_path):
        
        # Initialization of variables
        self.vehicle_track_dict = {}
        self.vru_track_dict = {}
        self.data = {}
        
        # Get path of context, vehicles, vrus
        self.vehicles_path = os.path.join(src_path, 'vehicles')
        self.vrus_path = os.path.join(src_path, 'vrus')
        self.lsa_path = os.path.join(src_path, 'context', 'traffic_light_signals.json')
        
        return
    
    
    def load_data(self):
        
        object_id_cnt = 0
        data_collector = []
        
        ## Load vru data ##
        # List all subfolders (= number of vrus)
        self.vru_list = sorted(os.listdir(self.vrus_path))
        
        for vru in self.vru_list:
            
            vru_path = os.path.join(self.vrus_path, vru)
            
            with open(vru_path + '/track.json', 'r') as file:
                
                vru_track = json.load(file)
                
                class_id = vru_track["overview"]["class_id"]
                
                for id in vru_track['track_data']:
                    
                    d = vru_track['track_data'][id]
                    
                    # [frame id, object id, class_id, x_pos, y_pos, x_velo, y_velo, cuboid, timestamp]
                    data_collector.append(np.array([0, object_id_cnt, class_id, d["coordinates"][0], d["coordinates"][1], None, None, None, int(d["ts"])], dtype=object))
                    
            object_id_cnt += 1
        
        ## Load vehicle data ##
        # List all subfolders (= number of vehicles)
        self.vehicle_list = sorted(os.listdir(self.vehicles_path))
        
        for vehicle in self.vehicle_list:
            
            vehicle_path = os.path.join(self.vehicles_path, vehicle)
            
            with open(vehicle_path + '/track.json', 'r') as file:
                
                vehicle_track = json.load(file)
                class_id = vehicle_track["overview"]["class_id"]
                
                for id in vehicle_track['track_data']:
                    
                    d = vehicle_track['track_data'][id]
                    
                    # Get four ground positions of 3D-cuboid
                    cuboid = []
                    cuboid.append(d['cuboid'][0][0][:2])
                    cuboid.append(d['cuboid'][1][1][:2])
                    cuboid.append(d['cuboid'][6][1][:2])
                    cuboid.append(d['cuboid'][2][1][:2])
                    cuboid = [tuple(point) for point in cuboid]
                    
                    # [frame id, object id, class_id, x_pos, y_pos, x_velo, y_velo, cuboid, timestamp]
                    data_collector.append(np.array([0, object_id_cnt, class_id, d["coordinates"][0], d["coordinates"][1], None, None, cuboid, int(d["ts"])], dtype=object))
                    
            object_id_cnt += 1
            
        ## Load traffic light signal data ##
        with open(self.lsa_path, 'r') as file:
                
            lsa_data = json.load(file)
            
        # Stack the arrays into a single array
        combined_data = np.vstack(data_collector)
        
        # Sort by the first column (timestamps)
        sorted_data = combined_data[combined_data[:, -1].argsort()]
        
        # Extract sorted timestamps
        timestamps = sorted_data[:, -1]
        
        # Create a mapping of unique timestamps to sequential numbers
        unique_timestamps = np.unique(timestamps)
        timestamp_to_number = {timestamp: idx for idx, timestamp in enumerate(unique_timestamps)}
        
        # Replace timestamps with corresponding numbers
        sorted_data[:, 0] = [timestamp_to_number[timestamp] for timestamp in timestamps]
        
        return sorted_data, lsa_data['status_data']