import numpy as np
import math


from shapely.geometry import Point, Polygon
from shapely.affinity import scale
from common import *


class Occupancy_Grid:
    
    def __init__(self, seg_map):
        
        self.map = seg_map
        self.static_layer = np.zeros_like(self.map, dtype=np.float32)
        self.object_layer = np.zeros_like(self.map, dtype=np.float32)
        self.h = self.map.shape[0]
        self.w = self.map.shape[1]
        
        # Static occupancy probabilities
        self.probability_mapping = {
            TARGET_SEG_DICT['unlabeled']: 0.0,
            TARGET_SEG_DICT['blocking']: 1.0,
            TARGET_SEG_DICT['terrain']: 0.25,
            TARGET_SEG_DICT['road']: 0.5,
            TARGET_SEG_DICT['sidewalk']: 0.0,
            TARGET_SEG_DICT['bikelane']: 0.5,
            TARGET_SEG_DICT['crosswalk']: 0.0,
            TARGET_SEG_DICT['vegetation']: 0.25
        }
        
        return
    
    
    def transform_point(self):
        
        raise NotImplementedError("Subclasses must implement this method")
    
    
    def create_static_layer(self):
        
        raise NotImplementedError("Subclasses must implement this method")
    
    
    def create_object_layer(self):
        
        raise NotImplementedError("Subclasses must implement this method")
    
    
    def create_ego_grid(self):
        
        raise NotImplementedError("Subclasses must implement this method")
    
    
    def validate_target(self, target):
        
        # Get target track in uv grid map coordinates
        uv_track = np.array([self.transform_point(p=pos) for pos in target['track']])
        filter_mask = (uv_track[...,0] >= 0) & (uv_track[...,0] < self.w) & (uv_track[...,1] >= 0) & (uv_track[...,1] < self.h)
        uv_track_filtered = uv_track[filter_mask]
        
        # Check if track moves into a forbidden space
        if any(self.map[row, col] == 1 for col, row in uv_track_filtered):
            
            return False
        
        else:
            
            return True
        
        
    def validate_tracks(self, agents, others, obs_len):
        
        vehicles = []
        vehicles_poly = []
        
        for obj in others:
            
            if obj['class_id'] == TARGET_CLASS_DICT['car'] or obj['class_id'] == TARGET_CLASS_DICT['truck_bus']: 
                
                cuboid = [self.transform_point(p=p) for p in obj['cuboid']]
                vehicles.append(obj)
                
                # Get vehicle cuboid as polygon and increase area
                poly = Polygon([cuboid[0], cuboid[1], cuboid[2], cuboid[3]])
                poly = scale(poly, xfact=1.41, yfact=1.41, origin=poly.centroid)
                vehicles_poly.append(poly)
                
        # check if a vehicle cuboid and a vru position match with each other
        k = 0
        while k < len(vehicles_poly):
            
            flag = False
                
            # if they match delete the vehicle object, because its a failure
            if np.any(vehicles_poly[k].contains([Point(a['track'][obs_len-1][0], a['track'][obs_len-1][1]) for a in agents])):
                del vehicles[k]
                del vehicles_poly[k]
                flag = True
                    
            if not flag:
                k += 1
        
        # return combined objects
        return vehicles
    
    
    def create_grid_roi(self, translation):
        
        # Calc top-left and bottom-right coordinates to create a bbox
        p = np.squeeze(np.array(self.transform_point(p=translation)))
        tl = np.array([p[0] - self.radius, p[1] - self.radius])
        br = np.array([p[0] + self.radius, p[1] + self.radius])
        mx = self.map.shape[1]
        my = self.map.shape[0]
        
        # Combine static and object map
        obj_map = self.object_layer
        sta_map = self.static_layer
        
        object_mask = obj_map > 0.5
        sta_map[object_mask] = 1.0
        padded_grid = np.clip(sta_map, None, 1.0)
        
        # Left side padding
        if tl[0] < 0:
            
            # ((top, bottom),(left, right))
            p = math.ceil(abs(tl[0]))
            padded_grid = np.pad(array=padded_grid, pad_width=((0,0),(p,0)), mode='constant', constant_values=1.0)
            tl[0] = 0
            br[0] = br[0] + p
            
        # Top side padding
        if tl[1] < 0:
            
            # ((top, bottom),(left, right))
            p = math.ceil(abs(tl[1]))
            padded_grid = np.pad(array=padded_grid, pad_width=((p,0),(0,0)), mode='constant', constant_values=1.0)
            tl[1] = 0
            br[1] = br[1] + p
            
        # Right side padding
        if br[0] > mx:
            
            # ((top, bottom),(left, right))
            p = math.ceil(abs(br[0]) - mx)
            padded_grid = np.pad(array=padded_grid, pad_width=((0,0),(0,p)), mode='constant', constant_values=1.0)
            
        # Bottom side padding
        if br[1] > my:
            
            # ((top, bottom),(left, right))
            p = math.ceil(abs(br[1]) - my)
            padded_grid = np.pad(array=padded_grid, pad_width=((0,p),(0,0)), mode='constant', constant_values=1.0)
        
        # Get roi
        roi_padded_grid = padded_grid[int(tl[1]):int(br[1]), int(tl[0]):int(br[0])]
        return roi_padded_grid, tl