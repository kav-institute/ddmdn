import math
import cv2
import numpy as np
import imutils

from skimage.draw import polygon
from common import *

# Segmentation
SDD_SEG_DICT = {'unlabeled': (0,0,0),
                'blocking': (70,70,71),
                'terrain': (81,0,81),
                'road': (128,64,128),
                'sidewalk': (244,35,232),
                'bikelane': (98,160,234),
                'crosswalk': (229,165,10),
                'vegetation': (107,142,35)
                }


class Occupancy_Grid_SDD(Occupancy_Grid):
    
    def __init__(self, seg_map, scale, radius, grid_size_px): 
        
        super().__init__(seg_map)
        
        self.lsa_red_value = 0.9
        
        # Seg map transformation parameters
        self.scale = scale
        self.radius = radius * self.scale
        self.grid_size_px = grid_size_px
        
        return
    
    
    def transform_point(self, p):
        """ Transform a 3d point from imptc world coords into 2d map coords
        
        Args:
            p (np.array): 3d ind position, only x and y
        Returns:
            np.array: transformed u,v map pixel coordinates
        """
        
        p = p * self.scale
        return p.astype(int)
    
    
    def create_static_layer(self):
        
        # convert segmentation map to cost probabilities
        for seg_value, prob_value in self.probability_mapping.items():
            self.static_layer[self.map == seg_value] = prob_value
        
        return
    
    
    def create_object_layer(self, others):
        
        for obj in others:
            
            # Car or Truck
            if obj['class_id'] == TARGET_CLASS_DICT['car'] or obj['class_id'] == TARGET_CLASS_DICT['truck_bus'] or obj['class_id'] == TARGET_CLASS_DICT['cart']:
                
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
        
        roi,_ = self.create_grid_roi(translation=translation)
        
        # Rotate
        if apply_rot: 
            
            roi = cv2.flip(imutils.rotate(image=roi, angle=math.degrees(-rotation_angle)), 0)
            
        # Resize to target grid size
        ego_grid = cv2.resize(roi, (self.grid_size_px, self.grid_size_px), interpolation=cv2.INTER_NEAREST)
        return ego_grid


def get_sdd_cuboid(top_left, bottom_right):
    
    # Split
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    # Calculate other two corners
    top_right = np.array([x2, y1])
    bottom_left = np.array([x1, y2])
    
    return np.vstack([top_left, top_right, bottom_right, bottom_left])