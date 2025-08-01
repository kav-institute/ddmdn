import math
import cv2
import numpy as np
import imutils

from skimage.draw import polygon
from common import *


# Segmentation
IND_SEG_DICT = {'unlabeled': (0,0,0),
                'blocking': (70,70,71),
                'terrain': (81,0,81),
                'road': (128,64,128),
                'sidewalk': (244,35,232),
                'bikelane': (98,160,234),
                'crosswalk': (229,165,10),
                'vegetation': (107,142,35)
                }

class Occupancy_Grid_IND(Occupancy_Grid):
    
    def __init__(self, seg_map, scale, radius, grid_size_px, seq_id): 
        
        super().__init__(seg_map)
        
        self.lsa_red_value = 0.9
        
        # Seg map transformation parameters
        self.scale = scale
        self.radius = radius * scale
        self.grid_size_px = grid_size_px
        
        # Ortho factor
        if int(seq_id) in [0,1,2,3,4,5,6]: self.ortho = 0.0126999352667008
        else: self.ortho = 0.00814636091724916
        
        return
    
    
    def transform_point(self, p):
        """ Transform a 3d point from imptc world coords into 2d map coords
        
        Args:
            p (np.array): 3d ind position, only x and y
        Returns:
            np.array: transformed u,v map pixel coordinates
        """
        
        p = abs(p / (self.ortho * self.scale))
        return p.astype(int)
    
    
    def create_static_layer(self):
        
        # convert segmentation map to cost probabilities
        for seg_value, prob_value in self.probability_mapping.items():
            self.static_layer[self.map == seg_value] = prob_value
        
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
        
        roi,_ = self.create_grid_roi(translation=translation)
        
        # Rotate
        if apply_rot: 
            
            roi = cv2.flip(imutils.rotate(image=roi, angle=math.degrees(-rotation_angle)), 0)
            
        # Resize to target grid size
        ego_grid = cv2.resize(roi, (self.grid_size_px, self.grid_size_px), interpolation=cv2.INTER_NEAREST)
        return ego_grid


def get_ind_rotated_bbox(center, length, width, heading):
    """
    Calculate the corners of a rotated bbox from the position, shape and heading
    
    :param center: x and y coordinates of the object center position
    :param length: objects length
    :param width: object width
    :param heading: object heading (rad)
    :return: Numpy array in the shape [4 (corners), 2 (dimensions)]
    """
    
    # Precalculate all components needed for the corner calculation
    l = np.array(length) / 2
    w = np.array(width) / 2
    c = np.cos(np.deg2rad(np.array(-heading)))
    s = np.sin(np.deg2rad(np.array(-heading)))
    
    lc = l * c
    ls = l * s
    wc = w * c
    ws = w * s
    
    # Calculate all four rotated bbox corner positions assuming the object is located at the origin.
    # To do so, rotate the corners at [+/- length/2, +/- width/2] as given by the orientation.
    # Use a vectorized approach using precalculated components for maximum efficiency
    rotated_bbox_vertices = np.empty((4, 2))
    
    # Front-right corner
    rotated_bbox_vertices[0, 0] = lc - ws
    rotated_bbox_vertices[0, 1] = ls + wc
    
    # Rear-right corner
    rotated_bbox_vertices[1, 0] = -lc - ws
    rotated_bbox_vertices[1, 1] = -ls + wc
    
    # Rear-left corner
    rotated_bbox_vertices[2, 0] = -lc + ws
    rotated_bbox_vertices[2, 1] = -ls - wc
    
    # Front-left corner
    rotated_bbox_vertices[3, 0] = lc + ws
    rotated_bbox_vertices[3, 1] = ls - wc
    
    # Move corners of rotated bounding box from the origin to the object's location
    rotated_bbox_vertices = rotated_bbox_vertices + center
    return rotated_bbox_vertices