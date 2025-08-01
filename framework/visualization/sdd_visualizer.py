import os
import sys
import numpy as np
import math
import cv2 as cv
import copy
import imutils
import json

sys.path.append("/workspace/repos/framework")
sys.path.append("/workspace/repos/framework/datasets")

from common import *
from full.sdd import *
from visualization import *


class SDD_Visualizer(Visualizer):
    
    def __init__(self, cfg):
        
        super().__init__(cfg=cfg)
        
        self.maps = {}
        self.scales = json.load(open(os.path.join(cfg.topview_path[:-7], 'homography', 'scales.json')))
        
        for p in sorted(os.listdir(cfg.topview_path)):
            
            id = p[:-12]
            self.maps[id] = cv.imread(os.path.join(cfg.topview_path, p), cv.IMREAD_COLOR)
        
        # Scale parameter
        self.context_radius = cfg.context_radius
        self.scale = None
        self.radius = None
        self.work_map = None
        
        return
    
    
    def set_map(self, vis_map=None, id=None):
        
        self.scale = 1 / self.scales[id]
        self.radius = self.scale * self.context_radius
        self.map = self.maps[id]
        self.work_map = copy.deepcopy(self.map)
        self.w = self.work_map.shape[1]
        self.h = self.work_map.shape[0]
        return
    
    
    def reset_map(self, vis_map=None, id=None):
        
        self.scale = 1 / self.scales[id]
        self.radius = self.scale * self.context_radius
        self.map = self.maps[id]
        self.work_map = copy.deepcopy(self.map)
        self.w = self.work_map.shape[1]
        self.h = self.work_map.shape[0]
        return
    
    
    def transform_point_from_world(self, p):
        
        p = p * self.scale
        return p.astype(int)
    
    
    def transform_point_from_ego(self, p):
        
        # Scale and shift
        p = np.matrix([p[0], -p[1]])
        p *= self.scale
        p += self.radius
        
        return np.array(p).reshape(-1).astype(int)
    
    
    def create_ego_map(self, rotation_angle, translation, flipped=False):
        
        self.get_roi_map_world(translation=translation)
        self._rotate_map(rotation_angle=rotation_angle, flipped=flipped)
        return
    
    
    def _rotate_map(self, rotation_angle, flipped=False):
        
        # rotate for ego coords and resize to target grid size
        self.work_map = cv.flip(imutils.rotate(image=self.work_map, angle=math.degrees(-rotation_angle)), 0)
        
        if flipped: 
            self.work_map = np.flip(self.work_map, axis=1)
            
        return