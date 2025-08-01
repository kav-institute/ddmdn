import os
import sys
import numpy as np
import math
import cv2 as cv
import copy
import imutils

sys.path.append("/workspace/repos/framework")
sys.path.append("/workspace/repos/framework/datasets")

from common import *
from full.imptc import *
from visualization import *


class IMPTC_Visualizer(Visualizer):
    
    def __init__(self, cfg):
        
        super().__init__(cfg=cfg)
        
        topview = cv.imread(os.path.join(cfg.topview_path, 'topview.png'), cv.IMREAD_COLOR)
        self.set_map(vis_map=topview)
        
        # Scale parameter
        self.scale = cfg.context_scale
        self.radius = cfg.context_radius * cfg.context_scale
        
        # Rotation angle
        self.gamma = 0.57 * math.pi
        
        # Rotation matrix
        self.R = np.matrix([[math.cos(self.gamma), -math.sin(self.gamma)],
                            [math.sin(self.gamma), math.cos(self.gamma)]])
        
        # Translation vector
        self.t = np.matrix([[760], [447]])
        
        return
    
    def set_map(self, vis_map=None, id=None):
        
        self.map = vis_map
        self.work_map = copy.deepcopy(self.map)
        self.w = self.work_map.shape[1]
        self.h = self.work_map.shape[0]
        return
    
    
    def reset_map(self, vis_map=None, id=None):
        
        self.work_map = copy.deepcopy(self.map)
        self.w = self.work_map.shape[1]
        self.h = self.work_map.shape[0]
        return
    
    
    def transform_point_from_world(self, p):
        
        # Scale and shift
        p = np.matrix([p[0], -p[1]])
        p *= self.scale
        p = self.R * p.transpose()
        p += self.t
        
        return np.array(p).reshape(-1).astype(int)
    
    
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
        self.work_map = imutils.rotate(image=self.work_map, angle=math.degrees(self.gamma) + math.degrees(rotation_angle))
        
        if flipped: 
            self.work_map = np.flip(self.work_map, axis=1)
            
        return
    
    
    def draw_lsa(self, lsa):
        
        alpha = 0.25
        overlay = self.work_map.copy()
        
        # check pedestrian crosswalks signal status
        for f in ['f1', 'f2', 'f3']:
            
            # get the indices of the pixels inside the area polygon
            area = IMPTC_MAP_LSA_POLYS.get(f)
            # red
            if lsa[f] == 10:
                
                overlay = cv.fillPoly(img=overlay, pts=np.array([area]), color=(0,0,244))
                
            # green
            elif lsa[f] == 4:
                
                overlay = cv.fillPoly(img=overlay, pts=np.array([area]), color=(0,244,0))
                
        self.work_map = cv.addWeighted(overlay, alpha, self.work_map, 1 - alpha, 0)
        
        return