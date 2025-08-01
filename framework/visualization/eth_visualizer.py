import os
import sys
import torch
import numpy as np
import cv2 as cv
import copy

sys.path.append("/workspace/repos/framework")
sys.path.append("/workspace/repos/framework/datasets")

from common import *
from transformation import *
from visualization import *

ETH_SCENES = ['eth', 'hotel', 'univ', 'zara01', 'zara02', 'zara03', 'students001', 'students003']


def filter_substrings(list_a, list_b, case_sensitive=True):
    
    if not case_sensitive:
        
        # Lowercase all strings for comparison
        list_a = [a.lower() for a in list_a]
        list_b = [b.lower() for b in list_b]
        
    return [a for a in list_a if any(a in b for b in list_b)]


class ETH_Visualizer(Visualizer):
    
    def __init__(self, cfg):
        
        super().__init__(cfg=cfg)
        
        self.maps = {}
        self.homographys = {}
        
        source_names = filter_substrings(list_a=ETH_SCENES, list_b=os.listdir(cfg.topview_path))
        
        for id in sorted(source_names):
            
            self.maps[id] = cv.imread(os.path.join(cfg.topview_path, f"{id}_topview.png"), cv.IMREAD_COLOR)
            self.homographys[id] = np.loadtxt(os.path.join(cfg.topview_path[:-7], 'homography', f"{id}_H.txt"))
        
        # Scale parameter
        self.scale = cfg.context_scale
        self.radius = cfg.context_radius * cfg.context_scale
        self.map = None
        self.H = None
        
        return
    
    def set_map(self, vis_map=None, id=None):
        
        self.H = self.homographys[id]
        self.map = self.maps[id]
        self.work_map = copy.deepcopy(self.map)
        self.w = self.work_map.shape[1]
        self.h = self.work_map.shape[0]
        return
    
    
    def reset_map(self, vis_map=None, id=None):
        
        self.H = self.homographys[id]
        self.map = self.maps[id]
        self.work_map = copy.deepcopy(self.map)
        self.w = self.work_map.shape[1]
        self.h = self.work_map.shape[0]
        return
    
    
    def transform_point_from_world(self, p):
        
        pt = self._world2image(coord=p, H=self.H)
        return pt.astype(int)
    
    
    def transform_point_from_ego(self, p):
        
        # Scale and shift
        p = np.matrix([p[0], -p[1]])
        p *= self.scale
        p += self.radius
        
        return np.array(p).reshape(-1).astype(int)
    
    
    def create_ego_map(self, rotation_angle, translation, flipped=None):
        """
        Returns
        -------
        roi_img_raw : perspective‑warp in camera native orientation
        roi_img_ego : 180°‑rotated copy that speaks ego axes (+X→right, +Y→up)
        fig         : matplotlib Figure with two sub‑plots (scene | ego patch)
        """
        # Ego Transformer
        ego_tf = Ego_Transformer(translation, rotation_angle)
        
        # Define corners of the desired square in ego coords
        ego_sq = np.array([
            [-self.radius, -self.radius],
            [ self.radius, -self.radius],
            [ self.radius,  self.radius],
            [-self.radius,  self.radius]
        ], np.float32)
        
        # Ego → World → Image (four corners)
        world_sq = ego_tf.revert(ego_sq)  # ego → world
        img_sq = self._world2image(world_sq, self.H).astype(np.float32)  # world → image
        
        # Average edge‑length in px, side length
        edge_px = np.array([
            np.linalg.norm(img_sq[1] - img_sq[0]),
            np.linalg.norm(img_sq[2] - img_sq[3]),
            np.linalg.norm(img_sq[3] - img_sq[0]),
            np.linalg.norm(img_sq[2] - img_sq[1]),
        ])
        
        side_px = max(1, int(round(edge_px.mean())))
        
        dst_sq = np.array([
            [0, 0],
            [side_px-1, 0],
            [side_px-1, side_px-1],
            [0, side_px-1]
        ], np.float32)
        
        # Perspective warp
        M = cv.getPerspectiveTransform(img_sq, dst_sq)
        self.work_map = cv.warpPerspective(
            self.work_map, M, (side_px, side_px),
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=1.0,
        )
        
        # Rotate at x-axis once so raster aligns with ego axes
        self.work_map  = cv.flip(self.work_map , 0)
        
        if flipped: 
            self.work_map = np.flip(self.work_map, axis=1)
        
        return
    
    
    def _world2image(self, coord, H):
        """Convert world coordinates to image coordinates.
        
        Args:
            coord (np.ndarray or torch.tensor): World coordinates, shape (..., 2).
            H (np.ndarray or torch.tensor): Homography matrix, shape (3, 3).
        
        Returns:
            np.ndarray: Image coordinates.
        """
        
        assert coord.shape[-1] == 2
        assert H.shape == (3, 3)
        assert type(coord) == type(H)
        
        shape = coord.shape
        coord = coord.reshape(-1, 2)
        
        if isinstance(coord, np.ndarray):
            x, y = coord[..., 0], coord[..., 1]
            image = (np.linalg.inv(H) @ np.stack([x, y, np.ones_like(x)], axis=-1).T).T
            image = image / image[..., [2]]
            image = image[..., :2]
        
        elif isinstance(coord, torch.Tensor):
            x, y = coord[..., 0], coord[..., 1]
            image = (torch.linalg.inv(H) @ torch.stack([x, y, torch.ones_like(x)], dim=-1).T).T
            image = image / image[..., [2]]
            image = image[..., :2]
            
        else:
            raise NotImplementedError
        
        return image.reshape(shape)