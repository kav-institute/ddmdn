import os
import sys
import numpy as np
import math
import cv2 as cv
import matplotlib
matplotlib.use("Agg") 

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

sys.path.append("/workspace/repos/framework/datasets")

from common import *


def plot_loss(train_loss_list, eval_loss_list, reliability_list, plot_dir):
    
    # Create Figure
    _, axes = plt.subplots(1, 2, figsize=(16.8, 8.4))
    
    # Subplot 1
    # Train scores
    keys = list(train_loss_list[0].keys())
    num_keys = len(keys)
    cmap = plt.get_cmap('Blues_r')
    
    # Draw
    for idx, key in enumerate(keys):
        
        values = [d[key] for d in train_loss_list]
        color = cmap(idx / num_keys)
        axes[0].plot(values, c=color, marker='.', linestyle='-', linewidth=2, markersize=10, label=key)
        
    # Eval scores
    keys = list(eval_loss_list[0].keys())
    num_keys = len(keys)
    cmap = plt.get_cmap('Reds_r')
    
    # Draw
    for idx, key in enumerate(keys):
        
        values = [d[key] for d in eval_loss_list]
        color = cmap(idx / num_keys)
        axes[0].plot(values, c=color, marker='.', linestyle='-', linewidth=2, markersize=10, label=key)
        
        if key == 'mixture_per_step_NLL':
            
            best_idx = int(np.argmin(values))
            epochs = np.arange(0, len(values) + 1)
            best_epoch = epochs[best_idx]
            best_score = values[best_idx] 
            
            # Highlight the best eval score
            axes[0].scatter(best_epoch, best_score,
                s=100,
                color=color,
                marker='x',
                linewidths=2,
                label='Best Mixture NLL'
                )
            
            # Add dashed lines at the best epoch and best score
            axes[0].axhline(y=best_score, color=color, linestyle='--')
            axes[0].axvline(x=best_epoch, color=color, linestyle='--')
    
    # Figure params
    axes[0].set_ylim([-4,2])
    axes[0].tick_params(axis='both', which='major', labelsize=14)
    axes[0].set_xticks(np.arange(0,len(train_loss_list),10))
    axes[0].set_xlabel('Epochs', fontsize=16)
    axes[0].set_ylabel('Loss Scores', fontsize=16)
    axes[0].grid()
    axes[0].legend(fontsize=12)
    axes[0].set_title(f'Loss Scores over Epochs', fontsize=20)
    
    # Subplot 2
    keys = list(reliability_list[0].keys())
    num_keys = len(keys)
    cmap = plt.get_cmap('viridis')
    
    # Draw
    for idx, key in enumerate(keys):
        
        values = [float(d[key].split()[0]) for d in reliability_list]
        color = cmap(idx / num_keys)
        axes[1].plot(values, c=color, marker='.', linestyle='-', linewidth=2, markersize=10, label=key)
        
        if key == 'Eval_Score':
            
            best_idx = int(np.argmax(values))
            epochs = np.arange(0, len(values) + 1)
            best_epoch = epochs[best_idx]
            best_score = values[best_idx] 
            
            # Highlight the best eval score
            axes[1].scatter(best_epoch, best_score,
                s=100,
                color=color,
                marker='x',
                linewidths=2,
                label='Best Eval Score'
                )
            
            # Add dashed lines at the best epoch and best score
            axes[1].axhline(y=best_score, color=color, linestyle='--')
            axes[1].axvline(x=best_epoch, color=color, linestyle='--')
    
    axes[1].set_ylim([0,100])
    axes[1].tick_params(axis='both', which='major', labelsize=14)
    axes[1].set_xticks(np.arange(0,len(train_loss_list),10))
    axes[1].set_xlabel('Epochs', fontsize=16)
    axes[1].set_ylabel('Reliability Scores', fontsize=16)
    axes[1].grid()
    axes[1].legend(fontsize=12)
    axes[1].set_title(f'Reliability over Epochs', fontsize=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'train_loss.png'))
    plt.close()
    
    return


def plot_per_timestep_confidence_examples(cfg, visualizer, scene_info, agent_past_traj, agent_future_traj, agent_plot_future_traj, agent_mask, Forecast_Data, dst_dir):
    
    B = agent_past_traj.shape[0]
    
    if cfg.mode == "test": step = cfg.test.plot_instance_step
    else: step = cfg.vis.plot_instance_step
    
    for sample_idx in range(0, B, step):
        
        if 'imptc' in cfg.target: 
            id = None
            
        elif 'ind' in cfg.target: 
            id = scene_info[sample_idx]['source'].split('_')[1]
            
        elif 'sdd' in cfg.target: 
            parts = scene_info[sample_idx]['source'].split('_')
            id = '_'.join(parts[:2])
            
        else: 
            id = scene_info[sample_idx]['source'].split('_')[0]
        
        # Map, lsa, forecast
        visualizer.reset_map(id=id)
        visualizer.draw_lsa(lsa=scene_info[sample_idx]['lsa'])
        visualizer.create_ego_map(
            translation=scene_info[sample_idx]['translation'],
            rotation_angle=scene_info[sample_idx]['rotation'],
            flipped=scene_info[sample_idx]['flipped']
            )
        visualizer.create_figure()
        visualizer.draw_map()
        
        # Split agents, target agent is always the first one
        target_past_traj = agent_past_traj[:, :1, :, :].squeeze(1)
        other_past_trajs = agent_past_traj[:, 1:, :, :]
        target_future_traj = agent_future_traj[:, :1, :, :].squeeze(1)
        other_future_trajs = agent_future_traj[:, 1:, :, :]
        target_plot_future_traj = agent_plot_future_traj[:, :1, :, :].squeeze(1)
        other_plot_future_trajs = agent_plot_future_traj[:, 1:, :, :]
        
        # Target Agent
        confidence_areas, top_mode_hypos, top_mode_mask, hypos, hypos_prob = Forecast_Data.get_single_forecast(sample_idx=sample_idx)
        active_modes = top_mode_mask.sum()
        
        # Confidence areas
        for kappa_idx, _ in enumerate(cfg.train.confidence_levels):
            
            visualizer.draw_confidence_area(forecasts=confidence_areas, kappa_idx=kappa_idx) 
        
        # Draw other agents
        for agent_idx, _ in enumerate(other_past_trajs[sample_idx]):
            
            # Check if valid agent - keep the false
            if agent_mask[sample_idx, agent_idx+1]:
                continue
            
            # Trajectory data
            visualizer.draw_track_as_line(track=other_past_trajs[sample_idx, agent_idx].numpy(), color='purple', alpha=1.0, linewidth=3)
            visualizer.draw_track_as_line(track=other_plot_future_trajs[sample_idx, agent_idx].numpy(), color='black', alpha=1.0, linewidth=5)
            visualizer.draw_track_as_line(track=other_plot_future_trajs[sample_idx, agent_idx].numpy(), color='magenta', alpha=1.0, linewidth=3)
            visualizer.draw_track_as_marks(track=other_future_trajs[sample_idx, agent_idx].numpy(), color='black', alpha=1.0, marker=".", size=16)
            visualizer.draw_track_as_marks(track=other_future_trajs[sample_idx, agent_idx].numpy(), color='magenta', alpha=1.0, marker=".", size=14)
            visualizer.draw_position(pos=other_past_trajs[sample_idx, agent_idx, -1].unsqueeze(0).numpy(), color='black', alpha=1.0, marker="*", size=18)
            visualizer.draw_position(pos=other_past_trajs[sample_idx, agent_idx, -1].unsqueeze(0).numpy(), color='purple', alpha=1.0, marker="*", size=16)
        
        # Last target agent input and GT trajectory
        visualizer.draw_track_as_line(track=target_past_traj[sample_idx].numpy(), color='firebrick', alpha=1.0, linewidth=3)
        visualizer.draw_track_as_line(track=target_plot_future_traj[sample_idx].numpy(), color='black', alpha=1.0, linewidth=5)
        visualizer.draw_track_as_line(track=target_plot_future_traj[sample_idx].numpy(), color='red', alpha=1.0, linewidth=3)
        visualizer.draw_track_as_marks(track=target_future_traj[sample_idx].numpy(), color='black', marker=".", alpha=1.0, size=16)
        visualizer.draw_track_as_marks(track=target_future_traj[sample_idx].numpy(), color='red', marker=".", alpha=1.0, size=14)
        visualizer.draw_position(pos=target_past_traj[sample_idx, -1].unsqueeze(0).numpy(), color='black', alpha=1.0, marker="*", size=20)
        visualizer.draw_position(pos=target_past_traj[sample_idx, -1].unsqueeze(0).numpy(), color='firebrick', alpha=1.0, marker="*", size=18)
        
        # Save
        visualizer.save_figure(plot_name=os.path.join(dst_dir, f"{scene_info[sample_idx]['source']}.png"), active_modes=active_modes, target_info=scene_info[sample_idx], flipped=scene_info[sample_idx]['flipped'])
    
    return


def plot_per_timestep_full_examples(cfg, visualizer, scene_info, agent_past_traj, agent_future_traj, agent_plot_future_traj, agent_mask, Forecast_Data, dst_dir):
    
    B = agent_past_traj.shape[0]
    
    if cfg.mode == "test": step = cfg.test.plot_instance_step
    else: step = cfg.vis.plot_instance_step
    
    for sample_idx in range(0, B, step):
        
        if 'imptc' in cfg.target: 
            id = None
            
        elif 'ind' in cfg.target: 
            id = scene_info[sample_idx]['source'].split('_')[1]
            
        elif 'sdd' in cfg.target: 
            parts = scene_info[sample_idx]['source'].split('_')
            id = '_'.join(parts[:2])
            
        else: 
            id = scene_info[sample_idx]['source'].split('_')[0]
        
        # Map, lsa, forecast
        visualizer.reset_map(id=id)
        visualizer.draw_lsa(lsa=scene_info[sample_idx]['lsa'])
        visualizer.create_ego_map(
            translation=scene_info[sample_idx]['translation'],
            rotation_angle=scene_info[sample_idx]['rotation'],
            flipped=scene_info[sample_idx]['flipped']
            )
        visualizer.create_figure()
        visualizer.draw_map()
        
        # Split agents, target agent is always the first one
        target_past_traj = agent_past_traj[:, :1, :, :].squeeze(1)
        other_past_trajs = agent_past_traj[:, 1:, :, :]
        target_future_traj = agent_future_traj[:, :1, :, :].squeeze(1)
        other_future_trajs = agent_future_traj[:, 1:, :, :]
        target_plot_future_traj = agent_plot_future_traj[:, :1, :, :].squeeze(1)
        other_plot_future_trajs = agent_plot_future_traj[:, 1:, :, :]
        
        # Target Agent
        confidence_areas, top_mode_hypos, top_mode_mask, hypos, hypos_prob = Forecast_Data.get_single_forecast(sample_idx=sample_idx)
        
        # First all forecasts hypotheses
        if cfg.with_discrete:
            
            active_modes = top_mode_mask.sum()
            
            # All K generated discrete hypotheses
            for k, h in enumerate(hypos):
                
                visualizer.draw_track_with_marks(track=np.concatenate([np.array([[0,0]]), h], axis=0), color='black', alpha=1.0, linewidth=4, marker=".", size=12)
                visualizer.draw_track_with_marks(track=np.concatenate([np.array([[0,0]]), h], axis=0), color=cfg.hypo_colors_rgb[k], alpha=1.0, linewidth=2, marker=".", size=10)
            
        else:
            
            active_modes = cfg.num_modes
            
        # Second confidence areas
        for kappa_idx, _ in enumerate(cfg.train.confidence_levels):
            
            visualizer.draw_confidence_area(forecasts=confidence_areas, kappa_idx=kappa_idx) 
            
        # Third top p forecasts hypotheses
        if cfg.with_discrete:
            
            for t in range(cfg.num_top_hypos):
                
                h = hypos[t]
                visualizer.draw_track_with_marks(track=np.concatenate([np.array([[0,0]]), h], axis=0), color='black', alpha=1.0, linewidth=4, marker=".", size=14)
                visualizer.draw_track_with_marks(track=np.concatenate([np.array([[0,0]]), h], axis=0), color=cfg.top_hypo_colors_rgb[t], alpha=1.0, linewidth=2, marker=".", size=12)
        
        # Draw other agents
        for agent_idx, _ in enumerate(other_past_trajs[sample_idx]):
            
            # Check if valid agent - keep the false
            if agent_mask[sample_idx, agent_idx+1]:
                continue
            
            # Trajectory data
            visualizer.draw_track_as_line(track=other_past_trajs[sample_idx, agent_idx].numpy(), color='purple', alpha=1.0, linewidth=3)
            visualizer.draw_track_as_line(track=other_plot_future_trajs[sample_idx, agent_idx].numpy(), color='black', alpha=1.0, linewidth=5)
            visualizer.draw_track_as_line(track=other_plot_future_trajs[sample_idx, agent_idx].numpy(), color='magenta', alpha=1.0, linewidth=3)
            visualizer.draw_track_as_marks(track=other_future_trajs[sample_idx, agent_idx].numpy(), color='black', alpha=1.0, marker=".", size=16)
            visualizer.draw_track_as_marks(track=other_future_trajs[sample_idx, agent_idx].numpy(), color='magenta', alpha=1.0, marker=".", size=14)
            visualizer.draw_position(pos=other_past_trajs[sample_idx, agent_idx, -1].unsqueeze(0).numpy(), color='black', alpha=1.0, marker="*", size=18)
            visualizer.draw_position(pos=other_past_trajs[sample_idx, agent_idx, -1].unsqueeze(0).numpy(), color='purple', alpha=1.0, marker="*", size=16)
        
        # Last target agent input and GT trajectory
        visualizer.draw_track_as_line(track=target_past_traj[sample_idx].numpy(), color='firebrick', alpha=1.0, linewidth=3)
        visualizer.draw_track_as_line(track=target_plot_future_traj[sample_idx].numpy(), color='black', alpha=1.0, linewidth=5)
        visualizer.draw_track_as_line(track=target_plot_future_traj[sample_idx].numpy(), color='red', alpha=1.0, linewidth=3)
        visualizer.draw_track_as_marks(track=target_future_traj[sample_idx].numpy(), color='black', marker=".", alpha=1.0, size=16)
        visualizer.draw_track_as_marks(track=target_future_traj[sample_idx].numpy(), color='red', marker=".", alpha=1.0, size=14)
        visualizer.draw_position(pos=target_past_traj[sample_idx, -1].unsqueeze(0).numpy(), color='black', alpha=1.0, marker="*", size=20)
        visualizer.draw_position(pos=target_past_traj[sample_idx, -1].unsqueeze(0).numpy(), color='firebrick', alpha=1.0, marker="*", size=18)
        
        # Save
        visualizer.save_figure(plot_name=os.path.join(dst_dir, f"{scene_info[sample_idx]['source']}.png"), active_modes=active_modes, target_info=scene_info[sample_idx], flipped=scene_info[sample_idx]['flipped'])
    
    return


def plot_per_full_trajectory_examples(cfg, visualizer, scene_info, agent_past_traj, agent_future_traj, agent_plot_future_traj, agent_mask, prune_delta, Forecast_Data, dst_dir):
    
    B = agent_past_traj.shape[0]
    
    if cfg.mode == "test": step = cfg.test.plot_instance_step
    else: step = cfg.vis.plot_instance_step
    
    for sample_idx in range(0, B, step):
        
        if 'imptc' in cfg.target: 
            id = None
            
        elif 'ind' in cfg.target: 
            id = scene_info[sample_idx]['source'].split('_')[1]
            
        elif 'sdd' in cfg.target: 
            parts = scene_info[sample_idx]['source'].split('_')
            id = '_'.join(parts[:2])
            
        else: 
            id = scene_info[sample_idx]['source'].split('_')[0]
        
        # Map, lsa, forecast
        visualizer.reset_map(id=id)
        visualizer.draw_lsa(lsa=scene_info[sample_idx]['lsa'])
        visualizer.create_ego_map(
            translation=scene_info[sample_idx]['translation'],
            rotation_angle=scene_info[sample_idx]['rotation'],
            flipped=scene_info[sample_idx]['flipped']
            )
        visualizer.create_figure()
        visualizer.draw_map()
        
        # Split agents, target agent is always the first one
        target_past_traj = agent_past_traj[:, :1, :, :].squeeze(1)
        other_past_trajs = agent_past_traj[:, 1:, :, :]
        target_future_traj = agent_future_traj[:, :1, :, :].squeeze(1)
        other_future_trajs = agent_future_traj[:, 1:, :, :]
        target_plot_future_traj = agent_plot_future_traj[:, :1, :, :].squeeze(1)
        other_plot_future_trajs = agent_plot_future_traj[:, 1:, :, :]
        
        # Target Agent
        confidence_areas, top_mode_hypos, top_mode_mask, hypos, hypos_prob = Forecast_Data.get_single_forecast(sample_idx=sample_idx)
        active_modes = top_mode_mask.sum()
        
        # Confidence areas with mean trajectories
        visualizer.plot_per_mode_density(mixture=Forecast_Data.per_anchor_mixture, batch_idx=sample_idx, prune_delta=prune_delta)
        
        # Draw other agents
        for agent_idx, _ in enumerate(other_past_trajs[sample_idx]):
            
            # Check if valid agent - keep the false
            if agent_mask[sample_idx, agent_idx+1]:
                continue
            
            # Trajectory data
            visualizer.draw_track_as_line(track=other_past_trajs[sample_idx, agent_idx].numpy(), color='purple', alpha=1.0, linewidth=3)
            visualizer.draw_track_as_line(track=other_plot_future_trajs[sample_idx, agent_idx].numpy(), color='black', alpha=1.0, linewidth=5)
            visualizer.draw_track_as_line(track=other_plot_future_trajs[sample_idx, agent_idx].numpy(), color='magenta', alpha=1.0, linewidth=3)
            visualizer.draw_track_as_marks(track=other_future_trajs[sample_idx, agent_idx].numpy(), color='black', alpha=1.0, marker=".", size=16)
            visualizer.draw_track_as_marks(track=other_future_trajs[sample_idx, agent_idx].numpy(), color='magenta', alpha=1.0, marker=".", size=14)
            visualizer.draw_position(pos=other_past_trajs[sample_idx, agent_idx, -1].unsqueeze(0).numpy(), color='black', alpha=1.0, marker="*", size=18)
            visualizer.draw_position(pos=other_past_trajs[sample_idx, agent_idx, -1].unsqueeze(0).numpy(), color='purple', alpha=1.0, marker="*", size=16)
        
        # Last target agent input and GT trajectory
        visualizer.draw_track_as_line(track=target_past_traj[sample_idx].numpy(), color='firebrick', alpha=1.0, linewidth=3)
        visualizer.draw_track_as_line(track=target_plot_future_traj[sample_idx].numpy(), color='black', alpha=1.0, linewidth=5)
        visualizer.draw_track_as_line(track=target_plot_future_traj[sample_idx].numpy(), color='red', alpha=1.0, linewidth=3)
        visualizer.draw_track_as_marks(track=target_future_traj[sample_idx].numpy(), color='black', marker=".", alpha=1.0, size=16)
        visualizer.draw_track_as_marks(track=target_future_traj[sample_idx].numpy(), color='red', marker=".", alpha=1.0, size=14)
        visualizer.draw_position(pos=target_past_traj[sample_idx, -1].unsqueeze(0).numpy(), color='black', alpha=1.0, marker="*", size=20)
        visualizer.draw_position(pos=target_past_traj[sample_idx, -1].unsqueeze(0).numpy(), color='firebrick', alpha=1.0, marker="*", size=18)
        
        # Save
        visualizer.save_figure(plot_name=os.path.join(dst_dir, f"{scene_info[sample_idx]['source']}.png"), active_modes=active_modes, target_info=scene_info[sample_idx], flipped=scene_info[sample_idx]['flipped'])
    
    return


class Visualizer:
    
    def __init__(self, cfg):
        
        self.cfg = cfg
        self.map = None
        self.work_map = None
        self.w = None
        self.h = None
        
        self.fig = None
        self.canvas = None
        self.ax = None
        self.lim = None
        
        # Scale parameter
        self.scale = 0
        self.radius = 0 
        
        # Color handling
        self.forecast_colors = cfg.confidence_colors_rgb
        return
    
    
    def set_map(self, vis_map=None, id=None):
        
        raise NotImplementedError("Subclasses must implement this method")
    
    
    def reset_map(self, vis_map=None, id=None):
        
        raise NotImplementedError("Subclasses must implement this method")
    
    
    def create_figure(self):
        
        # Create figure
        self.lim = self.cfg.grid_size_meter
        plt.rcParams["figure.figsize"] = [9, 9]
        plt.rcParams["figure.autolayout"] = True
        self.fig = plt.figure()
        self.canvas = self.fig.canvas
        self.ax = self.fig.add_subplot()
        
        
    def save_figure(self, plot_name, active_modes, target_info, flipped=False):
        
        # Grab the RGB buffer from the canvas
        self.canvas.draw()
        plt.gca().set_aspect('equal')
        plt.title(f"Sample: {target_info['source']}\nState: {target_info['motion_state']} - Active Modes: {active_modes} - Input Horizon: {target_info['input_horizon']} - Flipped: {flipped}")
        plt.xlabel("x / m", fontsize = 18)
        plt.ylabel("y / m", fontsize = 18)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.xlim([-self.lim, self.lim])
        plt.ylim([-self.lim, self.lim])
        plt.savefig(plot_name)
        plt.close()
        return
        
        
    def draw_map(self):
        
        self.work_map = cv.cvtColor(self.work_map, cv.COLOR_BGR2RGB)
        plt.imshow(self.work_map, interpolation='nearest', alpha=1.0, origin='upper', extent=[-self.lim, self.lim, -self.lim, self.lim]) # [xmin, xmax, ymin, ymax]->(no flip y), [xmin, xmax, ymax, ymin]>(flip y)
        return
        
        
    def draw_lsa(self, lsa=None):
        
        return
        
        
    def transform_point_from_world(self):
        
        raise NotImplementedError("Subclasses must implement this method")
    
    
    def transform_point_from_ego(self):
        
        raise NotImplementedError("Subclasses must implement this method")
    
    
    def draw_confidence_area(self, forecasts, kappa_idx):
        
        # plot confidence areas for each forecasted horizon
        for horizon_idx in reversed(range(self.cfg.len_forecast_horizon)):
                
            contours = forecasts[horizon_idx][kappa_idx]
            color = self.forecast_colors[horizon_idx]
            
            for _, c in enumerate(contours):
            
                #c = contour[kappa_idx]
                polygon = plt.Polygon(c, facecolor=color, edgecolor=color, alpha=0.35+(kappa_idx*0.35))
                self.ax.add_patch(polygon)
        
        return
    
    
    def draw_vehicle(self, cuboid, color):
        
        return
    
    
    def draw_track_as_line(self, track, color, alpha, linewidth):
        
        P = track[~np.isnan(track).any(axis=1)]
        self.ax.plot(P[:,0], P[:,1], color=color, linewidth=linewidth, linestyle='-', alpha=alpha)
        return
    
    
    def draw_track_with_marks(self, track, color, alpha, linewidth, marker, size):
        
        P = track[~np.isnan(track).any(axis=1)]
        self.ax.plot(P[:,0], P[:,1], color=color, linewidth=linewidth, linestyle='-', alpha=alpha, marker=marker, markersize=size)
        return
    
    
    def draw_track_as_marks(self, track, color, alpha, marker, size):
        
        P = track[~np.isnan(track).any(axis=1)]
        self.ax.plot(P[:,0], P[:,1], color=color, linestyle='None', alpha=alpha, marker=marker, markersize=size)
        return
    
    
    def draw_position(self, pos, color, alpha, marker, size):
        
        self.ax.plot(pos[:,0], pos[:,1], color=color, linestyle='None', alpha=alpha, marker=marker, markersize=size)
        return
    
    
    def draw_text(self, pos, color, alpha, message, size):
        
        self.ax.text(x=pos[0], y=pos[1], s=message, color=color, alpha=alpha, ha='center', fontsize=size, fontweight='bold')
        return
    
    
    def get_roi_map_world(self, translation):
        
        # calc top-left and bottom-right coordinates to create a bbox
        pt = np.squeeze(np.array(self.transform_point_from_world(p=translation)))
        tl = np.array([pt[0] - self.radius, pt[1] - self.radius])
        br = np.array([pt[0] + self.radius, pt[1] + self.radius])
        mx = self.work_map.shape[1]
        my = self.work_map.shape[0]
        padded_map = self.work_map
        
        # left side padding
        if tl[0] < 0:
            
            # ((top, bottom),(left, right))
            p = math.ceil(abs(tl[0]))
            padded_map = np.pad(array=padded_map, pad_width=((0,0),(p,0),(0,0)), mode='constant', constant_values=0.0)
            tl[0] = 0
            br[0] = br[0] + p
            
        # top side padding
        if tl[1] < 0:
            
            # ((top, bottom),(left, right))
            p = math.ceil(abs(tl[1]))
            padded_map = np.pad(array=padded_map, pad_width=((p,0),(0,0),(0,0)), mode='constant', constant_values=0.0)
            tl[1] = 0
            br[1] = br[1] + p
            
        # right side padding
        if br[0] > mx:
            
            # ((top, bottom),(left, right))
            p = math.ceil(abs(br[0]) - mx)
            padded_map = np.pad(array=padded_map, pad_width=((0,0),(0,p),(0,0)), mode='constant', constant_values=0.0)
            
        # bottom side padding
        if br[1] > my:
            
            # ((top, bottom),(left, right))
            p = math.ceil(abs(br[1]) - my)
            padded_map = np.pad(array=padded_map, pad_width=((0,p),(0,0),(0,0)), mode='constant', constant_values=0.0)
        
        # get roi
        self.work_map = padded_map[int(tl[1]):int(br[1]), int(tl[0]):int(br[0])]
        
        return
    
    
    def rotate_map(self):
        
        raise NotImplementedError("Subclasses must implement this method")
    
    
    def plot_per_mode_density(self, mixture, batch_idx, prune_delta, n_samples=128, grid_size=128):
        """
        For each mode in the mixture (for one batch item), pool all sampled (x,y)
        over time into one point‐cloud, KDE it, and plot a filled contour density map.
        Args:
            mixture:      MixtureSameFamily with
                        - mixture_distribution.probs [B, M]
                        - component_distribution.loc        [B, M, D]
                        - component_distribution.covariance_matrix [B, M, D, D]
            batch_idx:    which sample in [0..B)
            n_samples:    number of trajectories to draw per mode
            grid_size:    resolution of the square grid for density evaluation
        Returns:
            ax: the Matplotlib Axes with the overlayed density maps & mean paths
        """
        
        # Unpack
        means = mixture.component_distribution.loc[batch_idx]
        probs = mixture.mixture_distribution.probs[batch_idx]
        M, D = means.shape
        T = D // 2
        
        # Monte Carlo sampling
        samples = mixture.component_distribution.sample((n_samples,))[:,batch_idx]
        samples = samples.cpu().numpy()
        
        # Global grid bounds
        all_xy = samples.reshape(-1, 2)
        xmin, ymin = all_xy.min(axis=0) - 0.1
        xmax, ymax = all_xy.max(axis=0) + 0.1
        xi = np.linspace(xmin, xmax, grid_size)
        yi = np.linspace(ymin, ymax, grid_size)
        X, Y = np.meshgrid(xi, yi)
        grid_pts = np.vstack([X.ravel(), Y.ravel()])
        
        # Fixed confidence level and alphas
        lf1, lf2 = [0.95, 0.68]
        a1,  a2  = [0.35, 0.70]
        low, high = sorted((lf1, lf2))
        
        for m in range(M):
            
            if probs[m] < prune_delta:
                continue
            
            # Get color
            c = self.forecast_colors[m]
            
            # For each timestep, KDE & contour
            for t in range(T):
                
                pts = samples[:, m, 2*t:2*t+2]
                kde = gaussian_kde(pts.T)
                Z = kde(grid_pts).reshape(X.shape)
                Zmax = Z.max()
                
                # Confidence level for 95% and 68%
                self.ax.contourf(X, Y, Z, levels=[low * Zmax, high * Zmax], colors=[c], alpha=a1)
                self.ax.contourf(X, Y, Z, levels=[high * Zmax, Zmax], colors=[c], alpha=a2)
                
            # Plot mean trajectory once
            mean_xy = means[m].view(T,2).cpu().numpy()
            self.ax.plot(mean_xy[:,0], mean_xy[:,1], marker='.', markersize=16, color='k', linestyle='-', lw=3)
            self.ax.plot(mean_xy[:,0], mean_xy[:,1], marker='.', markersize=14, color=c, linestyle='-', lw=2)
            
            self.draw_track_as_line(track=mean_xy, color='black', alpha=1.0, linewidth=5)
            self.draw_track_as_line(track=mean_xy, color=c, alpha=1.0, linewidth=3)
            
            self.draw_track_as_marks(track=np.concatenate([np.array([[0,0]]), mean_xy], axis=0), color='black', alpha=1.0, marker=".", size=16)
            self.draw_track_as_marks(track=np.concatenate([np.array([[0,0]]), mean_xy], axis=0), color=c, alpha=1.0, marker=".", size=14)
            
        return 