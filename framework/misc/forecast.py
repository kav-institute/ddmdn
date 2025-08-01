import os
import torch
import copy
import numpy as np
import json
import matplotlib.pyplot as plt

from termcolor import colored
from skimage import measure
from misc.utils import get_mixture_mini_batch, copy_mixture


class Forecasts:
    """ Class loading parameters and and paths from external config file
    """
    
    def __init__(self, cfg, mini_B, device):
        
        self.cfg = cfg
        self.dataset = cfg.dataset
        self.per_step_mixture = []
        self.hypos = []
        self.hypos_prob = []
        self.top_mode_hypos = []
        self.top_mode_hypos_mask = []
        self.agent_future_traj = []
        self.agent_past_traj = []
        self.agent_mask = []
        self.device = device
        
        self.sharpness_sets = []
        self.gt_conf_sets = []
        self.mesh_conf_maps = []
        self.mesh_means = []
        self.mesh_modes = []
        self.confidence_levels = []
        self.scene_info = []
        
        self.full_grid_cell_size = cfg.full_grid_cell_size
        self.full_grid_resolution = cfg.full_grid_resolution
        self.grid_size_meter = cfg.grid_size_meter
        
        self.mini_B = mini_B
        self.T = self.cfg.len_forecast_horizon
        self.M = self.cfg.num_modes
        self.G = self.cfg.grid.full_cell_size * self.cfg.grid.full_cell_size
        
        # Load SDD scales to convert from meters to pixels for evaluation
        if self.dataset == 'sdd_benchmark':
            
            file_path = os.path.join(cfg.data_dir[:-len('\database')], 'homography', 'H_SDD.txt')
            
            self.sdd_scales_dict = {}
            with open(file_path, 'r') as hf:
                for row in hf.readlines():
                    item = row.strip().split('\t')
                    if not item: continue
                    if not "jpg" in item[0]: continue
                    if not "A" in item[3]: continue
                    scene = item[0][:-4]
                    scale = float(item[-1])
                    self.sdd_scales_dict[scene] = scale
                    
        elif self.dataset == "sdd_full":
            
            file_path = os.path.join(cfg.data_dir[:-len('\database')], 'homography', 'scales.json')
            
            with open(file_path, 'r', encoding='utf-8') as f:
                self.sdd_scales_dict = json.load(f)
        
        return
    
    
    def add_batch_data(self, batch_data):
        
        self.hypos.append(batch_data['hypos'])
        self.hypos_prob.append(batch_data['hypos_prob'])
        self.top_mode_hypos.append(batch_data["top_mode_hypos"])
        self.top_mode_hypos_mask.append(batch_data["top_mode_hypos_mask"])
        self.per_step_mixture.append(batch_data['per_step_mixture'])
        self.agent_future_traj.append(batch_data['agent_future_traj'])
        self.agent_past_traj.append(batch_data['agent_past_traj'])
        self.agent_mask.append(batch_data['agent_mask'])
        self.scene_info.append(batch_data['scene_info'])
        self.gt_conf_sets.append(batch_data['gt_conf_set'])
        self.mesh_conf_maps.append(batch_data['mesh_conf_map'])
        self.mesh_means.append(batch_data['mesh_means'])
        self.mesh_modes.append(batch_data['mesh_modes'])
        self.confidence_levels.append(batch_data["confidence_levels"])
        
        return
    
    
    def compute_reliability(self, epoch=0, type='Train'):
        
        if self.cfg.mode == "test": print(colored(f"Compute Reliability", 'green'))
        confidence_sets = np.concatenate(self.gt_conf_sets).reshape(-1, self.cfg.len_forecast_horizon)
        
        res_dict = {}
        bins = self.cfg.reliability_bins
        with_plot = self.cfg.plot_reliability
        plot_step = self.cfg.plot_reliability_step
        if type=='Train': plot_dir = self.cfg.train_reliability_path
        elif type=='Eval': plot_dir = self.cfg.eval_reliability_path
        elif type=='Test': plot_dir = self.cfg.test_reliability_path
        
        if with_plot and (epoch % plot_step == 0):
            
            plt.rcParams["figure.figsize"] = [9, 9]
            plt.rcParams["figure.autolayout"] = True
            plt.figure()
            plt.plot(bins, bins, 'k--', linewidth=4, label=f"ideal")
            
        # place/sort values into bins
        # attention!: digitize() returns indexes, with first index starting at 1 not 0
        bin_data = np.digitize(confidence_sets, bins=bins)
        reliability_errors = []
        
        for idx in range(1, bin_data.shape[1]):
            
            # build calibration curve
            # attention!: bincount() returns amount of each bin, first bin to count is bin at 0,
            # due to digitize behavior must increment len(bins) by 1 and later ignore the zero bin count
            f0 = np.array(np.bincount(bin_data[:,idx], minlength=len(bins)+1)).T
            
            # f0[1:]: because of the different start values of digitize and bincount, we remove/ignore the first value of f0
            acc_f0 = np.cumsum(f0[1:],axis=0)/confidence_sets.shape[0]
            
            # get differences for current step
            r = abs(acc_f0 - bins)
            reliability_errors.append(r)
            
            if with_plot and (epoch % plot_step == 0): 
                color = self.cfg.confidence_colors_rgb[idx]
                plt.plot(bins, acc_f0, color=color, linewidth=4, label=f"{round((self.cfg.output_horizons[idx]+1)*self.cfg.common.input_delta_t, 1)} sec @ avg: {(1 - np.mean(r))*100:.1f} %, min: {(1 - np.max(r))*100:.1f} %")
            
        # Get reliability scores, skip the reliability error for the first future timestep
        reliability_avg_score = (1 - np.mean(reliability_errors)) * 100
        reliability_min_score = (1- np.max(reliability_errors)) * 100
        res_dict[f"{type}_avg_RLS"] = f"{reliability_avg_score:.2f} %"
        res_dict[f"{type}_min_RLS"] = f"{reliability_min_score:.2f} %"
        
        if with_plot and (epoch % plot_step == 0):
            
            plt.minorticks_on()
            plt.grid(which='major', linestyle='-', linewidth=1.0, color='dimgray')
            plt.grid(which='minor', linestyle='--', linewidth=0.75, color='silver')
            plt.xticks(fontsize = 20)
            plt.yticks(fontsize = 20)
            plt.title(f'Reliability Calibration: - Avg: {reliability_avg_score:.1f} % - Min: {reliability_min_score:.1f} %', fontsize=20)
            plt.savefig(os.path.join(plot_dir, f'reliability_plain_{str(epoch).zfill(4)}.png'))
            plt.close()
        
        return res_dict
    
    
    def build_sharpness_sets(self):
        
        confidence_maps = np.stack([f for f in self.mesh_conf_maps])
        ss = []
        
        for kappa in self.cfg.kappas:
            
            ss.append((self._estimate_sharpness(conf_map=confidence_maps, kappa=kappa) * (self.cfg.grid_size_meter * self.cfg.grid_size_meter)).reshape(-1, self.cfg.len_forecast_horizon))
        
        self.sharpness_sets = np.stack(ss, 1)
        return
    
    
    def compute_sharpness(self):
        
        if self.cfg.mode == "test": print(colored(f"Compute Sharpness", 'green'))
        self.build_sharpness_sets()
        sharpness_scores = {}
        percentiles = self.cfg.reliability_bins
        
        for idk, cl in enumerate(self.cfg.kappas):
            
            s = self.sharpness_sets[:,idk,:].T
            SDist=np.zeros((self.cfg.len_forecast_horizon, len(percentiles)))
            
            for idp, p in enumerate(percentiles):
                
                for t in range(self.cfg.len_forecast_horizon):
                    
                    SDist[t,idp]=np.percentile(s[t,:], p*100, axis=-1)
            
            # calc mean sharpness score (i.e 50%) for current confidence level
            sharpness_score = sum([np.mean(SDist[ids,:] / ((step+1)*self.cfg.common.output_delta_t)) for ids, step in enumerate(range(self.cfg.len_forecast_horizon))]) * (1/(self.cfg.len_forecast_horizon*self.cfg.common.output_delta_t))
            sharpness_scores[f"SS@{cl*100}%"] = f"{sharpness_score:.2f} m/s²"
        
        return sharpness_scores
     
    
    def compute_min_displacement_bok(self):
        """
        Best‑of‑K ADE & FDE between initial hypotheses and ground truth.
        Args:
            hypotheses: List of Tensors of shape [B, K, T, 2]
            gt:         List of Tensors of shape [B, T, 2]
        Returns:
            min_ade: scalar min_k
            min_fde: scalar min_k
        """
        
        if self.cfg.mode == "test": print(colored(f"Compute Best-of-K Displacements", 'green'))
        if self.cfg.with_discrete:
            
            # Results must be in pixels
            if self.dataset == "sdd_full" or self.dataset == "sdd_benchmark":
                
                res_dict = {}
                min_ade_m = []
                min_fde_m = []
                min_ade_px = []
                min_fde_px = []
                K = self.cfg.num_hypos
                
                # Loop over batches
                for idx, _ in enumerate(self.hypos):
                    
                    hypos_b = self.hypos[idx]               # [B, K, T, 2]
                    gt_b    = self.agent_future_traj[idx]   # [B, T, 2]
                    
                    # Align gt to [B, K, T, 2]
                    gt_exp = gt_b.unsqueeze(1).expand_as(hypos_b)
                    
                    # L2 distances in meters: [B, K, T]
                    dists_m = torch.norm(hypos_b - gt_exp, dim=-1)
                    
                    # Collect scales [B,1,1]
                    scales = torch.tensor(
                        [self.sdd_scales_dict["_".join(info['source'].split('_')[:2])] for info in self.scene_info[idx]],
                        device=dists_m.device
                    ).view(-1, 1, 1)
                    
                    # Convert to pixels
                    dists_px = dists_m / scales
                    
                    # meters metrics
                    ade_k_m = dists_m.mean(dim=-1)  # [B, K]
                    fde_k_m = dists_m[..., -1]      # [B, K]
                    min_ade_m.append(ade_k_m.min(dim=-1).values.mean())
                    min_fde_m.append(fde_k_m.min(dim=-1).values.mean())
                    
                    #pixel metrics
                    ade_k_px = dists_px.mean(dim=-1)
                    fde_k_px = dists_px[..., -1]
                    min_ade_px.append(ade_k_px.min(dim=-1).values.mean())
                    min_fde_px.append(fde_k_px.min(dim=-1).values.mean())
                    
                # Aggregate and format
                res_dict[f"min_ADE@(K={K})"]    = f"{np.stack(min_ade_m).mean():.2f} m"
                res_dict[f"min_FDE@(K={K})"]    = f"{np.stack(min_fde_m).mean():.2f} m"
                res_dict[f"min_ADE@(K={K}_)"]   = f"{np.stack(min_ade_px).mean():.2f} px"
                res_dict[f"min_FDE@(K={K}_)"]   = f"{np.stack(min_fde_px).mean():.2f} px"
                
                return res_dict
            
            # Results in meters
            else:
                
                res_dict = {}
                min_ade_list = []
                min_fde_list = []
                K = self.cfg.num_hypos
                
                # Loop over batches
                for idx, _ in enumerate(self.hypos):
                    
                    # Align gt
                    gt_exp = self.agent_future_traj[idx].unsqueeze(1).expand(-1, self.hypos[idx].size(1), -1, -1)  # [B,K,T,2]
                    
                    # Compute L2 per-hypothesis, per-step
                    dists = torch.norm(self.hypos[idx] - gt_exp, dim=-1)  # [B,K,T]
                    
                    # ADE per hypothesis and mean over T
                    ade_k = dists.mean(dim=-1)  # [B,K]
                    
                    # Best ADE per (B), then mean
                    min_ade_list.append(ade_k.min(dim=-1).values.mean())
                    
                    # FDE per hypothesis
                    fde_k = dists[..., -1]  # [B,K]
                    
                    # Best FDE per (b,a), then mean
                    min_fde_list.append(fde_k.min(dim=-1).values.mean())
                
                res_dict[f"min_ADE@(K={K})"] = f"{np.stack(min_ade_list).sum() / len(min_ade_list):.2f} m"
                res_dict[f"min_FDE@(K={K})"] = f"{np.stack(min_fde_list).sum() / len(min_fde_list):.2f} m"
                return res_dict
        
        else:
            return {}
        
        
    def compute_min_displacement_top(self):
        """
        Best-of-P ADE & FDE among the P hypotheses with highest probability.
        Uses:
            self.hypotheses_pos  # List of Tensors of shape [B, K, T, 2]
            self.hypotheses_prob # List of Tensors of shape [B, K]
            self.agent_future_traj # List of Tensors of shape [B, T, 2]
        Returns:
            dict with
                'min_ADE_top': float
                'min_FDE_top': float
        """
        
        if self.cfg.mode == "test": print(colored(f"Compute Top-P Displacements", 'green'))
        if self.cfg.with_discrete:
            
            P = self.cfg.num_top_hypos
            min_ade_list = []
            min_fde_list = []
            avg_top_prob_list = []
            hit_count = 0
            total_count = 0
            
            for hypotheses, probs, gt in zip(self.hypos, self.hypos_prob, self.agent_future_traj):
                
                B, _, _, _ = hypotheses.shape
                
                # Avg top-P probabilities per batch element, then across B
                top_probs = probs[:, :P]  # [B, P]
                avg_top_prob_list.append(top_probs.mean(dim=0))  # [P]
                
                # Best-of-P ADE & FDE
                top_pos = hypotheses[:, :P]  # [B, P, T, 2]
                d_top = torch.norm(top_pos - gt.unsqueeze(1), dim=-1)  # [B, P, T]
                ade_p = d_top.mean(dim=-1)  # [B, P]
                fde_p = d_top[..., -1]  # [B, P]
                min_ade_list.append(ade_p.min(dim=-1).values.mean().item())
                min_fde_list.append(fde_p.min(dim=-1).values.mean().item())
                
                # How often does best-of-K lie within top-P
                d_all = torch.norm(hypotheses - gt.unsqueeze(1), dim=-1)  # [B, K, T]
                ade_all = d_all.mean(dim=-1)  # [B, K]
                best_idx = ade_all.argmin(dim=-1)  # [B]
                hit_count += (best_idx < P).sum().item()
                total_count += B
                
            res_dict = {
                f"min_ADE@(top {P})": f"{np.mean(min_ade_list):.2f} m",
                f"min_FDE@(top {P})": f"{np.mean(min_fde_list):.2f} m",
                f"Hitrate@(top {P})": f"{((hit_count/total_count)*100):.2f} %",
                f"Avg_Probs@(top {P})": [round(x, 3) for x in torch.stack(avg_top_prob_list).mean(dim=0).tolist()],
            }
            
            return res_dict
        
        else:
            
            return {}
                  
        
    def _estimate_sharpness(self, conf_map, kappa):
        
        area = np.where(conf_map <= kappa, 1.0, 0.0)
        area = area.mean(axis=3)
        return area
    
    
class Forecast_Batch:
    """ Class loading parameters and and paths from external config file
    """
    
    def __init__(self, cfg, mini_B, per_step_mixture, per_anchor_mixture, per_step_mixture_flat, hypos, hypos_prob, top_mode_hypos, top_mode_hypos_mask, agent_past_traj, agent_future_traj, mesh_grid, 
                agent_mask, scene_info, device):
        
        self.cfg = cfg
        self.per_step_mixture = per_step_mixture
        self.per_step_mixture_flat = per_step_mixture_flat
        self.per_anchor_mixture = per_anchor_mixture
        self.hypos = hypos
        self.hypos_prob = hypos_prob
        self.top_mode_hypos = top_mode_hypos
        self.top_mode_hypos_mask = top_mode_hypos_mask
        self.agent_future_traj = agent_future_traj[:, 0 ,: ,:]
        self.agent_past_traj = agent_past_traj[:, 0 ,: ,:]
        self.agent_mask = agent_mask
        self.mesh_grid = mesh_grid
        self.scene_info = scene_info
        self.device = device
        
        # Internal data
        self.gt_conf_set = None
        self.mesh_conf_map = None
        self.mesh_means = None
        self.mesh_modes = None
        self.confidence_levels_data = None
        
        self.full_grid_cell_size = cfg.full_grid_cell_size
        self.full_grid_resolution = cfg.full_grid_resolution
        self.grid_size_meter = cfg.grid_size_meter
        
        self.T = cfg.len_forecast_horizon
        self.B, _ = self.per_step_mixture.batch_shape
        self.mini_B = mini_B
        self.M = self.cfg.num_modes
        self.G = self.full_grid_cell_size * self.full_grid_cell_size 
        self.S = self.B * self.T
        
        return
    
    
    def build_gt_confidence_set(self):
        
        # Only for target agent
        # Log‐prob of the true trajectories:
        gt_log_prob = self.per_step_mixture.log_prob(self.agent_future_traj)
        
        # Draw samples from the full mixture and get their log_probs
        samples = self.per_step_mixture.sample(sample_shape=torch.Size([self.cfg.train.num_samples]))
        samples_log_prob = self.per_step_mixture.log_prob(samples)
        
        # For each batch element, count how many samples exceed the GT log‐prob
        idx_mask = (samples_log_prob.unsqueeze(1) > gt_log_prob).float()
        
        # Confidence set per element: fraction of samples more likely than GT
        gt_conf_set = (torch.sum(idx_mask, 0) / samples.shape[0]).squeeze(0)
        self.gt_conf_set = gt_conf_set.detach().cpu()
        
        return
    
    
    def build_mesh_confidence_set(self):
        
        mesh_conf_map = torch.zeros((self.G, self.S), device=self.device)
        self.mesh_means = torch.zeros((self.S, 2), device=self.device)
        
        # Process as mini batches/chunks to avoid huge intermediate tensors
        for idx in range(0, self.S, self.mini_B):
        
            mini_mixtures = get_mixture_mini_batch(mixtures=self.per_step_mixture_flat, indices=range(idx, idx+self.mini_B))
            gt_log_prob = mini_mixtures.log_prob(self.mesh_grid)
            samples = mini_mixtures.sample(sample_shape=torch.Size([self.cfg.train.num_samples]))
            samples_log_prob = mini_mixtures.log_prob(samples)
            idx_mask = (samples_log_prob.unsqueeze(1) > gt_log_prob).float()
            mesh_conf_map[..., idx:idx+self.mini_B] = torch.sum(idx_mask, 0)/samples[:, idx:idx+self.mini_B, :].shape[0]
            self.mesh_means[idx:idx+self.mini_B] = samples.mean(dim=0)
            
        m = self.mesh_grid[torch.argmin(mesh_conf_map, dim=0, keepdim=True)]
        self.mesh_modes = m[0,:,0,:].view(self.B, self.T, -1)
        self.mesh_means = self.mesh_means.view(self.B, self.T, -1)
        self.mesh_conf_map = mesh_conf_map.detach().cpu().t().view(self.B, self.T, -1)
        
        return 
    
    
    def build_confidence_levels(self):
        
        self.confidence_levels_data = [[[None for v in self.cfg.kappas] for _ in range (self.T)] for _ in range(self.B)]
        
        for b in range(self.B):
                
                for t in range (self.T):
                    
                    for k, kappa in enumerate(self.cfg.kappas):
                        
                        cmap = self.mesh_conf_map[b,t]
                        mode = self.mesh_modes[b,t].detach().cpu().numpy()
                        mean = self.mesh_means[b,t].detach().cpu().numpy()
                        self.confidence_levels_data[b][t][k] = self.compute_confidence_level(conf_map=cmap, mode=mode, mean=mean, kappa=kappa)
            
        return
    
    
    def compute_confidence_level(self, conf_map, mode, mean, kappa):
            
        conf_area = np.where(conf_map.numpy() <= kappa, 1, 0).reshape(self.full_grid_cell_size, self.full_grid_cell_size)
            
        # Get contour(s)
        contour = measure.find_contours(conf_area, 0.5)
        
        # Uni modal dist
        if len(contour) == 1:
            
            cont = [np.flip(m=np.squeeze(np.array(contour, dtype=np.float32) * self.full_grid_resolution - self.grid_size_meter))]
            
        # Multi modal dist
        elif len(contour) >= 2:
            
            cont = [np.array(contour[i], dtype=np.float32) * self.full_grid_resolution - self.grid_size_meter for i in range(0, len(contour))]
            cont = [np.flip(m=ct) for ct in cont]
            
        # Dist area smaller or equal to single point or grid size resolution, i.e mode of this dist
        else:
            
            cont = [np.array(mode, dtype=np.float32)[None, ...]]
            cont = [np.squeeze(a=cont, axis=0)]
            
        return cont
    
    
    def get_single_forecast(self, sample_idx):
        
        confidence_areas = [self.confidence_levels_data[sample_idx][t] for t in range(self.T)]
        
        if self.cfg.with_discrete:
            
            hypos = self.hypos[sample_idx].detach().clone().to('cpu').numpy()
            hypos_prob = self.hypos_prob[sample_idx].detach().clone().to('cpu').numpy()
            top_mode_hypos = self.top_mode_hypos[sample_idx].detach().clone().to('cpu').numpy()
            top_mode_mask = self.top_mode_hypos_mask[sample_idx].detach().clone().to('cpu').numpy()
            
        else:
            
            hypos = None
            hypos_prob = None
            top_mode_hypos = None
            top_mode_mask = None
        
        return confidence_areas, top_mode_hypos, top_mode_mask, hypos, hypos_prob
    
    
    def get_data(self):
        
        # Get a copy of the mixtures from gpu to cpu
        per_step_mixture = copy_mixture(mixtures=self.per_step_mixture, target_device='cpu')
        
        # Clone/Copy all data to fully detach from all gpu related objects/connections
        data = {
            "hypos": self.hypos.detach().clone().to('cpu') if self.cfg.with_discrete else None,
            "hypos_prob": self.hypos_prob.detach().clone().to('cpu') if self.cfg.with_discrete else None,
            "top_mode_hypos": self.top_mode_hypos.detach().clone().to('cpu') if self.cfg.with_discrete else None,
            "top_mode_hypos_mask": self.top_mode_hypos_mask.detach().clone().to('cpu') if self.cfg.with_discrete else None,
            "per_step_mixture": per_step_mixture,
            "agent_past_traj": self.agent_past_traj.detach().clone(),
            "agent_future_traj": self.agent_future_traj.detach().clone().to('cpu'),
            "agent_mask": self.agent_mask.detach().clone().to('cpu'),
            "gt_conf_set": self.gt_conf_set.detach().clone() if self.gt_conf_set is not None else [],
            "mesh_conf_map": self.mesh_conf_map.detach().clone() if self.mesh_conf_map is not None else [],
            "mesh_means": self.mesh_means.detach().clone().to('cpu') if self.mesh_means is not None else [],
            "mesh_modes": self.mesh_modes.detach().clone().to('cpu') if self.mesh_modes is not None else [],
            "confidence_levels": copy.deepcopy(self.confidence_levels_data),
            "scene_info": copy.deepcopy(self.scene_info)
        }
        
        return data