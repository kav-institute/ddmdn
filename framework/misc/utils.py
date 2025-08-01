import torch
import os
import gc
from box import Box

from termcolor import colored
from torch.distributions import Independent, Normal, Laplace, LowRankMultivariateNormal, MultivariateNormal, MixtureSameFamily, Categorical
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, ReduceLROnPlateau



class EarlyStopping:
    """
    Stops training when validation loss hasn't improved for 'patience' epochs.
    Saves the best model state.
    """
    
    def __init__(self, patience=5):
        
        self.patience = patience
        self.best_loss = float('-inf')
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False
        
        
    def __call__(self, cfg, val_loss, model, optimizer, epoch, annealing_status, dst_dir, prefix=None):
        
        # Check if the loss has improved
        if val_loss > self.best_loss:
            
            # Feedback
            if cfg.with_print: print(colored(f"Saving model: Eval Score: {val_loss:.2f} % > {self.best_loss:.2f} %", 'green'))
            cfg.logger.info(f"Saving model: Eval Score: {val_loss:.2f} % > {self.best_loss:.2f} %")
            
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            
            # Save the best model
            save_model(model=model, optimizer=optimizer, epoch=epoch, annealing_status=annealing_status, dst_dir=dst_dir, prefix=prefix)
            
        else:
            
            self.counter += 1
            if cfg.with_print: print(colored(f"Early Stopping: [{self.counter}/{self.patience}]", 'green'))
            cfg.logger.info(f"Early Stopping: [{self.counter}/{self.patience}]")
            
            if self.counter >= self.patience:
                
                self.early_stop = True
                
        return self.best_loss, self.best_epoch
                
                
def build_scheduler(cfg, optimizer):
    
    # Setup Learning Rate Scheduler
    sched_type = cfg.train.lr_scheduler.lower()
    
    # Reduce On Plateau learning rate
    if sched_type == "plateau":
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=cfg.train.plateau_factor,
            patience=cfg.train.plateau_patience,
            min_lr=cfg.train.final_learning_rate,
            verbose=True
        )
        
        return scheduler
    
    # Linear or Cosine with warm up and cool down
    else:
        
        # Warmup at lr_start
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=1.0,
            total_iters=cfg.train.warmup_epochs
        )
        
        # Cooldown at lr_end
        cooldown_scheduler = LinearLR(
            optimizer,
            start_factor=(cfg.train.final_learning_rate / cfg.train.start_learning_rate),
            end_factor=(cfg.train.final_learning_rate / cfg.train.start_learning_rate),
            total_iters=cfg.train.cooldown_epochs
        )
        
        # Linear learning rate
        if sched_type == "linear":
            
            # Linear decay
            regular_scheduler = LinearLR(
                optimizer=optimizer,
                start_factor=1.0,
                end_factor=(cfg.train.final_learning_rate / cfg.train.start_learning_rate),
                total_iters=cfg.train.num_total_epochs - cfg.train.warmup_epochs - cfg.train.cooldown_epochs
            )
            
        # Cosine annealing learning rate
        elif sched_type == "cosine":
            
            # Cosine decay
            regular_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=cfg.train.num_total_epochs - cfg.train.warmup_epochs - cfg.train.cooldown_epochs,
                eta_min=cfg.train.final_learning_rate
            )
            
        # Stitch them together
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, regular_scheduler, cooldown_scheduler],
            milestones=[cfg.train.warmup_epochs, cfg.train.num_total_epochs - cfg.train.cooldown_epochs]
        )
        
        return scheduler
    
    
def build_mesh_grid(mesh_range_x, mesh_range_y, mesh_resolution):
    """
    Generate a mesh grid over specified ranges.
    Args:
        mesh_range_x (float): Range limit in x-direction.
        mesh_range_y (float): Range limit in y-direction.
        mesh_resolution (float): Grid resolution step size.
    Returns:
        tuple: (
            grid (Tensor): Flattened grid points of shape [num_cells, 1, 2],
            x (Tensor): x-coordinate mesh of shape [steps, steps],
            y (Tensor): y-coordinate mesh of shape [steps, steps]
        )
    """
    
    # build grid
    steps = int(((mesh_range_x + mesh_range_y) / mesh_resolution))
    xs = torch.linspace(-mesh_range_x, mesh_range_x, steps=steps)
    ys = torch.linspace(-mesh_range_y, mesh_range_y, steps=steps)
    x, y = torch.meshgrid(xs, ys, indexing='xy')
    grid = torch.stack([x, y],dim=-1)
    grid = grid.reshape((-1,2))[:,None,:]
    return grid, x, y


def get_score(avg_rel, min_rel, ade, fde, mode="weighted_avg"):
    
    ade_max = 1.0
    fde_max = 1.0
    weights = (0.15,0.15,0.35,0.35)
    
    # Clamp & normalize reliability to [0,1]
    r_avg = max(0.0, min(100.0, avg_rel)) / 100.0
    r_min = max(0.0, min(100.0, min_rel)) / 100.0
    
    # Clamp ADE/FDE to [0,ade_max] & flip so bigger is better
    s_ade = 1.0 - min(max(0.0, ade), ade_max) / ade_max
    s_fde = 1.0 - min(max(0.0, fde), fde_max) / fde_max
    
    comps = [r_avg, r_min, s_ade, s_fde]
    
    if mode == "weighted_avg":
        
        w1, w2, w3, w4 = weights
        score = w1 * comps[0] + w2 * comps[1] + w3 * comps[2] + w4 * comps[3]
        
    elif mode == "geometric":
        
        eps = 1e-6
        prod = 1.0
        for c in comps:
            prod *= (c + eps)
        score = prod ** (1.0 / len(comps))
        
    elif mode == "harmonic":
        
        eps = 1e-6
        denom = 0.0
        for c in comps:
            denom += 1.0 / (c + eps)
        score = len(comps) / denom
        
    return score*100


def copy_mixture(mixtures, target_device):
    """
    Create a deep copy of a MixtureSameFamily distribution onto CPU.
    Args:
        mixtures (MixtureSameFamily): Original mixture distribution.
    Returns:
        MixtureSameFamily: New mixture distribution with cloned parameters on CPU.
    """
    
    # Identify the component distribution
    component = mixtures.component_distribution
    distribution = mixtures.mixture_distribution
    
    # Copy the mixture weights (Categorical)
    probs = distribution.probs.detach().clone().to(target_device)
    categorical = Categorical(probs=probs)
    
    # Identify the component distribution
    if isinstance(component, MultivariateNormal):
        
        loc = component.loc.detach().clone().to(target_device)
        covs = component.covariance_matrix.detach().clone().to(target_device)
        component_distribution = MultivariateNormal(loc=loc, covariance_matrix=covs)
        
    # Construct the final MixtureSameFamily
    new_mixtures = MixtureSameFamily(mixture_distribution=categorical, component_distribution=component_distribution)
    
    return new_mixtures


def get_mixture_mini_batch(mixtures, indices):
    """
    Extract a subset of a mixture distribution based on provided indices.
    Args:
        mixtures (MixtureSameFamily): Original mixture distribution.
        indices (Tensor or array-like): Indices to select mixture components.
    Returns:
        MixtureSameFamily: New mixture distribution restricted to selected indices.
    """
    
    # Extract the categorical distribution that holds the mixing weights
    new_cat = Categorical(probs=mixtures.mixture_distribution.probs[indices])
    
    # Identify the component distribution
    component = mixtures.component_distribution
    
    # MultivariateNormal
    if isinstance(component, MultivariateNormal):
        
        new_component = MultivariateNormal(component.loc[indices], component.covariance_matrix[indices])
           
    # Construct a new MixtureSameFamily distribution
    mixture_mini_batch = MixtureSameFamily(mixture_distribution=new_cat, component_distribution=new_component)
    return mixture_mini_batch


def get_mixture_components(mixture):
    """
    Given a MixtureSameFamily, return a tuple (loc, var) where
        - loc.shape = batch_shape + component_shape + event_dims
        - var.shape = same as loc, giving per-dimension variances
    
    Supports:
        - Independent(Normal)           → loc = base.loc, var = base.scale**2
        - Independent(Laplace)          → loc = base.loc, var = 2·base.scale**2
        - MultivariateNormal            → loc = comp.loc, var = diag(covariance_matrix)
        - LowRankMultivariateNormal     → loc = comp.loc, var = (cov_factor² summed + cov_diag)
    """
    
    comp = mixture.component_distribution
            
    # Full‐covariance multivariate normal
    if isinstance(comp, MultivariateNormal):
        
        # covariance_matrix shape [..., D, D], extract diag
        loc = comp.loc
        var = comp.covariance_matrix.diagonal(dim1=-2, dim2=-1)
    
    return loc, var


class AnnealingSchedules:
    """
    Stops training when validation loss hasn't improved for 'patience' epochs.
    Saves the best model state.
    """
    
    def __init__(self, cfg, train_data_size, train_batch_size):
        
        # Update annealing schedules based on train data size
        self.cfg = cfg
        self.num_batches_per_epoch = int(train_data_size / train_batch_size)
        self.num_total_steps = cfg.train.num_total_epochs * self.num_batches_per_epoch
        
        # Winner takes it all 
        self.wta_tau_steps = int(self.num_total_steps * 0.5)
        self.wta_tau_init, self.wta_tau_final = cfg.wta_tau
        
        # Mode pruning
        self.mode_pruning_end_epoch_idx = int(cfg.train.num_total_epochs * 0.5)
        self.mode_prune_tau_init, self.mode_prune_tau_final = cfg.prune_tau
        self.mode_prune_delta_init, self.mode_prune_delta_final = cfg.prune_delta
        
        self.mode_prune_delta = 0.0
        self.mode_prune_tau = 0.0
        
        # Top P (Nucleus)
        self.top_p_end_epoch_idx = int(cfg.train.num_total_epochs * 0.5)
        self.top_p_init, self.top_p_final = cfg.top_p_trunc
        
        self._update_wta(current_step=0)
        self._update_top_p(epoch=0)
        self._update_mode_prune(epoch=0)
        
        return
    
    
    def get_status(self):
        
        res = {
            "wta_tau": self.wta_tau,
            "top_p_trunc": self.top_p_trunc,
            "mode_prune_delta": self.mode_prune_delta,
            "mode_prune_tau": self.mode_prune_tau
        }
        
        return res
    
    # WTA Temperature tau anneals linearly towards zero
    def _update_wta(self, current_step):
        
        self.wta_tau = max(self.wta_tau_final, self.wta_tau_init - (self.wta_tau_init - self.wta_tau_final) * current_step / self.wta_tau_steps)
        return self.wta_tau
        
    def _get_wta(self):
        
        return self.wta_tau
    
    
    # Top P (Nucleus)
    def _update_top_p(self, epoch):
        
        self.top_p_trunc = max(self.top_p_final, self.top_p_init + (self.top_p_final - self.top_p_init) * epoch / self.top_p_end_epoch_idx)
        return self.top_p_trunc
    
    def _get_top_p(self):
        
        return self.top_p_trunc
    
    
    # Mode Pruning
    def _update_mode_prune(self, epoch):
        
        if self.cfg.with_dynamic_modes:
        
            self.mode_prune_delta = min(self.mode_prune_delta_final, self.mode_prune_delta_init + (self.mode_prune_delta_final - self.mode_prune_delta_init) * epoch / self.mode_pruning_end_epoch_idx)
            self.mode_prune_tau = max(self.mode_prune_tau_final, self.mode_prune_tau_init - (self.mode_prune_tau_init - self.mode_prune_tau_final) * epoch / self.mode_pruning_end_epoch_idx)
            return self.mode_prune_delta, self.mode_prune_tau
        
        else:
            
            return 0.0, 0.0
    
    def _get_mode_prune(self):
        
        if self.cfg.with_dynamic_modes:
        
            return self.mode_prune_delta, self.mode_prune_tau
        
        else:
            
            return 0.0, 0.0



def save_model(model, optimizer, epoch, annealing_status, dst_dir, prefix=None):
    """
    Save model and optimizer state to disk.
    Args:
        model (nn.Module): Model to save.
        optimizer (Optimizer): Optimizer to save.
        epoch (int): Current epoch number.
        dst_dir (str): Destination directory for checkpoint.
        prefix (str, optional): Optional filename prefix. Defaults to None.
    Returns:
        None
    """
    
    # create checkpoint
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'annealing_status': annealing_status
    }
    
    if prefix is not None:
        
        torch.save(obj=checkpoint, f=os.path.join(dst_dir, prefix))
        
    else:
    
        torch.save(obj=checkpoint, f=os.path.join(dst_dir, f"model_weights.pt"))
    
    return


def clear_cuda():
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return
    
    
def cfg_by_prefix(cfg, prefix):
    """
    Extract sub-configuration properties by prefix.
    Args:
        cfg (Box): Configuration containing keys with the given prefix.
        prefix (str): Prefix to filter configuration keys.
    Returns:
        Box: Sub-configuration with keys stripped of the prefix.
    """
    
    res = Box({key[len(prefix):]: value for key, value in cfg.items() if key.startswith(prefix)})
    return res