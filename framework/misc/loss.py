import torch
import torch.nn.functional as F


def compute_per_traj_nll(mixture, gt):
    """
    Compute the negative log-likelihood of the ground truth under a mixture distribution.
    Evaluates the log-probability of each ground-truth trajectory, and returns the mean NLL loss, encouraging high model confidence.
    Args:
        mixture (MixtureSameFamily): Mixture distribution for predicted trajectories.
        gt (Tensor): Ground truth trajectory tensor of shape [B, T, 2].
    Returns:
        Tensor: Scalar mean negative log-likelihood loss.
    """
    
    B, T, D = gt.shape
    
    # flatten each trajectory to shape [B, 2T]
    gt_flat = gt.reshape(B, T * D)
    
    # mixture.log_prob now returns [B]
    log_prob = mixture.log_prob(gt_flat)
    
    # mean over batch
    per_traj_nll_score = -log_prob.mean()
    return per_traj_nll_score


def compute_per_step_nll(mixture, gt):
    """
    mixture:   MixtureSameFamily over batch_shape=[B,T], event_shape=[2]
    gt:        [B, T, 2]
    returns:   scalar mean NLL over all B×T
    """
    logp = mixture.log_prob(gt)         # [B, T]
    per_step_nll_score = -logp.mean()
    return per_step_nll_score


def compute_wta(cfg, hypos, gt, wta_tau):
    """
    Winner-takes-all loss that softly selects the best hypothesis via annealed softmax weights.
    Computes L2 distances between each hypothesis and ground truth,
    applies a temperature-annealed softmax to emphasize the closest trajectory,
    and returns the weighted mean-squared error, approaching hard WTA as tau→0.
    Args:
        cfg (Config): Configuration containing wta_tau schedule (init, final, steps).
        hypos (Tensor): [B,K,T,2] K trajectory hypotheses per batch.
        gt (Tensor)    : [B,T,2] Ground truth trajectories.
        wta_tau (int): Current linear tau annealing value.
    Returns:
        Scalar: Weighted MSE loss under soft WTA weighting.
    """
    
    B, K, _, _ = hypos.shape
    
    # L2 distance in trajectory space
    dists = ((hypos - gt.unsqueeze(1))**2).mean(dim=(2,3))  # [B,K]
    
    # Soft weights w_k to hard WTA as tau comes toward zero
    w = torch.softmax(-dists / wta_tau, dim=-1).detach()  # [B,K]
    
    # Weighted MSE over K (keeps every head alive early)
    w_expanded = w.view(B,K,1,1)  # [B,K,1,1]
    wta_mse = ((hypos - gt.unsqueeze(1))**2 * w_expanded).sum(dim=1).mean()
    return wta_mse


def compute_inside_penalty(mixture, hypos, conf_level, num_samples):
    """
    Penalise hypotheses whose log-probability lies below the
    (1−conf_level) Monte-Carlo quantile of the mixture.
    """
    
    q = 1.0 - conf_level
    
    with torch.no_grad():
        
        samples = mixture.sample((num_samples,))          # [N,B,T,2]
        thresh = torch.quantile(mixture.log_prob(samples), q, dim=0)  # [B]
        
    logp_h = mixture.log_prob(hypos.permute(1, 0, 2, 3).contiguous())
    logp_h = logp_h.permute(1, 0, 2).contiguous()  # [B,K,T]
    deficit = F.relu(thresh.unsqueeze(1) - logp_h)  # positive gap
    return deficit.max(dim=1).values.mean()


def compute_discrete_initial_minMSE(cfg, hypos, gt):
    """
    Distance loss squared-L2 (weighted minMSE, best of K):
      L_distance = mean_batch [ min_k ( sum_t ( w_t * ||gt_t - hypos_{k,t}||^2 ) ) ]
    Args:
        cfg (Config): Contains temporal_reweight tensor of shape [T_fut].
        hypos (Tensor): Hypotheses tensor of shape [B, K, T_fut, 2].
        gt (Tensor)   : Ground truth trajectories tensor of shape [B, T_fut, 2].
    Returns:
        Tensor: Scalar weighted minMSE loss.
    """
    
    # per-hypothesis, per-timestep squared error
    sq_err = (gt.unsqueeze(1) - hypos).pow(2).sum(dim=-1)  # [B,K,T_fut]
    
    # apply temporal weights and sum over time
    weighted = sq_err * cfg.temporal_reweight.view(1, 1, -1)  # [B,K,T_fut]
    sum_err = weighted.sum(dim=-1)  # [B,K]
    
    # best-of-K per batch
    min_err = sum_err.min(dim=1)[0]  # [B]
    
    # average across batch
    loss_dist = min_err.mean() / 10  # scalar
    return loss_dist