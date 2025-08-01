import torch
import random
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg") 

sys.path.append("/workspace/repos/framework")
sys.path.append("/workspace/repos/framework/ddmdn")

from tqdm import tqdm
from termcolor import colored
from functools import partial
from torch.utils.data import DataLoader
from ddmdn.model import DDMDN
from ddmdn.loader import HDF5_Dataset, collate_fn

from misc import *
from visualization import *


torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


def test(cfg, annealing_status, test_loader, mesh_grid, model, per_step_visualizer, per_full_traj_visualizer, test_size, device):
    
    # Plot destination handling
    per_step_dst_dir = os.path.join(cfg.test_examples_path, 'per_step')
    per_full_traj_dst_dir = os.path.join(cfg.test_examples_path, 'per_traj')
    if not os.path.exists(per_step_dst_dir): os.makedirs(per_step_dst_dir)
    if not os.path.exists(per_full_traj_dst_dir): os.makedirs(per_full_traj_dst_dir)
    
    print(colored(f"Start Testing", 'green'))
    cfg.logger.info(f"Start Testing")
    
    with torch.no_grad():
        
        # Set test mode
        model.eval()
        test_losses = {}
        Test_Forecast_Storage = Forecasts(cfg=cfg, mini_B=cfg.test_mini_batch_size, device='cpu')
        
        prune_delta = annealing_status['mode_prune_delta']
        prune_tau = annealing_status['mode_prune_tau']
        top_p_trunc = annealing_status['top_p_trunc']
        wta_tau = annealing_status['wta_tau']
        
        # Batch processing
        # agent_past_traj: (B, A, I, 6)
        # agent_future_traj: (B, A, T, 4)
        # agent_plot_future_traj: (B, A, T, 2)
        # target_agent_grid: [B, 3, C, C]
        # agent_mask: (B, A)
        # scene_info [B, dict]
        for batch_idx, (agent_past_traj, agent_future_traj, agent_plot_future_traj, target_agent_grid, agent_mask, scene_info) in enumerate(tqdm(test_loader, desc="Testing Batches")):
            
            # Load data to device
            agent_past_traj_gpu = agent_past_traj.to(device)
            agent_future_traj_gpu = agent_future_traj.to(device)
            target_agent_grid = target_agent_grid.to(device)
            agent_mask_gpu = agent_mask.to(device)
            
            # Forward pass
            model_output, model_success = model(
                agent_past=agent_past_traj_gpu,
                gt=agent_future_traj_gpu,
                agent_grid=target_agent_grid,
                agent_mask=agent_mask_gpu,
                prune_delta=prune_delta, 
                prune_tau=prune_tau,
                top_p_trunc=top_p_trunc
            )
            
            # Train loss
            _, test_scores, loss_success = model.compute_loss(
                cfg=cfg,
                model_output=model_output, 
                gt=agent_future_traj_gpu[...,:2],
                wta_tau=wta_tau,
                prefix='Test'
            )
            
            # Check if the output contains invalids or mixture cant be build due to divergence
            if not (model_success and loss_success):
                
                print(colored(f"Testing diverged, exit...", 'red'))
                cfg.logger.info(f"Testing diverged, exit...")
                sys.exit()
            
            # Forecast management
            Forecast_Data = Forecast_Batch(
                cfg=cfg, 
                mini_B=cfg.test_mini_batch_size,
                hypos=model_output['hypos'], 
                hypos_prob=model_output['hypos_prob'],
                top_mode_hypos=model_output["top_mode_hypos"],
                top_mode_hypos_mask=model_output["top_mode_hypos_mask"],
                per_step_mixture=model_output['mdn_per_step_mixture'], 
                per_anchor_mixture=model_output['mdn_per_anchor_mixture'], 
                per_step_mixture_flat=model_output['mdn_per_step_mixture_flat'],
                agent_past_traj=agent_past_traj, 
                agent_future_traj=agent_future_traj_gpu[...,:2], 
                agent_mask=agent_mask,
                mesh_grid=mesh_grid, 
                scene_info=scene_info, 
                device=device
            )
            
            Forecast_Data.build_gt_confidence_set()
            #Forecast_Data.build_mesh_confidence_set()
            Test_Forecast_Storage.add_batch_data(batch_data=Forecast_Data.get_data())
            
            # Accumulate loss
            if batch_idx == 0:
                
                test_scores_dict = test_scores
                
            else:
                
                for k, v in test_scores.items():
                    test_scores_dict[k] += v
            
            # Plot test dataset per timestep example forecasts
            if cfg.test.plot_per_step_examples and batch_idx % cfg.test.plot_batch_step == 0:
                
                # Create forecasts
                Forecast_Data.build_confidence_levels()
                
                # Plot per timestep examples
                plot_per_timestep_full_examples(
                    cfg=cfg,
                    visualizer=per_step_visualizer,
                    scene_info=scene_info,
                    agent_past_traj=agent_past_traj,
                    agent_future_traj=agent_future_traj[...,:2], 
                    agent_plot_future_traj=agent_plot_future_traj, 
                    agent_mask=agent_mask,
                    Forecast_Data=Forecast_Data, 
                    dst_dir=per_step_dst_dir
                )
            
            # Plot test dataset per full trajectory example forecasts
            if cfg.test.plot_per_traj_examples and batch_idx % cfg.test.plot_batch_step == 0:
                
                # Plot per full trajectory examples
                plot_per_full_trajectory_examples(
                    cfg=cfg,
                    visualizer=per_full_traj_visualizer,
                    scene_info=scene_info,
                    agent_past_traj=agent_past_traj,
                    agent_future_traj=agent_future_traj[...,:2], 
                    agent_plot_future_traj=agent_plot_future_traj, 
                    agent_mask=agent_mask,
                    prune_delta=prune_delta,
                    Forecast_Data=Forecast_Data, 
                    dst_dir=per_full_traj_dst_dir
                )
        
        # Compute average epoch losses
        for k, v in test_scores_dict.items():
            
            test_losses[k] = v / test_size
        
        # Compute test scores
        test_scores = {
            **Test_Forecast_Storage.compute_reliability(epoch=0, type='Test'),
            **Test_Forecast_Storage.compute_sharpness(),
            **Test_Forecast_Storage.compute_min_displacement_bok(), 
            **Test_Forecast_Storage.compute_min_displacement_top()
            }
        
        # Feedback
        print(colored("Test Losses:  || " + "".join(f"{k}: {v:.2f} - " for k, v in test_losses.items()), 'green'))
        print(colored("Test Scores:  || " + "".join(f"{k}: {v} - " for k, v in test_scores.items()), 'cyan'))
        cfg.logger.info("Test Losses:  || " + "".join(f"{k}: {v:.2f} - " for k, v in test_losses.items()))
        cfg.logger.info("Test Scores:  || " + "".join(f"{k}: {v} - " for k, v in test_scores.items()))
        
    return


def run_test(cfg, device):
    
    print(colored(f"Testing for: {args.cfg} config - on gpu: {args.gpu}", 'green'))
    cfg.logger.info(f"Testing for: {args.cfg} config - on gpu: {args.gpu}")
    
    # Build mesh grid
    mesh_grid, target_x_layer, target_y_layer = build_mesh_grid(
        mesh_range_x=cfg.grid.size_meter,
        mesh_range_y=cfg.grid.size_meter, 
        mesh_resolution=cfg.full_grid_resolution
    )
    
    # Create cord conv representation
    cord_convs = torch.stack([target_x_layer, target_y_layer], dim=0).unsqueeze(0)
    cord_convs_batched = cord_convs.expand(cfg.test_batch_size, cord_convs.shape[1], cfg.grid.full_cell_size, cfg.grid.full_cell_size)
    
    # Dataset specific train data dataloader
    test_dataset = HDF5_Dataset(
        cfg=cfg,
        path=cfg.test_data_path,
        typ='test',
        size=cfg.test.dataloader_size
    )
    
    # Feedback
    print(colored(f"Loaded train data: {len(test_dataset)} samples", 'green'))
    cfg.logger.info(f"Loaded train data: {len(test_dataset)} samples")
    
    # Define dataset collate function
    collate_fn_with_params = partial(
        collate_fn, 
        cfg=cfg,
        typ='test',
        cord_convs_batched=cord_convs_batched,
    )
    
    # Dataloader init function for seeds
    def worker_init_fn(worker_id):
        
        seed = cfg.common.random_seed + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Setup Test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn_with_params,
        drop_last=True,
        pin_memory=cfg.test.dataloader_pin_memory,
        num_workers=cfg.test.dataloader_num_workers,
        prefetch_factor=cfg.test.dataloader_prefetch,
        persistent_workers=True
    )
    
    # Setup Model and load weights
    model = DDMDN(cfg=cfg)
    checkpoint = torch.load(f=os.path.join(cfg.checkpoint_path, cfg.test.model_weights), map_location=device)
    model.load_state_dict(checkpoint['model'])
    annealing_status = checkpoint['annealing_status']
    cfg.wta_tau.append(int(1.0))
    cfg.prune_delta.append(int(1.0))
    cfg.prune_tau.append(int(1.0))
    
    # Setup dataset specific visualizer
    VIS_MAP = {
        'imptc': IMPTC_Visualizer,
        'ind':   IND_Visualizer,
        'sdd':   SDD_Visualizer,
        'eth':   ETH_Visualizer,
        'hotel': ETH_Visualizer,
        'univ':  ETH_Visualizer,
        'zara01': ETH_Visualizer,
        'zara02': ETH_Visualizer,
    }
    
    for key, Vis in VIS_MAP.items():
        if key in cfg.target:
            per_step_visualizer      = Vis(cfg=cfg)
            per_full_traj_visualizer = Vis(cfg=cfg)
            break
    
    # Load model to processing device
    model.to(device)
    test_size = len(test_loader)
    
    # Start training loop
    test(
        cfg=cfg,
        annealing_status=annealing_status,
        test_loader=test_loader,
        model=model,
        per_step_visualizer=per_step_visualizer, 
        per_full_traj_visualizer=per_full_traj_visualizer,
        mesh_grid=mesh_grid.to(device),
        test_size=test_size, 
        device=device
        )
    
    # Close logging
    cfg.log_file_handler.close()
    cfg.logger.removeHandler(cfg.log_file_handler)
    
    # Feedbck
    print(colored(f'End testing, shutting down...', 'green'))
    cfg.logger.info(f'End testing, shutting down...')
    return


if __name__ == "__main__":
    
    clear_cuda()
    
    # Get args
    args = parse_args()
    
    # Setup gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup framework
    res_dir = '/workspace/data/trained_models'
    cfg = Config_Setup(args=args, mode='test', dir=res_dir)
    
    # Set seeds
    random.seed(cfg.common.random_seed)
    np.random.seed(cfg.common.random_seed)
    torch.manual_seed(cfg.common.random_seed)
    torch.cuda.manual_seed(cfg.common.random_seed)
    torch.cuda.manual_seed_all(cfg.common.random_seed)
    
    # Run training
    run_test(cfg=cfg, device=device)
    sys.exit(0)