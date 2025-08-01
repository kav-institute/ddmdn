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
from torch.optim.lr_scheduler import ReduceLROnPlateau

from misc import *
from visualization import *

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


def train(cfg, annealing_schedules, train_loader, eval_loader, mesh_grid, model, optimizer, scheduler, visualizer, train_size, eval_size, device):
    
    train_loss_list = []
    eval_loss_list = []
    reliability_list = []
    global_step = 0
    
    # Early stopping
    early_stopping = EarlyStopping(patience=cfg.model.early_stopping_patience)
    
    print(colored(f"Start Training - using gpu: {cfg.gpuid}", 'green'))
    cfg.logger.info(f"Start Training")
    
    for epoch in range(0, (cfg.train.num_total_epochs+1), 1):
        
        #--- Train path
        # Set train mode
        cfg.epoch = epoch
        model.train()
        train_losses = {}
        Train_Forecast_Storage = Forecasts(cfg=cfg, mini_B=cfg.train_mini_batch_size, device='cpu')
        Eval_Forecast_Storage = Forecasts(cfg=cfg, mini_B=cfg.eval_mini_batch_size, device='cpu')
        
        # Annealing schedules
        prune_delta, prune_tau = annealing_schedules._update_mode_prune(epoch=epoch)
        top_p_trunc = annealing_schedules._update_top_p(epoch=epoch)
        
        # Feedback
        if cfg.with_print: print(colored(f"Epoch [{epoch}/{cfg.train.num_total_epochs}]:", 'green'))
        cfg.logger.info(f"Epoch [{epoch}/{cfg.train.num_total_epochs}]:")
        
        # Batch processing
        # agent_past_traj: (B, A, I, 6)
        # agent_future_traj: (B, A, T, 4)
        # agent_plot_future_traj: (B, A, T, 2)
        # target_agent_grid: [B, 3, C, C]
        # agent_mask: (B, A)
        # scene_info [B, dict]
        for batch_idx, (agent_past_traj, agent_future_traj, agent_plot_future_traj, target_agent_grid, agent_mask, scene_info) in enumerate(tqdm(train_loader, desc="Training Batches", disable=not cfg.tqdm_bar)):
            
            # Load data to device
            agent_past_traj_gpu = agent_past_traj.to(device)
            agent_future_traj_gpu = agent_future_traj.to(device)
            target_agent_grid_gpu = target_agent_grid.to(device)
            agent_mask_gpu = agent_mask.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            model_output, model_success = model(
                agent_past=agent_past_traj_gpu,
                gt=agent_future_traj_gpu,
                agent_grid=target_agent_grid_gpu,
                agent_mask=agent_mask_gpu,
                prune_delta=prune_delta, 
                prune_tau=prune_tau,
                top_p_trunc=top_p_trunc
            )
            
            # Train loss
            wta_tau = annealing_schedules._update_wta(current_step=global_step)
            train_loss, train_scores, loss_success = model.compute_loss(
                cfg=cfg,
                model_output=model_output, 
                gt=agent_future_traj_gpu[...,:2],
                wta_tau=wta_tau,
                prefix='Train'
            )
            
            # Check if the output contains invalids or mixture cant be build due to divergence
            if not (model_success and loss_success):
                
                if cfg.with_print: print(colored(f"Training diverged at epoch: {epoch}, exit training...", 'red'))
                cfg.logger.info(f"Training diverged at epoch: {epoch}, exit training...")
                sys.exit()
            
            # Forecast management
            Forecast_Data = Forecast_Batch(
                cfg=cfg, 
                mini_B=cfg.train_mini_batch_size,
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
            Train_Forecast_Storage.add_batch_data(batch_data=Forecast_Data.get_data())
            
            # Backpropagation and Forward
            train_loss.backward()
            
            # Gradient clipping
            if cfg.train.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.train.clip_grad_norm)
            
            # Forward
            optimizer.step()
            global_step += 1
            
            # Accumulate loss
            if batch_idx == 0:
                
                train_scores_dict = train_scores
                
            else:
                
                for k, v in train_scores.items():
                    train_scores_dict[k] += v
            
            # Plot train dataset example forecasts
            if cfg.vis.plot_train_examples and \
                epoch % cfg.vis.plot_epoch_step == 0 and \
                batch_idx % cfg.vis.plot_batch_step == 0 and \
                epoch != 0:
                
                # Create forecasts
                Forecast_Data.build_mesh_confidence_set()
                Forecast_Data.build_confidence_levels()
                
                # Destination handling
                dst_dir = os.path.join(cfg.train_examples_path, f"epoch_{str(epoch).zfill(4)}")
                if not os.path.exists(dst_dir): os.makedirs(dst_dir)
                
                # Plot examples
                plot_per_timestep_full_examples(
                    cfg=cfg,
                    visualizer=visualizer,
                    scene_info=scene_info,
                    agent_past_traj=agent_past_traj,
                    agent_future_traj=agent_future_traj[...,:2], 
                    agent_plot_future_traj=agent_plot_future_traj, 
                    agent_mask=agent_mask,
                    Forecast_Data=Forecast_Data, 
                    dst_dir=dst_dir
                )
        
        # Compute average epoch losses
        for k, v in train_scores_dict.items():
            
            train_losses[k] = v / train_size
        
        # Compute train reliability
        train_reliability_scores = Train_Forecast_Storage.compute_reliability(epoch=epoch, type='Train')
        
        # Save this epochs train loss data
        train_loss_list.append(train_losses)
        
        # Compute train scores
        train_scores = {
            **Train_Forecast_Storage.compute_min_displacement_bok()
            }
        
        # Save current weights
        save_model(
            model=model, 
            optimizer=optimizer,
            epoch=epoch, 
            annealing_status=annealing_schedules.get_status(),
            dst_dir=cfg.checkpoint_path, 
            prefix=f"model_weights_latest.pt"
            )
        
        # When to run evaluation
        if epoch % cfg.train.eval_step == 0 or epoch == 0:
        
            #--- Eval path
            with torch.no_grad():
                
                # Set eval mode
                model.eval()
                eval_losses = {}
                eval_scores = {}
                
                # Batch processing
                # agent_past_traj: (B, A, I, 6)
                # agent_future_traj: (B, A, T, 4)
                # agent_plot_future_traj: (B, A, T, 2)
                # target_agent_grid: [B, 3, C, C]
                # agent_mask: (B, A)
                # scene_info [B, dict]
                for batch_idx, (agent_past_traj, agent_future_traj, agent_plot_future_traj, target_agent_grid, agent_mask, scene_info) in enumerate(tqdm(eval_loader, desc="Eval Batches", disable=not cfg.tqdm_bar)):
                    
                    # Load data to device
                    agent_past_traj_gpu = agent_past_traj.to(device)
                    agent_future_traj_gpu = agent_future_traj.to(device)
                    target_agent_grid_gpu = target_agent_grid.to(device)
                    agent_mask_gpu = agent_mask.to(device)
                    
                    # Forward pass
                    model_output, model_success = model(
                        agent_past=agent_past_traj_gpu,
                        gt=agent_future_traj_gpu,
                        agent_grid=target_agent_grid_gpu,
                        agent_mask=agent_mask_gpu,
                        prune_delta=prune_delta, 
                        prune_tau=prune_tau,
                        top_p_trunc=top_p_trunc
                    )
                    
                    # Eval loss
                    _, eval_scores, loss_success = model.compute_loss(
                        cfg=cfg,
                        model_output=model_output, 
                        gt=agent_future_traj_gpu[...,:2],
                        wta_tau=wta_tau,
                        prefix='Eval'
                    )
                    
                    # Check if the output contains invalids or mixture cant be build due to divergence
                    if not (model_success and loss_success):
                        
                        if cfg.with_print: print(colored(f"Training diverged at epoch: {epoch}, exit training...", 'red'))
                        cfg.logger.info(f"Training diverged at epoch: {epoch}, exit training...")
                        sys.exit()
                    
                    # Forecast management
                    Forecast_Data = Forecast_Batch(
                        cfg=cfg, 
                        mini_B=cfg.eval_mini_batch_size,
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
                    Eval_Forecast_Storage.add_batch_data(batch_data=Forecast_Data.get_data())
                    
                    # Accumulate loss
                    if batch_idx == 0:
                        
                        eval_scores_dict = eval_scores
                        
                    else:
                        
                        for k, v in eval_scores.items():
                            eval_scores_dict[k] += v
                    
                    # Plot eval dataset example forecasts
                    if cfg.vis.plot_eval_examples and \
                        epoch % cfg.vis.plot_epoch_step == 0 and \
                        batch_idx % cfg.vis.plot_batch_step == 0 and \
                        epoch != 0:
                        
                        # Create forecasts
                        Forecast_Data.build_mesh_confidence_set()
                        Forecast_Data.build_confidence_levels()
                        
                        # Destination handling
                        dst_dir = os.path.join(cfg.eval_examples_path, f"epoch_{str(epoch).zfill(4)}")
                        if not os.path.exists(dst_dir): os.makedirs(dst_dir)
                        
                        # Plot examples
                        plot_per_timestep_full_examples(
                            cfg=cfg,
                            visualizer=visualizer,
                            scene_info=scene_info,
                            agent_past_traj=agent_past_traj,
                            agent_future_traj=agent_future_traj[...,:2], 
                            agent_plot_future_traj=agent_plot_future_traj,
                            agent_mask=agent_mask, 
                            Forecast_Data=Forecast_Data, 
                            dst_dir=dst_dir
                        )
                
                # Compute average epoch losses
                for k, v in eval_scores_dict.items():
                
                    eval_losses[k] = v / eval_size
                
                # Learning rate update and schedular step
                current_lr = optimizer.param_groups[0]["lr"]
                
                # Compute eval reliability
                eval_reliability_scores = Eval_Forecast_Storage.compute_reliability(epoch=epoch, type='Eval')
                
                # Compute eval scores
                eval_scores = {
                    **Eval_Forecast_Storage.compute_min_displacement_bok()
                    }
                
                # Save this epochs eval loss data
                eval_loss_list.append(eval_losses)
                
                # Get evaluation loss
                if cfg.with_discrete:
                    
                    final_eval_score = get_score(
                        avg_rel=float(eval_reliability_scores['Eval_avg_RLS'].split()[0]),
                        min_rel=float(eval_reliability_scores['Eval_min_RLS'].split()[0]), 
                        ade=float(eval_scores['min_ADE@(K=20)'].split()[0]), 
                        fde=float(eval_scores['min_FDE@(K=20)'].split()[0]), 
                        mode="weighted_avg"
                        )
                
                else:
                    
                    final_eval_score = get_score(
                        avg_rel=float(eval_reliability_scores['Eval_avg_RLS'].split()[0]),
                        min_rel=float(eval_reliability_scores['Eval_min_RLS'].split()[0]), 
                        ade=0.0, 
                        fde=0.0, 
                        mode="weighted_avg"
                        )
                
                # Save this epochs reliability data
                reliability_list.append(train_reliability_scores | eval_reliability_scores | {"Eval_Score": f"{final_eval_score:.2f} %"})
                
                # Early stopping check
                best_eval_score, best_epoch = early_stopping(
                    cfg=cfg,
                    val_loss=final_eval_score,
                    model=model, 
                    optimizer=optimizer, 
                    epoch=epoch, 
                    annealing_status=annealing_schedules.get_status(),
                    dst_dir=cfg.checkpoint_path,
                    prefix=f"model_weights_best.pt"
                )
                
        else:
            
            eval_loss_list.append(eval_loss_list[-1])
            reliability_list.append(train_reliability_scores | eval_reliability_scores | {"Eval_Score": f"{final_eval_score:.2f} %"})
            
        # Scheduler step
        if isinstance(scheduler, ReduceLROnPlateau): scheduler.step(eval_losses.get('mixture_per_step_NLL', float('nan')))
        else: scheduler.step()
            
        # Auto save of weights
        if epoch % cfg.train.auto_save_step == 0 and epoch != 0:
            
            save_model(
                model=model, 
                optimizer=optimizer,
                annealing_status=annealing_schedules.get_status(), 
                epoch=epoch, 
                dst_dir=cfg.checkpoint_path,
                prefix=f"model_weights_{str(epoch).zfill(4)}.pt"
                )
            
        # Plot losses and reliability scores
        plot_loss(
            train_loss_list=train_loss_list, 
            eval_loss_list=eval_loss_list, 
            reliability_list=reliability_list, 
            plot_dir=cfg.train_path)
        
        # Feedback
        if cfg.with_print: 
            print(colored("Train Losses: || " + "".join(f"{k}: {v:.2f} - " for k, v in train_losses.items()) + f"LR: {current_lr:.8f}", 'green'))
            if epoch % cfg.train.eval_step == 0 or epoch == 0: print(colored("Eval Losses:  || " + "".join(f"{k}: {v:.2f} - " for k, v in eval_losses.items()) + f"Eval Score: {best_eval_score:.2f} % - @ Epoch: {best_epoch}", 'green'))
            print(colored(f"Train Scores: || RLS_avg: {train_reliability_scores['Train_avg_RLS']} - RLS_min: {train_reliability_scores['Train_min_RLS']} - " + "".join(f"{k}: {v} - " for k, v in train_scores.items()), 'cyan'))
            if epoch % cfg.train.eval_step == 0 or epoch == 0: print(colored(f"Eval Scores:  || RLS_avg: {eval_reliability_scores['Eval_avg_RLS']} - RLS_min: {eval_reliability_scores['Eval_min_RLS']} - " + "".join(f"{k}: {v} - " for k, v in eval_scores.items()), 'cyan'))
            
        cfg.logger.info("Train Losses: || " + "".join(f"{k}: {v:.2f} - " for k, v in train_losses.items()) + f"LR: {current_lr:.8f}")
        if epoch % cfg.train.eval_step == 0 or epoch == 0: cfg.logger.info("Eval Losses:  || " + "".join(f"{k}: {v:.2f} - " for k, v in eval_losses.items()) + f"Eval Score: {best_eval_score:.2f} % - @ Epoch: {best_epoch}")
        cfg.logger.info(f"Train Scores: || RLS_avg: {train_reliability_scores['Train_avg_RLS']} - RLS_min: {train_reliability_scores['Train_min_RLS']} - " + "".join(f"{k}: {v} - " for k, v in train_scores.items()))
        if epoch % cfg.train.eval_step == 0 or epoch == 0: cfg.logger.info(f"Eval Scores:  || RLS_avg: {eval_reliability_scores['Eval_avg_RLS']} - RLS_min: {eval_reliability_scores['Eval_min_RLS']} - " + "".join(f"{k}: {v} - " for k, v in eval_scores.items()))
        
        # Early stopping
        if early_stopping.early_stop:
        
            # Feedback
            if cfg.with_print: print(colored(f"Early Stopping - Shutdown Training...", 'green'))
            cfg.logger.info(f"Early Stopping - Shutdown Training...")
            break
        
    return


def run_training(cfg, device):
    
    print(colored(f"Training for: {args.cfg} config - on gpu: {args.gpu}", 'green'))
    cfg.logger.info(f"Training for: {args.cfg} config - on gpu: {args.gpu}")
    
    # Build mesh grid
    mesh_grid, target_x_layer, target_y_layer = build_mesh_grid(
        mesh_range_x=cfg.grid.size_meter,
        mesh_range_y=cfg.grid.size_meter, 
        mesh_resolution=cfg.full_grid_resolution
    )
    
    # Create cord conv representation
    cord_convs = torch.stack([target_x_layer, target_y_layer], dim=0).unsqueeze(0)
    train_cord_convs_batched = cord_convs.expand(cfg.train_batch_size, cord_convs.shape[1], cfg.grid.full_cell_size, cfg.grid.full_cell_size)
    eval_cord_convs_batched = cord_convs.expand(cfg.eval_batch_size, cord_convs.shape[1], cfg.grid.full_cell_size, cfg.grid.full_cell_size)
    
    
    # Dataset specific train data dataloader
    train_dataset = HDF5_Dataset(
        cfg=cfg,
        path=cfg.train_data_path,
        typ='train',
        size=cfg.train.dataloader_train_size
    )
    
    # Dataset specific eval data dataloader
    eval_dataset = HDF5_Dataset(
        cfg=cfg,
        path=cfg.eval_data_path, 
        typ='eval',
        size=cfg.train.dataloader_eval_size
    )
    
    # Feedback
    print(colored(f"Loaded train data: {len(train_dataset)} samples", 'green'))
    print(colored(f"Loaded eval data: {len(eval_dataset)} samples", 'green'))
    cfg.logger.info(f"Loaded train data: {len(train_dataset)} samples")
    cfg.logger.info(f"Loaded eval data: {len(eval_dataset)} samples")
    
    # Define dataset collate function
    collate_train_fn_with_params = partial(
        collate_fn, 
        cfg=cfg,
        typ='train',
        cord_convs_batched=train_cord_convs_batched,
    )
    
    collate_eval_fn_with_params = partial(
        collate_fn, 
        cfg=cfg,
        typ='eval',
        cord_convs_batched=eval_cord_convs_batched,
    )
    
    # Dataloader init function for seeds
    def worker_init_fn(worker_id):
        
        seed = cfg.common.random_seed + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Setup Train dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_train_fn_with_params,
        drop_last=True,
        pin_memory=cfg.train.dataloader_pin_memory,
        num_workers=cfg.train.dataloader_num_workers,
        prefetch_factor=cfg.train.dataloader_prefetch,
        persistent_workers=True
    )
    
    # Setup Eval dataloader
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_eval_fn_with_params,
        drop_last=True,
        pin_memory=cfg.train.dataloader_pin_memory,
        num_workers=cfg.train.dataloader_num_workers,
        prefetch_factor=cfg.train.dataloader_prefetch,
        persistent_workers=True
    )
    
    annealing_schedules = AnnealingSchedules(
        cfg=cfg, 
        train_data_size=len(train_loader.dataset.indices), 
        train_batch_size=train_loader.batch_size
    )
    
    # Setup Model
    model = DDMDN(cfg=cfg)
    
    # Setup Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.train.start_learning_rate, 
        weight_decay=cfg.train.weight_decay
    )
    
    # Resume a previous training or use pretrained weights
    if cfg.train.resume_train:
        
        # Load
        checkpoint = torch.load(f=os.path.join(cfg.checkpoint_path, cfg.train.train_weights), map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Shift optimizer states to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        
    # Setup Learning Rate Scheduleres
    scheduler = build_scheduler(cfg=cfg, optimizer=optimizer)
    
    # Setup dataset specific visualizer
    if 'imptc' in cfg.target: visualizer = IMPTC_Visualizer(cfg=cfg)
    elif 'ind' in cfg.target: visualizer = IND_Visualizer(cfg=cfg)
    elif 'sdd' in cfg.target: visualizer = SDD_Visualizer(cfg=cfg)
    elif 'eth' in cfg.target : visualizer = ETH_Visualizer(cfg=cfg)
    elif 'hotel' in cfg.target : visualizer = ETH_Visualizer(cfg=cfg)
    elif 'univ' in cfg.target : visualizer = ETH_Visualizer(cfg=cfg)
    elif 'zara01' in cfg.target : visualizer = ETH_Visualizer(cfg=cfg)
    elif 'zara02' in cfg.target : visualizer = ETH_Visualizer(cfg=cfg)
    
    # Load model to processing device (gpu)
    model.to(device)
    train_size = len(train_loader)
    eval_size = len(eval_loader)
    
    # Start training loop
    train(
        cfg=cfg,
        annealing_schedules=annealing_schedules,
        train_loader=train_loader,
        eval_loader=eval_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler, 
        visualizer=visualizer,
        mesh_grid=mesh_grid.to(device),
        train_size=train_size, 
        eval_size=eval_size, 
        device=device
        )
    
    # Close logging
    cfg.log_file_handler.close()
    cfg.logger.removeHandler(cfg.log_file_handler)
    
    # Feedbck
    print(colored(f'End training, shutting down...', 'green'))
    cfg.logger.info(f'End training, shutting down...')
    return


if __name__ == "__main__":
    
    clear_cuda()
    
    # Get args
    args = parse_args()
    
    # Setup gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup framework
    cfg = Config_Setup(args=args, mode='train', dir=os.getcwd())
    
    # Set seeds
    random.seed(cfg.common.random_seed)
    np.random.seed(cfg.common.random_seed)
    torch.manual_seed(cfg.common.random_seed)
    torch.cuda.manual_seed(cfg.common.random_seed)
    torch.cuda.manual_seed_all(cfg.common.random_seed)
    
    # Run training
    run_training(cfg=cfg, device=device)
    sys.exit(0)