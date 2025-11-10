import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
from utils import *
from constants import *
import os
from torch.utils.tensorboard import SummaryWriter
import datetime
from datasets.lingo import LingoDataset
from tqdm import tqdm
import time
import wandb

os.environ['ROOT_DIR'] = '..'
os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['CURRENT_TIME'] = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['NCCL_P2P_DISABLE'] = '0'
os.environ['NCCL_IB_DISABLE'] = '0'

import sys
sys.path.append(os.path.join(os.environ['ROOT_DIR'], 'code'))


@hydra.main(version_base=None, config_path="config", config_name="config_train_lingo")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = find_free_port()
    world_size = cfg.num_gpus
    print('Usable GPUS: ', torch.cuda.device_count(), flush=True)
    torch.multiprocessing.spawn(train_ddp,
                                args=(world_size, cfg),
                                nprocs=world_size,
                                join=True)

def train_ddp(rank, world_size, cfg):

    OmegaConf.register_new_resolver("times", lambda x, y: int(x) * int(y))

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    cfg.device = f"cuda:{rank}"
    print(f'Training on {device}', flush=True)
    print('Initializing Distributed', flush=True)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    model = init_model(list(cfg.model.values())[0], device=rank, eval=False, load_state_dict=cfg.load_state_dict)

    synhsi_dataset = LingoDataset(**cfg.dataset)

    sampler = DistributedSampler(synhsi_dataset)
    dataloader = DataLoader(synhsi_dataset, batch_size=cfg.batch_size, drop_last=True, num_workers=cfg.num_workers,
                            sampler=sampler, pin_memory=True)

    trainer = hydra.utils.instantiate(list(cfg.sampler.values())[0])
    trainer.set_dataset_and_model(synhsi_dataset, model)

    optimizer = Adam(model.parameters(), lr=cfg.lr)

    if cfg.use_tensorboard and rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(cfg.exp_dir, 'tensorboard_logs'))
    
    # Determine run ID and checkpoint directory for this run
    if cfg.use_wandb and rank == 0:
        # Try to resolve config, but fallback to unresolved if interpolation errors occur
        try:
            wandb_config = OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            # If resolution fails (e.g., missing interpolation keys), use unresolved config
            wandb_config = OmegaConf.to_container(cfg, resolve=False)
        
        # Create informative run name with key hyperparameters
        run_name = f"{cfg.exp_name}_bs{cfg.batch_size}_lr{cfg.lr}_ws{cfg.max_window_size}_{cfg.scene_type}"
        if hasattr(cfg, 'CURRENT_TIME') and cfg.CURRENT_TIME:
            run_name += f"_{cfg.CURRENT_TIME}"
        elif os.environ.get('CURRENT_TIME'):
            run_name += f"_{os.environ.get('CURRENT_TIME')}"
        
        wandb.init(
            project=cfg.wandb_project if hasattr(cfg, 'wandb_project') else 'lingo-training',
            name=run_name,
            config=wandb_config,
            dir=cfg.exp_dir
        )
        run_id = wandb.run.name
    else:
        run_id = os.environ.get('CURRENT_TIME', 'unknown')
    
    # Create checkpoint directory for this run (fixed for entire training)
    if rank == 0:
        ckpt_folder = os.path.join(cfg.exp_dir, 'checkpoints', run_id)
        os.makedirs(ckpt_folder, exist_ok=True)

    # Only show progress bar on rank 0
    pbar_epochs = tqdm(
        range(cfg.epochs), 
        desc='Training Progress', 
        disable=(rank != 0), 
        position=0,
        unit='epoch',
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    for epoch in pbar_epochs:
        sampler.set_epoch(epoch)
        
        # Create progress bar for dataloader (step-level)
        if rank == 0:
            pbar_batches = tqdm(
                dataloader, 
                desc=f'Epoch {epoch+1}/{cfg.epochs}', 
                disable=False, 
                position=1, 
                leave=False,
                unit='step',
                unit_scale=False,
                ncols=120,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
        else:
            pbar_batches = dataloader
        
        running_loss = 0.0
        step = 0
        epoch_start_time = time.time()
        
        for batch in pbar_batches:
            step_start_time = time.time()
            step += 1
            optimizer.zero_grad()

            joints, mat, scene_flag, text_clip_embedding, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir, pi, need_pi, is_loco = batch
            joints, mat, scene_flag, text_clip_embedding, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir, is_loco = joints.to(device), \
                                                                                                        mat.to(device), scene_flag.to(device), \
                                                                                                        text_clip_embedding.to(device), \
                                                                                                        pelvis_goal.to(device), hand_goal.to(device), \
                                                                                                        is_pick.to(device), need_scene.to(device), need_pelvis_dir.to(device), \
                                                                                                        is_loco.to(device)

            t = torch.randint(0, trainer.timesteps, (cfg.batch_size,), device=device).long()
            with torch.no_grad():
                mask, _, _ = get_mask(joints, -1, p=1., fixed_frame=cfg.auto_regre_num)

            loss = trainer.p_losses(joints, mat, scene_flag, mask, t, text_clip_embedding, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir, is_loco)

            loss.backward()
            optimizer.step()
            
            # Calculate step time
            step_time = time.time() - step_start_time
            steps_per_sec = 1.0 / step_time if step_time > 0 else 0.0
            
            # Update statistics
            loss_item = loss.item()
            running_loss += loss_item
            
            # Update batch progress bar with detailed info
            if rank == 0:
                avg_loss = running_loss / step
                pbar_batches.set_postfix({
                    'loss': f'{loss_item:.6f}',
                    'avg': f'{avg_loss:.6f}',
                    'speed': f'{steps_per_sec:.2f} step/s'
                })
            
            # Log to tensorboard
            if cfg.use_tensorboard and rank == 0:
                writer.add_scalar('Loss', loss_item, epoch * len(dataloader) + step)
                writer.add_scalar('Loss/Average', running_loss / step, epoch * len(dataloader) + step)
                writer.add_scalar('Training/Speed', steps_per_sec, epoch * len(dataloader) + step)
            
            # Log to wandb
            if cfg.use_wandb and rank == 0:
                wandb.log({
                    'Loss': loss_item,
                    'Loss/Average': running_loss / step,
                    'Training/Speed': steps_per_sec,
                    'epoch': epoch,
                    'step': epoch * len(dataloader) + step
                })

        # Update epoch progress bar
        if rank == 0:
            epoch_avg_loss = running_loss / step if step > 0 else 0.0
            epoch_time = time.time() - epoch_start_time
            pbar_epochs.set_postfix({
                'loss': f'{epoch_avg_loss:.6f}',
                'lr': f'{cfg.lr:.6f}',
                'time': f'{epoch_time:.1f}s'
            })
            
            # Log epoch-level metrics to wandb
            if cfg.use_wandb:
                # Use the same step calculation as step-level logs to maintain monotonicity
                current_step = (epoch + 1) * len(dataloader)
                wandb.log({
                    'Epoch/Loss': epoch_avg_loss,
                    'Epoch/Time': epoch_time,
                    'Epoch/LearningRate': cfg.lr
                }, step=current_step)

        if rank == 0 and epoch % cfg.ckpt_interval == 0:
            tqdm.write(f'Saving checkpoint at epoch {epoch}')
            ckpt_filename = f"{cfg.exp_name}_epoch{epoch:03d}.pth"
            torch.save(model.module.state_dict(), os.path.join(ckpt_folder, ckpt_filename))

        torch.distributed.barrier()

        if rank == 0:
            tqdm.write('Clearing cache')
        torch.cuda.empty_cache()
    
    if rank == 0:
        pbar_epochs.close()
        if cfg.use_wandb:
            wandb.finish()


def get_mask(x_start, ind, p, fixed_frame=0, mask_y=True):
    '''
    get mask for the input sequence of pre frames and final goal frame
    '''
    mask_frame = torch.zeros_like(x_start).to(dtype=torch.bool, device=x_start.device)
    mask_goal = torch.zeros_like(x_start).to(dtype=torch.bool, device=x_start.device)

    # goal mask
    if ind != -1:
        rand_batch = torch.rand(x_start.shape[0]).to(x_start.device) < p
        mask_goal[rand_batch, -1, ind * 3: ind * 3 + 3] = True
        if not mask_y:
            mask_goal[rand_batch, -1, ind * 3 + 1] = False

    # prefix frame mask
    if fixed_frame > 0:
        rand_batch = torch.rand(x_start.shape[0]).to(x_start.device) < p
        mask_frame[rand_batch, :fixed_frame, :] = True
    mask = torch.logical_or(mask_frame, mask_goal)
    return mask, mask_frame, mask_goal


if __name__ == '__main__':
    train()
