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
import random
from models.lang_vae import LanguageVAE

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

    # Load VAE for language latent compression
    lang_vae = None
    if cfg.use_lang_vae and cfg.lang_vae_ckpt:
        lang_vae = LanguageVAE(
            input_dim=768,
            latent_dim=64,
            hidden_dim=256
        ).to(device)
        lang_vae.load_state_dict(torch.load(cfg.lang_vae_ckpt, map_location=device))
        lang_vae.eval()
        print(f'Loaded language VAE from {cfg.lang_vae_ckpt}', flush=True)
        # Set language feature dim to 64 when using VAE
        list(cfg.model.values())[0].language_feature_dim = 64
    else:
        # Set language feature dim to 768 when not using VAE
        list(cfg.model.values())[0].language_feature_dim = 768

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
    
    if cfg.use_wandb and rank == 0:
        try:
            wandb_config = OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            wandb_config = OmegaConf.to_container(cfg, resolve=False)
        
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
    
    if rank == 0:
        ckpt_folder = os.path.join(cfg.exp_dir, 'checkpoints', run_id)
        os.makedirs(ckpt_folder, exist_ok=True)

    pbar_epochs = tqdm(
        range(cfg.epochs), 
        desc='Training Progress', 
        disable=(rank != 0), 
        position=0,
        unit='epoch',
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    N_timesteps = trainer.timesteps

    for epoch in pbar_epochs:
        sampler.set_epoch(epoch)
        
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
        running_loss_motion = 0.0
        running_loss_lang = 0.0
        
        step = 0
        epoch_start_time = time.time()
        
        for batch in pbar_batches:
            step_start_time = time.time()
            step += 1
            optimizer.zero_grad()

            joints, mat, scene_flag, text_clip_embedding, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir, _, _, is_loco = batch
            
            x_start = joints
            lang_emb_start = text_clip_embedding

            # Move all *required* data to device
            x_start, mat, scene_flag, lang_emb_start, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir, is_loco = \
                x_start.to(device), mat.to(device), scene_flag.to(device), lang_emb_start.to(device), \
                pelvis_goal.to(device), hand_goal.to(device), is_pick.to(device), need_scene.to(device), \
                need_pelvis_dir.to(device), is_loco.to(device)
            
            # Encode language embedding to 64-dim latent if VAE is used
            if lang_vae is not None:
                with torch.no_grad():
                    lang_emb_start = lang_vae.encode_to_latent(lang_emb_start)  # (B, 64)
                    lang_emb_start = lang_emb_start.unsqueeze(1)  # (B, 1, 64)


            t_motion = torch.randint(0, N_timesteps, (cfg.batch_size,), device=device).long()
            t_text = torch.randint(0, N_timesteps, (cfg.batch_size,), device=device).long()
            
            loss_motion_mask = 1.0
            loss_lang_mask = 1.0

            mode = random.choice(['t2m', 'm2t', 'joint', 'uncond_m', 'uncond_t'])

            if mode == 't2m':
                t_text.fill_(0)
                loss_lang_mask = 0.0
            
            elif mode == 'm2t':
                t_motion.fill_(0)
                loss_motion_mask = 0.0
            
            elif mode == 'joint':
                pass
            
            elif mode == 'uncond_m':
                t_text.fill_(N_timesteps - 1)
                loss_lang_mask = 0.0
            
            elif mode == 'uncond_t':
                t_motion.fill_(N_timesteps - 1)
                loss_motion_mask = 0.0

            with torch.no_grad():
                mask, _, _ = get_mask(x_start, -1, p=1., fixed_frame=cfg.auto_regre_num)
            loss_motion, loss_lang = trainer.p_losses(
                x_start, lang_emb_start, mat, scene_flag, mask, 
                t_motion, t_text, 
                pelvis_goal, hand_goal, is_pick, 
                need_scene, need_pelvis_dir, is_loco
            )

            total_loss = (loss_motion * loss_motion_mask) + (loss_lang * loss_lang_mask)

            total_loss.backward()
            optimizer.step()
            
            # Calculate step time
            step_time = time.time() - step_start_time
            steps_per_sec = 1.0 / step_time if step_time > 0 else 0.0
            
            # Update statistics
            loss_item = total_loss.item()
            loss_motion_item = loss_motion.item()
            loss_lang_item = loss_lang.item()
            
            running_loss += loss_item
            running_loss_motion += loss_motion_item
            running_loss_lang += loss_lang_item
            
            # --- 5. Logging ---
            if rank == 0:
                avg_loss = running_loss / step
                avg_loss_motion = running_loss_motion / step
                avg_loss_lang = running_loss_lang / step
                
                pbar_batches.set_postfix({
                    'total_loss': f'{loss_item:.6f}',
                    'avg': f'{avg_loss:.6f}',
                    'm_loss': f'{loss_motion_item:.6f}',
                    'l_loss': f'{loss_lang_item:.6f}',
                    'mode': mode,
                    'speed': f'{steps_per_sec:.2f} step/s'
                })
            
            if cfg.use_tensorboard and rank == 0:
                writer.add_scalar('Loss/Total', loss_item, epoch * len(dataloader) + step)
                writer.add_scalar('Loss/Average', avg_loss, epoch * len(dataloader) + step)
                writer.add_scalar('Loss/Motion', loss_motion_item, epoch * len(dataloader) + step)
                writer.add_scalar('Loss/Language', loss_lang_item, epoch * len(dataloader) + step)
                writer.add_scalar('Loss/Motion_Average', avg_loss_motion, epoch * len(dataloader) + step)
                writer.add_scalar('Loss/Language_Average', avg_loss_lang, epoch * len(dataloader) + step)
                writer.add_scalar('Training/Speed', steps_per_sec, epoch * len(dataloader) + step)
            
            if cfg.use_wandb and rank == 0:
                wandb.log({
                    'Loss/Total': loss_item,
                    'Loss/Average': avg_loss,
                    'Loss/Motion': loss_motion_item,
                    'Loss/Language': loss_lang_item,
                    'Loss/Motion_Average': avg_loss_motion,
                    'Loss/Language_Average': avg_loss_lang,
                    'Training/Speed': steps_per_sec,
                    'epoch': epoch,
                    'step': epoch * len(dataloader) + step,
                    'mode': mode
                })

        if rank == 0:
            epoch_avg_loss = running_loss / step if step > 0 else 0.0
            epoch_avg_loss_motion = running_loss_motion / step if step > 0 else 0.0
            epoch_avg_loss_lang = running_loss_lang / step if step > 0 else 0.0
            
            epoch_time = time.time() - epoch_start_time
            pbar_epochs.set_postfix({
                'loss': f'{epoch_avg_loss:.6f}',
                'lr': f'{cfg.lr:.6f}',
                'time': f'{epoch_time:.1f}s'
            })
            
            if cfg.use_wandb:
                current_step = (epoch + 1) * len(dataloader)
                wandb.log({
                    'Epoch/Loss_Total': epoch_avg_loss,
                    'Epoch/Loss_Motion': epoch_avg_loss_motion,
                    'Epoch/Loss_Language': epoch_avg_loss_lang,
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