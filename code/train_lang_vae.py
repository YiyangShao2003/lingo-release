import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
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
from models.lang_vae import LanguageVAE, vae_loss

os.environ['ROOT_DIR'] = '..'
os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['CURRENT_TIME'] = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

import sys
sys.path.append(os.path.join(os.environ['ROOT_DIR'], 'code'))


@hydra.main(version_base=None, config_path="config", config_name="config_train_lang_vae")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    
    device = cfg.device
    print(f'Training on {device}', flush=True)
    
    # Initialize model
    model = LanguageVAE(
        input_dim=cfg.input_dim,
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim
    ).to(device)
    
    # Initialize dataset
    # Use language-only mode so that the dataset does not spend CPU on
    # motion / scene processing that the VAE never uses.
    dataset = LingoDataset(**cfg.dataset, lang_only=True)
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=cfg.lr)
    
    # Logging
    if cfg.use_tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(cfg.exp_dir, 'tensorboard_logs'))
    
    if cfg.use_wandb:
        try:
            wandb_config = OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            wandb_config = OmegaConf.to_container(cfg, resolve=False)
        
        run_name = f"{cfg.exp_name}_bs{cfg.batch_size}_lr{cfg.lr}_ld{cfg.latent_dim}"
        if hasattr(cfg, 'CURRENT_TIME') and cfg.CURRENT_TIME:
            run_name += f"_{cfg.CURRENT_TIME}"
        elif os.environ.get('CURRENT_TIME'):
            run_name += f"_{os.environ.get('CURRENT_TIME')}"
        
        wandb.init(
            project=cfg.wandb_project if hasattr(cfg, 'wandb_project') else 'lang-vae-training',
            name=run_name,
            config=wandb_config,
            dir=cfg.exp_dir
        )
        run_id = wandb.run.name
    else:
        run_id = os.environ.get('CURRENT_TIME', 'unknown')
    
    ckpt_folder = os.path.join(cfg.exp_dir, 'checkpoints', run_id)
    os.makedirs(ckpt_folder, exist_ok=True)
    
    pbar_epochs = tqdm(
        range(cfg.epochs), 
        desc='Training Progress', 
        unit='epoch',
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    best_loss = float('inf')
    
    for epoch in pbar_epochs:
        running_loss = 0.0
        running_recon_loss = 0.0
        running_kl_loss = 0.0
        
        step = 0
        epoch_start_time = time.time()
        
        pbar_batches = tqdm(
            dataloader, 
            desc=f'Epoch {epoch+1}/{cfg.epochs}', 
            leave=False,
            unit='step',
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        for batch in pbar_batches:
            step += 1
            optimizer.zero_grad()
            
            # Extract language embedding from batch
            _, _, _, text_clip_embedding, _, _, _, _, _, _, _, _ = batch
            
            # Move to device
            text_clip_embedding = text_clip_embedding.to(device)  # (B, 1, 768)
            
            # Forward pass
            x_recon, mu, logvar, z = model(text_clip_embedding)
            
            # Calculate loss
            total_loss, recon_loss, kl_loss = vae_loss(
                x_recon, text_clip_embedding, mu, logvar, beta=cfg.beta
            )
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Update statistics
            loss_item = total_loss.item()
            recon_loss_item = recon_loss.item()
            kl_loss_item = kl_loss.item()
            
            running_loss += loss_item
            running_recon_loss += recon_loss_item
            running_kl_loss += kl_loss_item
            
            # Update progress bar
            avg_loss = running_loss / step
            avg_recon_loss = running_recon_loss / step
            avg_kl_loss = running_kl_loss / step
            
            pbar_batches.set_postfix({
                'loss': f'{loss_item:.6f}',
                'avg': f'{avg_loss:.6f}',
                'recon': f'{recon_loss_item:.6f}',
                'kl': f'{kl_loss_item:.6f}',
            })
            
            # Logging
            if cfg.use_tensorboard:
                writer.add_scalar('Loss/Total', loss_item, epoch * len(dataloader) + step)
                writer.add_scalar('Loss/Reconstruction', recon_loss_item, epoch * len(dataloader) + step)
                writer.add_scalar('Loss/KL', kl_loss_item, epoch * len(dataloader) + step)
                writer.add_scalar('Loss/Average', avg_loss, epoch * len(dataloader) + step)
            
            if cfg.use_wandb:
                wandb.log({
                    'Loss/Total': loss_item,
                    'Loss/Reconstruction': recon_loss_item,
                    'Loss/KL': kl_loss_item,
                    'Loss/Average': avg_loss,
                    'epoch': epoch,
                    'step': epoch * len(dataloader) + step,
                })
        
        epoch_avg_loss = running_loss / step if step > 0 else 0.0
        epoch_avg_recon_loss = running_recon_loss / step if step > 0 else 0.0
        epoch_avg_kl_loss = running_kl_loss / step if step > 0 else 0.0
        
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
                'Epoch/Loss_Reconstruction': epoch_avg_recon_loss,
                'Epoch/Loss_KL': epoch_avg_kl_loss,
                'Epoch/Time': epoch_time,
                'Epoch/LearningRate': cfg.lr
            }, step=current_step)
        
        # Save checkpoint
        if epoch % cfg.ckpt_interval == 0:
            tqdm.write(f'Saving checkpoint at epoch {epoch}')
            ckpt_filename = f"{cfg.exp_name}_epoch{epoch:03d}.pth"
            torch.save(model.state_dict(), os.path.join(ckpt_folder, ckpt_filename))
        
        # Save best model
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            tqdm.write(f'Saving best model at epoch {epoch} with loss {best_loss:.6f}')
            torch.save(model.state_dict(), os.path.join(ckpt_folder, 'best.pth'))
        
        torch.cuda.empty_cache()
    
    pbar_epochs.close()
    if cfg.use_wandb:
        wandb.finish()
    
    print(f'Training completed. Best loss: {best_loss:.6f}')
    print(f'Checkpoints saved in: {ckpt_folder}')


if __name__ == '__main__':
    os.environ['HYDRA_FULL_ERROR'] = '1'
    os.environ['ROOT_DIR'] = '../'

    OmegaConf.register_new_resolver("times", lambda x, y: int(x) * int(y))
    train()
