import os
import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import hydra
import wandb

from datasets.lingo_vae import LingoVAEDataset
from models.latent_vae import MotionVAE
from utils import seed_everything


@hydra.main(version_base=None, config_path="config", config_name="config_train_vae")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    os.environ['ROOT_DIR'] = '..'
    os.environ['HYDRA_FULL_ERROR'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['CURRENT_TIME'] = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

    OmegaConf.register_new_resolver("times", lambda x, y: int(x) * int(y))

    seed_everything(cfg.seed)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU!")
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    
    # Lightweight dataset: only joints, minimal CPU preprocessing
    dataset = LingoVAEDataset(
        folder=cfg.dataset.folder,
        step=cfg.dataset.step,
        max_window_size=cfg.dataset.max_window_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,
        persistent_workers=True if cfg.num_workers > 0 else False,
        prefetch_factor=2 if cfg.num_workers > 0 else None
    )

    model: MotionVAE = hydra.utils.instantiate(cfg.vae.model)
    model.to(device)
    print(f"Model device: {next(model.parameters()).device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    os.makedirs(cfg.exp_dir, exist_ok=True)
    ckpt_dir = os.path.join(cfg.exp_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = cfg.vae.ckpt_path

    mse = nn.MSELoss()

    # WandB setup
    if cfg.get('use_wandb', False):
        try:
            wandb_config = OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            wandb_config = OmegaConf.to_container(cfg, resolve=False)
        
        run_name = f"{cfg.exp_name}_bs{cfg.batch_size}_lr{cfg.lr}_beta{cfg.beta_kl}_latent{cfg.latent.dim}"
        if os.environ.get('CURRENT_TIME'):
            run_name += f"_{os.environ.get('CURRENT_TIME')}"
        
        wandb.init(
            project=cfg.get('wandb_project', 'lingo-vae-training'),
            name=run_name,
            config=wandb_config,
            dir=cfg.exp_dir
        )
        print(f"WandB initialized: {wandb.run.name}")

    # Logging setup
    log_file = os.path.join(cfg.exp_dir, 'training_log.txt')
    with open(log_file, 'w') as f:
        f.write(f"VAE Training Log\n")
        f.write(f"Config: {OmegaConf.to_yaml(cfg)}\n")
        f.write(f"{'='*80}\n")

    pbar_epochs = tqdm(range(cfg.epochs), desc='VAE Training', unit='epoch')

    best_loss = float('inf')
    nan_epochs = 0
    max_nan_epochs = 3

    for epoch in pbar_epochs:
        model.train()
        running_loss = 0.0
        running_recon = 0.0
        running_kl = 0.0
        step_count = 0
        
        for step, batch in enumerate(dataloader):
            # Dataset returns single tensor, so batch is already (B, W, D)
            if isinstance(batch, (list, tuple)):
                joints = batch[0].to(device, non_blocking=True)
            else:
                joints = batch.to(device, non_blocking=True)
            
            # Data validation
            if torch.isnan(joints).any() or torch.isinf(joints).any():
                print(f"ERROR: Invalid data detected at epoch {epoch}, step {step}")
                print(f"  NaN count: {torch.isnan(joints).sum().item()}")
                print(f"  Inf count: {torch.isinf(joints).sum().item()}")
                print(f"  Joints range: [{joints.min().item():.4f}, {joints.max().item():.4f}]")
                continue
            
            bsz, window, dim = joints.shape
            joints_flat = joints.reshape(-1, dim)

            recon, mu, logvar = model(joints_flat)
            
            # Check for NaN in model outputs
            if torch.isnan(recon).any() or torch.isnan(mu).any() or torch.isnan(logvar).any():
                print(f"ERROR: NaN in model outputs at epoch {epoch}, step {step}")
                print(f"  Recon NaN: {torch.isnan(recon).sum().item()}")
                print(f"  Mu NaN: {torch.isnan(mu).sum().item()}")
                print(f"  Logvar NaN: {torch.isnan(logvar).sum().item()}")
                print(f"  Mu range: [{mu.min().item():.4f}, {mu.max().item():.4f}]")
                print(f"  Logvar range: [{logvar.min().item():.4f}, {logvar.max().item():.4f}]")
                continue

            recon_loss = mse(recon, joints_flat)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + cfg.beta_kl * kl_loss

            # Check for NaN in losses
            if torch.isnan(loss) or torch.isnan(recon_loss) or torch.isnan(kl_loss):
                print(f"ERROR: NaN loss at epoch {epoch}, step {step}")
                print(f"  Recon loss: {recon_loss.item():.6f}")
                print(f"  KL loss: {kl_loss.item():.6f}")
                print(f"  Total loss: {loss.item():.6f}")
                print(f"  Beta KL: {cfg.beta_kl}")
                continue

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Check for NaN gradients
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"ERROR: NaN gradient in {name} at epoch {epoch}, step {step}")
                    has_nan_grad = True
            if has_nan_grad:
                continue
            
            optimizer.step()

            running_loss += loss.item()
            running_recon += recon_loss.item()
            running_kl += kl_loss.item()
            step_count += 1
            
            # Log to WandB every N steps
            if cfg.get('use_wandb', False) and step % 100 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/recon_loss': recon_loss.item(),
                    'train/kl_loss': kl_loss.item(),
                    'train/step': epoch * len(dataloader) + step
                })

        if step_count == 0:
            print(f"WARNING: No valid steps in epoch {epoch}")
            nan_epochs += 1
            if nan_epochs >= max_nan_epochs:
                print(f"ERROR: Too many NaN epochs ({nan_epochs}), stopping training")
                break
            continue

        avg_loss = running_loss / step_count
        avg_recon = running_recon / step_count
        avg_kl = running_kl / step_count
        
        # Log to file
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch:3d}: loss={avg_loss:.6f}, recon={avg_recon:.6f}, kl={avg_kl:.6f}\n")
        
        pbar_epochs.set_postfix({
            'loss': f'{avg_loss:.6f}',
            'recon': f'{avg_recon:.6f}',
            'kl': f'{avg_kl:.6f}'
        })
        
        # Log to WandB at end of epoch
        if cfg.get('use_wandb', False):
            wandb.log({
                'epoch/loss': avg_loss,
                'epoch/recon_loss': avg_recon,
                'epoch/kl_loss': avg_kl,
                'epoch': epoch
            })

        # Check for NaN in average loss
        if torch.isnan(torch.tensor(avg_loss)):
            print(f"ERROR: NaN average loss at epoch {epoch}")
            nan_epochs += 1
            if nan_epochs >= max_nan_epochs:
                print(f"ERROR: Too many NaN epochs ({nan_epochs}), stopping training")
                break
            continue
        else:
            nan_epochs = 0  # Reset counter on successful epoch

        torch.save({'state_dict': model.state_dict()}, ckpt_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'state_dict': model.state_dict()}, os.path.join(ckpt_dir, 'vae_best.pth'))
            if cfg.get('use_wandb', False):
                wandb.log({'best_loss': best_loss})
    
    if cfg.get('use_wandb', False):
        wandb.finish()
    
    print(f"\nTraining complete. Log saved to: {log_file}")


if __name__ == '__main__':
    train()

