import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def prepare_latent_cache(dataset,
                         vae,
                         cache_path,
                         batch_size,
                         device,
                         num_workers=0):
    if os.path.exists(cache_path):
        os.remove(cache_path)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    total_len = len(dataset)
    latent_dim = vae.latent_dim
    window = dataset.max_window_size
    lat_memmap = np.lib.format.open_memmap(
        cache_path,
        mode='w+',
        dtype=np.float16,
        shape=(total_len, window, latent_dim)
    )

    offset = 0
    vae.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc='Encoding latents'):
            joints = batch[0].to(device)
            latents = vae.encode_sequence(joints, deterministic=True)
            bsz = latents.shape[0]
            lat_memmap[offset:offset + bsz] = latents.cpu().numpy().astype(np.float16)
            offset += bsz

    lat_memmap.flush()
    return cache_path

