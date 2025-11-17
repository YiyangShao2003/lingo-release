import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle as pkl


class LingoVAEDataset(Dataset):
    """
    Lightweight dataset for VAE training.
    Only loads and returns normalized joint sequences; it intentionally
    skips language, scene, and goal preprocessing to keep CPU load low.
    """

    def __init__(
        self,
        folder,
        step,
        max_window_size,
        **kwargs,
    ):
        self.folder = folder
        self.step = step
        self.max_window_size = max_window_size

        # Load joints and normalization stats
        self.joints = np.load(os.path.join(folder, "human_joints_aligned.npy"))  # (N, 28, 3)

        if self.max_window_size == 16:
            language_motion_dict_filename = "language_motion_dict__inter_and_loco__16.pkl"
        else:
            raise ValueError(f"Unsupported max_window_size: {self.max_window_size}")

        with open(
            os.path.join(self.folder, "language_motion_dict", language_motion_dict_filename),
            "rb",
        ) as f:
            language_motion_dict = pkl.load(f)

        # Segment boundaries
        self.start_ind = language_motion_dict["start_idx"]
        self.end_ind = language_motion_dict["end_idx"]

        # Normalization range (same as main dataset)
        norm = np.load(os.path.join(folder, "norm_inter_and_loco__16frames.npy"))
        self.min = norm[0].astype(np.float32)
        self.max = norm[1].astype(np.float32)

    def __len__(self):
        return len(self.start_ind)

    def __getitem__(self, idx):
        start_idx = int(self.start_ind[idx])
        end_idx = int(self.end_ind[idx])

        # Expect 3 * max_window_size frames between start and end (same as main dataset)
        assert end_idx - start_idx == self.max_window_size * 3

        joints = self.joints[start_idx:end_idx:self.step]  # (W, 28, 3)
        joints = self.normalize(joints)
        joints = joints.astype(np.float32).reshape(joints.shape[0], -1)  # (W, 84)
        
        # Convert to torch tensor
        return torch.from_numpy(joints)

    def normalize(self, data):
        shape_orig = data.shape
        data = data.reshape((-1, 3))
        data = -1.0 + 2.0 * (data - self.min) / (self.max - self.min)
        data = data.reshape(shape_orig)
        return data


