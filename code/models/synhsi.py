import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from vit_pytorch import ViT
from tqdm import tqdm
from utils import *


class Sampler:
    def __init__(self, device, mask_ind, emb_f, batch_size, channel, auto_regre_num, timesteps, sampler_type='ddpm', ddim_steps=50, **kwargs):
        self.device = device
        self.mask_ind = mask_ind
        self.emb_f = emb_f
        self.batch_size = batch_size
        self.channel = channel
        self.auto_regre_num = auto_regre_num
        self.timesteps = timesteps
        self.sampler_type = sampler_type.lower()
        self.ddim_steps = ddim_steps
        self.motion_len = kwargs.get('motion_len', None)
        self.scene_type = kwargs.get('scene_type', None)
        self.get_scheduler()
        self.vae = None

    def set_dataset_and_model(self, dataset, model):
        self.dataset = dataset
        if dataset.load_scene:
            self.grid = dataset.create_meshgrid(batch_size=self.batch_size).to(self.device)
        self.model = model
        nb_voxels = dataset.nb_voxels
        self.occ_idx = torch.arange(0, nb_voxels[1], 1).to(self.device)

    def attach_vae(self, vae):
        self.vae = vae
        self.latent_dim = vae.latent_dim

    def get_scheduler(self):
        betas = linear_beta_schedule(timesteps=self.timesteps)

        # define alphas
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

    def q_sample(self, x_start, t, noise):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


    def p_losses(self, x_start, lang_emb_start, mat, scene_flag, mask, t_motion, t_text, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir, is_loco, loss_type='huber'):
        
        # 1. Create noise for both modalities
        noise_motion = torch.randn_like(x_start)
        noise_motion[mask] = 0.
        
        noise_lang = torch.randn_like(lang_emb_start)

        # 2. Create noisy versions
        x_noisy = self.q_sample(x_start=x_start, t=t_motion, noise=noise_motion)
        lang_noisy = self.q_sample(x_start=lang_emb_start, t=t_text, noise=noise_lang)

        # 3. Get scene conditions
        if self.dataset.load_scene:
            # If using latent diffusion, decode to joint space for scene querying
            # If using vanilla diffusion, x_noisy is already in joint space
            if self.vae is not None:
                x_noisy_decoded = self.decode_latent(x_noisy)
            else:
                x_noisy_decoded = x_noisy
            with torch.no_grad():
                x_orig = transform_points(self.dataset.denormalize_torch(x_noisy_decoded), mat)
                mat_for_query = mat.clone()
                target_ind = self.mask_ind if self.mask_ind != -1 else 0
                mat_for_query[:, :3, 3] = x_orig[:, self.emb_f, target_ind * 3: target_ind * 3 + 3]
                mat_for_query[:, 1, 3] = 0
                query_points = transform_points(self.grid, mat_for_query)
                occ = self.dataset.get_occ_for_points(query_points, scene_flag)
                nb_voxels = self.dataset.nb_voxels
                occ = occ.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()

                if self.scene_type in ['plane_two', 'occ_two']:
                    mat_for_query_goal = mat.clone()
                    pelvis_goal_copy = pelvis_goal.clone()
                    pelvis_goal_copy[is_loco] = pelvis_goal_copy[is_loco] / (torch.norm(pelvis_goal_copy[is_loco], dim=-1, keepdim=True) + 1e-6) * 0.8
                    pelvis_goal_orig = transform_points(pelvis_goal_copy.unsqueeze(1), mat).squeeze(1)

                    mat_for_query_goal[need_pelvis_dir, :3, 3] = pelvis_goal_orig[need_pelvis_dir]
                    mat_for_query_goal[torch.logical_not(need_pelvis_dir), :3, 3] = mat_for_query[torch.logical_not(need_pelvis_dir), :3, 3].clone()
                    mat_for_query_goal[:, 1, 3] = 0.
                    query_points = transform_points(self.grid, mat_for_query_goal)
                    occ_goal = self.dataset.get_occ_for_points(query_points, scene_flag)
                    nb_voxels = self.dataset.nb_voxels
                    occ_goal = occ_goal.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()

                if self.scene_type == 'occ':
                    occ = occ.permute(0, 2, 1, 3)
                elif self.scene_type == 'plane':
                    occ = occ.permute(0, 1, 3, 2)
                    occ_cnt = occ * self.occ_idx
                    occ = torch.argmax(occ_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]
                elif self.scene_type == 'plane_two':
                    occ = occ.permute(0, 1, 3, 2)
                    occ_cnt = occ * self.occ_idx
                    occ = torch.argmax(occ_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]

                    occ_goal = occ_goal.permute(0, 1, 3, 2)
                    occ_goal_cnt = occ_goal * self.occ_idx
                    occ_goal = torch.argmax(occ_goal_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]
                    occ = torch.cat([occ, occ_goal], dim=1)
                elif self.scene_type == 'occ_two':
                    occ = occ.permute(0, 2, 1, 3)
                    occ_goal = occ_goal.permute(0, 2, 1, 3)
                    occ = torch.cat([occ, occ_goal], dim=1)
        else:
            occ = None

        # 4. Get model predictions
        pred_motion_noise, pred_lang_noise = self.model(
            x_noisy, lang_noisy, occ, 
            t_motion, t_text, 
            pelvis_goal, hand_goal, is_pick, 
            need_scene, need_pelvis_dir
        )

        # 5. Calculate losses
        mask_inv = torch.logical_not(mask)

        if loss_type == 'l1':
            loss_motion = F.l1_loss(noise_motion[mask_inv], pred_motion_noise[mask_inv])
            loss_lang = F.l1_loss(noise_lang, pred_lang_noise)
        elif loss_type == 'l2':
            loss_motion = F.mse_loss(noise_motion[mask_inv], pred_motion_noise[mask_inv])
            loss_lang = F.mse_loss(noise_lang, pred_lang_noise)
        elif loss_type == "huber":
            loss_motion = F.smooth_l1_loss(noise_motion[mask_inv], pred_motion_noise[mask_inv])
            loss_lang = F.smooth_l1_loss(noise_lang, pred_lang_noise)
        else:
            raise NotImplementedError()

        return loss_motion, loss_lang

    @torch.no_grad()
    def p_sample_loop(self, mode, conditions, fixed_points, mat):
        """
        New sampling loop that supports different modes.
        
        :param mode: str, 't2m', 'm2t', 'joint'
        :param conditions: dict, contains all necessary clean conditions
        :param fixed_points: tensor, for auto-regression
        :param mat: tensor, for coordinate transformation
        """
        device = next(self.model.parameters()).device
        shape = (self.batch_size, self.dataset.max_window_size, self.channel)
        
        noisy_motion = torch.randn(shape, device=device)
        
        lang_shape = (self.batch_size, 1, self.model.out_lang.out_features) # (B, 1, D_lang)
        noisy_lang = torch.randn(lang_shape, device=device)

        # 2. Set timesteps based on mode and sampler type
        if self.sampler_type == 'ddim':
            # DDIM: use fewer steps, evenly spaced
            step_size = max(1, self.timesteps // self.ddim_steps)
            loop_timesteps = list(range(self.timesteps - 1, -1, -step_size))
            if loop_timesteps[-1] != 0:
                loop_timesteps.append(0)  # Always include t=0
            loop_timesteps = loop_timesteps[::-1]  # Reverse to go from high to low
        else:
            # DDPM: use all timesteps
            loop_timesteps = list(reversed(range(0, self.timesteps)))
        
        if mode == 't2m':
            t_text_val = 0 # Language is clean
        elif mode == 'joint':
            t_text_val = -1 # Use same timestep as motion
        else:
            raise NotImplementedError(f"Mode {mode} not implemented for sampling.")

        # 3. Apply auto-regressive fixed points
        if self.auto_regre_num > 0:
            self.set_fixed_points(noisy_motion, None, fixed_points, mat, joint_id=self.mask_ind, fix_mode=True, fix_goal=False)

        # Store intermediate results
        imgs = []
        occs = []
        
        # 4. Run the diffusion loop
        total_steps = len(loop_timesteps)
        for step_idx, i in enumerate(tqdm(loop_timesteps, desc=f'sampling loop ({mode}, {self.sampler_type})', total=total_steps)):
            t_motion = torch.full((self.batch_size,), i, device=device, dtype=torch.long)
            
            # Determine previous timestep
            if step_idx + 1 < len(loop_timesteps):
                t_prev_motion_val = loop_timesteps[step_idx + 1]
            else:
                t_prev_motion_val = -1  # Final step
            t_prev_motion = torch.full((self.batch_size,), t_prev_motion_val, device=device, dtype=torch.long)
            
            if t_text_val == 0:
                t_text = torch.full((self.batch_size,), 0, device=device, dtype=torch.long)
                t_prev_text = torch.full((self.batch_size,), -1, device=device, dtype=torch.long)
                lang_payload = conditions['text_emb']
            else:
                t_text = t_motion.clone()
                t_prev_text = t_prev_motion.clone()
                lang_payload = noisy_lang

            # Get denoised predictions for one step
            if self.sampler_type == 'ddim':
                noisy_motion, noisy_lang, occ = self.p_sample_ddim(
                    mode, noisy_motion, lang_payload, mat, 
                    t_motion, t_text, t_prev_motion, t_prev_text, conditions
                )
            else:
                noisy_motion, noisy_lang, occ = self.p_sample(
                    mode, noisy_motion, lang_payload, mat, t_motion, t_text, conditions
                )
            
            if self.auto_regre_num > 0:
                self.set_fixed_points(noisy_motion, None, fixed_points, mat, joint_id=self.mask_ind, fix_mode=True, fix_goal=False)

            imgs.append(noisy_motion)
            if occ is not None:
                occs.append(occ.cpu().numpy())

        return imgs, occs, noisy_lang

    @torch.no_grad()
    def p_sample(self, mode, x_motion, x_lang, mat, t_motion, t_text, conditions):
        
        # 1. Extract conditions
        scene_flag = conditions['scene_flag']
        pelvis_goal = conditions['pelvis_goal']
        hand_goal = conditions['hand_goal']
        is_pick = conditions['is_pick']
        need_scene = conditions['need_scene']
        need_pelvis_dir = conditions['need_pelvis_dir']
        is_loco = conditions.get('is_loco', torch.zeros_like(need_pelvis_dir))
        
        # Ensure boolean masks are 1D for indexing
        is_pick = is_pick.squeeze() if is_pick.dim() > 1 else is_pick
        need_scene = need_scene.squeeze() if need_scene.dim() > 1 else need_scene
        need_pelvis_dir = need_pelvis_dir.squeeze() if need_pelvis_dir.dim() > 1 else need_pelvis_dir
        is_loco = is_loco.squeeze() if is_loco.dim() > 1 else is_loco
        
        # 2. Get betas and alphas for denoising
        betas_t_motion = extract(self.betas, t_motion, x_motion.shape)
        sqrt_one_minus_alphas_cumprod_t_motion = extract(self.sqrt_one_minus_alphas_cumprod, t_motion, x_motion.shape)
        sqrt_recip_alphas_t_motion = extract(self.sqrt_recip_alphas, t_motion, x_motion.shape)
        
        betas_t_lang = extract(self.betas, t_text, x_lang.shape)
        sqrt_one_minus_alphas_cumprod_t_lang = extract(self.sqrt_one_minus_alphas_cumprod, t_text, x_lang.shape)
        sqrt_recip_alphas_t_lang = extract(self.sqrt_recip_alphas, t_text, x_lang.shape)

        # 3. Get scene occupancy grid
        if self.dataset.load_scene:
            # If using latent diffusion, decode to joint space for scene querying
            # If using vanilla diffusion, x_motion is already in joint space
            if self.vae is not None:
                x_motion_decoded = self.decode_latent(x_motion)
            else:
                x_motion_decoded = x_motion
            x_orig = transform_points(self.dataset.denormalize_torch(x_motion_decoded), mat)
            mat_for_query = mat.clone()
            target_ind = self.mask_ind if self.mask_ind != -1 else 0
            mat_for_query[:, :3, 3] = x_orig[:, self.emb_f, target_ind * 3: target_ind * 3 + 3]
            mat_for_query[:, 1, 3] = 0
            query_points = transform_points(self.grid, mat_for_query)
            occ = self.dataset.get_occ_for_points(query_points, scene_flag)
            nb_voxels = self.dataset.nb_voxels
            occ = occ.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()

            if self.scene_type in ['plane_two', 'occ_two']:
                mat_for_query_goal = mat.clone()
                pelvis_goal_copy = pelvis_goal.clone()
                pelvis_goal_copy[is_loco] = pelvis_goal_copy[is_loco] / (
                            torch.norm(pelvis_goal_copy[is_loco], dim=-1, keepdim=True) + 1e-6) * 0.8
                pelvis_goal_orig = transform_points(pelvis_goal_copy, mat)

                mat_for_query_goal[need_pelvis_dir, :3, 3] = pelvis_goal_orig[need_pelvis_dir].squeeze(1)
                mat_for_query_goal[torch.logical_not(need_pelvis_dir), :3, 3] = mat_for_query[
                                                                                torch.logical_not(need_pelvis_dir), :3,
                                                                                3].clone()
                mat_for_query_goal[:, 1, 3] = 0.
                query_points_goal = transform_points(self.grid, mat_for_query_goal)
                occ_goal = self.dataset.get_occ_for_points(query_points_goal, scene_flag)
                nb_voxels = self.dataset.nb_voxels
                occ_goal = occ_goal.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()

            if self.scene_type == 'occ':
                occ = occ.permute(0, 2, 1, 3)
            elif self.scene_type == 'occ_two':
                occ = occ.permute(0, 2, 1, 3)
                occ_goal = occ_goal.permute(0, 2, 1, 3)
                occ = torch.cat([occ, occ_goal], dim=1)
        else:
            occ = None

        # 4. Call the model
        pred_motion_noise, pred_lang_noise = self.model(
            x_motion, x_lang, occ, 
            t_motion, t_text, 
            pelvis_goal, hand_goal, is_pick, 
            need_scene, need_pelvis_dir
        )
        
        # 5. Denoise based on mode
        denoised_motion = sqrt_recip_alphas_t_motion * (
                x_motion - betas_t_motion * pred_motion_noise / sqrt_one_minus_alphas_cumprod_t_motion
        )
        
        if mode == 'joint':
            denoised_lang = sqrt_recip_alphas_t_lang * (
                    x_lang - betas_t_lang * pred_lang_noise / sqrt_one_minus_alphas_cumprod_t_lang
            )
        else:
            denoised_lang = x_lang
            
        # 6. Apply noise for next step (if not at t=0)
        
        final_denoised_motion = denoised_motion
        if t_motion[0].item() != 0:
            posterior_variance_t_motion = extract(self.posterior_variance, t_motion, x_motion.shape)
            final_denoised_motion = denoised_motion + torch.sqrt(posterior_variance_t_motion) * torch.randn_like(x_motion)

        final_denoised_lang = denoised_lang
        if mode == 'joint' and t_text[0].item() != 0:
            posterior_variance_t_lang = extract(self.posterior_variance, t_text, x_lang.shape)
            final_denoised_lang = denoised_lang + torch.sqrt(posterior_variance_t_lang) * torch.randn_like(x_lang)

        return final_denoised_motion, final_denoised_lang, occ

    @torch.no_grad()
    def p_sample_ddim(self, mode, x_motion, x_lang, mat, t_motion, t_text, t_prev_motion, t_prev_text, conditions):
        """
        DDIM sampling step - deterministic, allows fewer steps.
        
        :param t_prev_motion: previous timestep for motion (can skip steps)
        :param t_prev_text: previous timestep for language
        """
        # 1. Extract conditions (same as DDPM)
        scene_flag = conditions['scene_flag']
        pelvis_goal = conditions['pelvis_goal']
        hand_goal = conditions['hand_goal']
        is_pick = conditions['is_pick']
        need_scene = conditions['need_scene']
        need_pelvis_dir = conditions['need_pelvis_dir']
        is_loco = conditions.get('is_loco', torch.zeros_like(need_pelvis_dir))
        
        # Ensure boolean masks are 1D for indexing
        is_pick = is_pick.squeeze() if is_pick.dim() > 1 else is_pick
        need_scene = need_scene.squeeze() if need_scene.dim() > 1 else need_scene
        need_pelvis_dir = need_pelvis_dir.squeeze() if need_pelvis_dir.dim() > 1 else need_pelvis_dir
        is_loco = is_loco.squeeze() if is_loco.dim() > 1 else is_loco
        
        # 2. Get scene occupancy grid (same as DDPM)
        if self.dataset.load_scene:
            # If using latent diffusion, decode to joint space for scene querying
            # If using vanilla diffusion, x_motion is already in joint space
            if self.vae is not None:
                x_motion_decoded = self.decode_latent(x_motion)
            else:
                x_motion_decoded = x_motion
            x_orig = transform_points(self.dataset.denormalize_torch(x_motion_decoded), mat)
            mat_for_query = mat.clone()
            target_ind = self.mask_ind if self.mask_ind != -1 else 0
            mat_for_query[:, :3, 3] = x_orig[:, self.emb_f, target_ind * 3: target_ind * 3 + 3]
            mat_for_query[:, 1, 3] = 0
            query_points = transform_points(self.grid, mat_for_query)
            occ = self.dataset.get_occ_for_points(query_points, scene_flag)
            nb_voxels = self.dataset.nb_voxels
            occ = occ.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()

            if self.scene_type in ['plane_two', 'occ_two']:
                mat_for_query_goal = mat.clone()
                pelvis_goal_copy = pelvis_goal.clone()
                pelvis_goal_copy[is_loco] = pelvis_goal_copy[is_loco] / (
                            torch.norm(pelvis_goal_copy[is_loco], dim=-1, keepdim=True) + 1e-6) * 0.8
                pelvis_goal_orig = transform_points(pelvis_goal_copy, mat)

                mat_for_query_goal[need_pelvis_dir, :3, 3] = pelvis_goal_orig[need_pelvis_dir].squeeze(1)
                mat_for_query_goal[torch.logical_not(need_pelvis_dir), :3, 3] = mat_for_query[
                                                                                torch.logical_not(need_pelvis_dir), :3,
                                                                                3].clone()
                mat_for_query_goal[:, 1, 3] = 0.
                query_points_goal = transform_points(self.grid, mat_for_query_goal)
                occ_goal = self.dataset.get_occ_for_points(query_points_goal, scene_flag)
                nb_voxels = self.dataset.nb_voxels
                occ_goal = occ_goal.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()

            if self.scene_type == 'occ':
                occ = occ.permute(0, 2, 1, 3)
            elif self.scene_type == 'occ_two':
                occ = occ.permute(0, 2, 1, 3)
                occ_goal = occ_goal.permute(0, 2, 1, 3)
                occ = torch.cat([occ, occ_goal], dim=1)
        else:
            occ = None

        # 3. Call the model
        pred_motion_noise, pred_lang_noise = self.model(
            x_motion, x_lang, occ, 
            t_motion, t_text, 
            pelvis_goal, hand_goal, is_pick, 
            need_scene, need_pelvis_dir
        )
        
        # 4. DDIM update: deterministic, no random noise
        # pred_x0 = (x_t - sqrt(1 - alpha_t) * pred_noise) / sqrt(alpha_t)
        # x_{t-1} = sqrt(alpha_{t-1}) * pred_x0 + sqrt(1 - alpha_{t-1}) * pred_noise
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t_motion, x_motion.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t_motion, x_motion.shape)
        
        # Predict x0 from x_t and predicted noise
        pred_x0_motion = (x_motion - sqrt_one_minus_alphas_cumprod_t * pred_motion_noise) / sqrt_alphas_cumprod_t
        
        # DDIM step: deterministic update
        if t_prev_motion[0].item() >= 0:
            sqrt_alphas_cumprod_prev = extract(self.sqrt_alphas_cumprod, t_prev_motion, x_motion.shape)
            sqrt_one_minus_alphas_cumprod_prev = extract(self.sqrt_one_minus_alphas_cumprod, t_prev_motion, x_motion.shape)
            denoised_motion = sqrt_alphas_cumprod_prev * pred_x0_motion + sqrt_one_minus_alphas_cumprod_prev * pred_motion_noise
        else:
            denoised_motion = pred_x0_motion
        
        # Language handling
        if mode == 'joint':
            sqrt_alphas_cumprod_t_lang = extract(self.sqrt_alphas_cumprod, t_text, x_lang.shape)
            sqrt_one_minus_alphas_cumprod_t_lang = extract(self.sqrt_one_minus_alphas_cumprod, t_text, x_lang.shape)
            pred_x0_lang = (x_lang - sqrt_one_minus_alphas_cumprod_t_lang * pred_lang_noise) / sqrt_alphas_cumprod_t_lang
            
            if t_prev_text[0].item() >= 0:
                sqrt_alphas_cumprod_prev_lang = extract(self.sqrt_alphas_cumprod, t_prev_text, x_lang.shape)
                sqrt_one_minus_alphas_cumprod_prev_lang = extract(self.sqrt_one_minus_alphas_cumprod, t_prev_text, x_lang.shape)
                denoised_lang = sqrt_alphas_cumprod_prev_lang * pred_x0_lang + sqrt_one_minus_alphas_cumprod_prev_lang * pred_lang_noise
            else:
                denoised_lang = pred_x0_lang
        else:
            denoised_lang = x_lang

        return denoised_motion, denoised_lang, occ

    def set_fixed_points(self, img, goal, fixed_points, mat, joint_id, fix_mode, fix_goal):
        '''
        set fixed points of goal and prefix frames

        img: [b, max_window_size, 3 * joint_num]
        fixed_points: [b, auto_regre_num, 3 * joint_num]

        '''

        if goal is not None and fix_goal:
            goal_len = goal.shape[1]
            goal = self.dataset.normalize_torch(transform_points(goal, torch.inverse(mat)))

            img[:, -goal_len:, joint_id * 3] = goal[:, :, 0]
            if joint_id != 0:
                img[:, -goal_len:, joint_id * 3 + 1] = goal[:, :, 1]
            img[:, -goal_len:, joint_id * 3 + 2] = goal[:, :, 2]

        if fixed_points is not None and fix_mode:
            img[:, :fixed_points.shape[1], :] = fixed_points

    @torch.no_grad()
    def decode_latent(self, latent):
        """Decode latent to joint space. If VAE is None, assume input is already in joint space."""
        if self.vae is None:
            return latent
        return self.vae.decode_sequence(latent)

    @torch.no_grad()
    def encode_motion(self, motion):
        """Encode motion to latent space. If VAE is None, return motion as-is (vanilla diffusion)."""
        if self.vae is None:
            return motion
        return self.vae.encode_sequence(motion, deterministic=True)
    
    @torch.no_grad()
    def decode_motion(self, motion_latents):
        """Decode motion latents to joint space. If VAE is None, return as-is (vanilla diffusion)."""
        if self.vae is None:
            return motion_latents
        bsz, window, dim_latent = motion_latents.shape
        latent_flat = motion_latents.reshape(-1, dim_latent)
        decoded_joints = self.vae.decode(latent_flat)
        return decoded_joints.reshape(bsz, window, self.channel)


class Unet(nn.Module):
    """
    UWM-style multi-modal diffusion backbone.

    Tokens:
      - motion tokens (diffused)
      - language token (diffused or clean depending on mode)

    Conditions (only used for FiLM, not as tokens):
      - timesteps (motion / language)
      - scene embedding
      - hand / pelvis goal embeddings
      - simple binary flags (is_pick, need_scene, need_pelvis_dir)

    All conditions only modulate the transformer blocks through
    FiLM-style adaptive LayerNorm and gates, inspired by UWM.
    """

    def __init__(
            self,
            dim_model,
            num_heads,
            num_layers,
            dropout_p,
            dim_input,
            dim_output,
            nb_voxels=None,
            free_p=0.1,
            load_scene=True,
            load_language=True,
            load_hand_goal=True,
            load_pelvis_goal=True,
            language_feature_dim=768,
            scene_type=None,
            **kwargs
    ):
        super().__init__()

        self.model_type = "TransformerEncoder"
        self.dim_model = dim_model
        self.load_scene = load_scene
        self.load_language = load_language
        self.load_hand_goal = load_hand_goal
        self.load_pelvis_goal = load_pelvis_goal
        self.scene_type = scene_type

        # Scene encoder (same ViT as before, but used only for FiLM conditioning)
        if self.scene_type == 'plane':
            vit_channels = 1
        elif self.scene_type == 'occ':
            vit_channels = nb_voxels[1]
        elif self.scene_type == 'plane_two':
            vit_channels = 2
        elif self.scene_type == 'occ_two':
            vit_channels = 2 * nb_voxels[1]
        else:
            vit_channels = 1

        if self.load_scene:
            self.scene_embedding = ViT(
                image_size=nb_voxels[0],
                patch_size=8,
                channels=vit_channels,
                num_classes=dim_model,
                dim=512,
                depth=6,
                heads=16,
                mlp_dim=1024,
                dropout=0.1,
                emb_dropout=0.1
            )

        self.free_p = free_p

        # Positional encoding over the joint token sequence
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )

        # Input projections
        self.embedding_input = nn.Linear(dim_input, dim_model)

        if self.load_language:
            self.embedding_language_input = nn.Linear(language_feature_dim, dim_model)

        if self.load_hand_goal:
            self.embedding_hand_goal = GoalEncoder(mode='hand', dim_output=dim_model)

        if self.load_pelvis_goal:
            self.embedding_pelvis_goal = GoalEncoder(mode='pelvis', dim_output=dim_model)

        # Timestep embedders (used only on FiLM side)
        self.embed_timestep = TimestepEmbedder(self.dim_model, self.positional_encoder)

        if self.load_language:
            self.embed_timestep_lang = TimestepEmbedder(self.dim_model, self.positional_encoder)

        # Conditioner MLP that builds a compact conditioning vector
        cond_in_dim = 0
        # motion time
        cond_in_dim += dim_model
        # language time
        cond_in_dim += dim_model
        # scene
        cond_in_dim += dim_model
        # hand and pelvis goals
        cond_in_dim += dim_model
        cond_in_dim += dim_model
        # simple binary flags (is_pick, need_scene, need_pelvis_dir)
        cond_in_dim += 3

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in_dim, dim_model * 2),
            nn.SiLU(inplace=False),
            nn.Linear(dim_model * 2, dim_model)
        )

        # Stack of UWM-style transformer blocks
        self.blocks = nn.ModuleList([
            UWMBlock(dim_model=dim_model,
                     num_heads=num_heads,
                     dropout_p=dropout_p)
            for _ in range(num_layers)
        ])

        # Output heads
        self.out = nn.Linear(dim_model, dim_output)

        if self.load_language:
            self.out_lang = nn.Linear(dim_model, language_feature_dim)

    def _encode_scene(self, cond, need_scene):
        """
        Encode occupancy grid / scene into a single (B, D) vector.
        """
        if not self.load_scene or cond is None:
            return torch.zeros(need_scene.shape[0], self.dim_model, device=need_scene.device, dtype=torch.float32)

        scene_emb_raw = self.scene_embedding(cond)
        if scene_emb_raw.dim() == 2:
            scene_emb = scene_emb_raw
        else:
            B, D = scene_emb_raw.shape[0], scene_emb_raw.shape[-1]
            scene_emb = scene_emb_raw.reshape(B, -1, D).mean(dim=1)

        if scene_emb.shape[1] != self.dim_model:
            scene_emb = scene_emb.reshape(scene_emb.shape[0], -1)[:, :self.dim_model]

        need_scene = need_scene.squeeze() if need_scene.dim() > 1 else need_scene
        scene_emb[torch.logical_not(need_scene)] = 0.0
        return scene_emb

    def _encode_goals(self, pelvis_goal, hand_goal, is_pick, need_pelvis_dir):
        B = pelvis_goal.shape[0]
        device = pelvis_goal.device

        # Ensure boolean masks are 1D
        is_pick = is_pick.squeeze() if is_pick.dim() > 1 else is_pick
        need_pelvis_dir = need_pelvis_dir.squeeze() if need_pelvis_dir.dim() > 1 else need_pelvis_dir

        if self.load_hand_goal and hand_goal is not None:
            hand_emb = self.embedding_hand_goal(hand_goal)
            hand_emb = hand_emb.squeeze(1) if hand_emb.dim() == 3 else hand_emb
            hand_emb[torch.logical_not(is_pick)] = 0.0
        else:
            hand_emb = torch.zeros(B, self.dim_model, device=device, dtype=torch.float32)

        if self.load_pelvis_goal and pelvis_goal is not None:
            pelvis_emb = self.embedding_pelvis_goal(pelvis_goal)
            pelvis_emb = pelvis_emb.squeeze(1) if pelvis_emb.dim() == 3 else pelvis_emb
            pelvis_emb[torch.logical_not(need_pelvis_dir)] = 0.0
        else:
            pelvis_emb = torch.zeros(B, self.dim_model, device=device, dtype=torch.float32)

        return pelvis_emb, hand_emb

    def _build_condition_vector(
            self,
            t_motion_emb,
            t_lang_emb,
            scene_emb,
            pelvis_emb,
            hand_emb,
            is_pick,
            need_scene,
            need_pelvis_dir,
            batch_size=None
    ):
        # Ensure boolean masks are 1D
        is_pick = is_pick.squeeze() if is_pick.dim() > 1 else is_pick
        need_scene = need_scene.squeeze() if need_scene.dim() > 1 else need_scene
        need_pelvis_dir = need_pelvis_dir.squeeze() if need_pelvis_dir.dim() > 1 else need_pelvis_dir
        
        flags = torch.stack([is_pick.float(), need_scene.float(), need_pelvis_dir.float()], dim=-1)
        
        if batch_size is None:
            batch_size = flags.shape[0]
        
        # Ensure all embeddings are (B, D)
        def ensure_2d(tensor, B):
            if tensor.dim() == 1:
                return tensor.unsqueeze(-1) if tensor.shape[0] == B else tensor.unsqueeze(0).expand(B, -1)
            elif tensor.dim() == 2:
                return tensor.t() if tensor.shape[1] == B else tensor
            elif tensor.dim() > 2:
                return tensor.view(B, -1)
            return tensor
        
        t_motion_emb = ensure_2d(t_motion_emb, batch_size)
        t_lang_emb = ensure_2d(t_lang_emb, batch_size)
        scene_emb = ensure_2d(scene_emb, batch_size)
        pelvis_emb = ensure_2d(pelvis_emb, batch_size)
        hand_emb = ensure_2d(hand_emb, batch_size)
        
        cond_concat = torch.cat([t_motion_emb, t_lang_emb, scene_emb, pelvis_emb, hand_emb, flags], dim=-1)
        cond_vec = self.cond_mlp(cond_concat)
        return cond_vec.view(batch_size, -1)

    def forward(
            self,
            x_motion,
            noisy_lang_emb,
            cond,
            t_motion,
            t_text,
            pelvis_goal,
            hand_goal,
            is_pick,
            need_scene,
            need_pelvis_dir
    ):
        """
        x_motion: (B, T, dim_input)
        noisy_lang_emb: (B, 1, language_feature_dim)
        cond: scene occupancy grid (shape handled by ViT)
        t_motion, t_text: (B,)
        """
        B, T, _ = x_motion.shape

        # --- 1. Timestep embeddings (B, D) ---
        t_motion_emb = self.embed_timestep(t_motion)
        t_lang_emb = self.embed_timestep_lang(t_text)

        # --- 2. Scene and goals encoded as (B, D) ---
        scene_emb = self._encode_scene(cond, need_scene)
        pelvis_emb, hand_emb = self._encode_goals(
            pelvis_goal, hand_goal, is_pick, need_pelvis_dir
        )

        # --- 3. Build conditioning vector for FiLM / gates ---
        cond_vec = self._build_condition_vector(
            t_motion_emb,
            t_lang_emb,
            scene_emb,
            pelvis_emb,
            hand_emb,
            is_pick,
            need_scene,
            need_pelvis_dir,
            batch_size=B,
        )  # (B, D_model)

        # --- 4. Build token sequence (motion + language) ---
        # Motion tokens: (T, B, D)
        x_motion_tokens = self.embedding_input(x_motion.permute(1, 0, 2))

        # Language token: (1, B, D)
        if self.load_language and noisy_lang_emb is not None:
            lang_token = self.embedding_language_input(noisy_lang_emb)  # (B, 1, D)
            lang_token = lang_token.permute(1, 0, 2)  # (1, B, D)
        else:
            lang_token = torch.zeros(
                1, B, self.dim_model, device=x_motion.device, dtype=x_motion.dtype
            )

        tokens = torch.cat([x_motion_tokens, lang_token], dim=0)  # (T+1, B, D)

        # Positional encoding over the full sequence
        tokens = self.positional_encoder(tokens)

        # --- 5. Transformer stack with FiLM conditioning ---
        for block in self.blocks:
            tokens = block(tokens, cond_vec)

        # --- 6. Prediction heads ---
        # Motion noise: first T positions
        motion_tokens_out = tokens[:T]  # (T, B, D)
        pred_motion_noise = self.out(motion_tokens_out).permute(1, 0, 2)

        # Language noise: position T
        lang_token_out = tokens[T:T + 1]  # (1, B, D)
        pred_lang_noise = self.out_lang(lang_token_out).permute(1, 0, 2)

        return pred_motion_noise, pred_lang_noise


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).reshape(-1, 1)  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(inplace=False),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        # timesteps: (B,)
        pos_emb = self.sequence_pos_encoder.pos_encoding[timesteps]  # (B, 1, D)
        emb = self.time_embed(pos_emb)  # (B, 1, D)
        return emb.squeeze(1) if emb.dim() == 3 else emb.view(emb.shape[0], -1) if emb.dim() > 3 else emb


class ProgressIndicatorEmbedding(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

    def forward(self, timesteps):
        return self.sequence_pos_encoder.pos_encoding[timesteps]


class AdaptiveLayerNorm(nn.Module):
    """
    LayerNorm with FiLM-style conditioning.
    The scale and shift are produced from a conditioning vector.
    """

    def __init__(self, dim_model, cond_dim=None):
        super().__init__()
        self.dim_model = dim_model
        self.norm = nn.LayerNorm(dim_model)
        if cond_dim is None:
            cond_dim = dim_model
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, dim_model * 2),
            nn.SiLU(inplace=False),
            nn.Linear(dim_model * 2, dim_model * 2),
        )

    def forward(self, x, cond_vec):
        """
        x: (S, B, D)
        cond_vec: (B, C)
        """
        B = x.shape[1]
        cond = self.cond_mlp(cond_vec)  # (B, 2D)
        gamma, beta = torch.chunk(cond, 2, dim=-1)  # (B, D) each
        gamma = gamma.unsqueeze(0)  # (1, B, D)
        beta = beta.unsqueeze(0)    # (1, B, D)
        x_norm = self.norm(x)
        return gamma * x_norm + beta


class UWMBlock(nn.Module):
    """
    Simplified UWM-style transformer block:
      - self-attention with FiLM-conditioned LayerNorm and gate
      - feed-forward with FiLM-conditioned LayerNorm and gate
    """

    def __init__(self, dim_model, num_heads, dropout_p):
        super().__init__()
        self.dim_model = dim_model

        self.attn = nn.MultiheadAttention(
            embed_dim=dim_model,
            num_heads=num_heads,
            dropout=dropout_p,
            batch_first=False,
        )
        self.ff = nn.Sequential(
            nn.Linear(dim_model, dim_model * 4),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(dim_model * 4, dim_model),
            nn.Dropout(dropout_p),
        )

        self.adapt_ln_attn = AdaptiveLayerNorm(dim_model)
        self.adapt_ln_ff = AdaptiveLayerNorm(dim_model)

        # Scalar gates for attention and feed-forward
        self.gate_mlp_attn = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.SiLU(inplace=False),
            nn.Linear(dim_model, 1),
        )
        self.gate_mlp_ff = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.SiLU(inplace=False),
            nn.Linear(dim_model, 1),
        )

    def forward(self, x, cond_vec):
        """
        x: (S, B, D)
        cond_vec: (B, D_cond) -> used for FiLM and gates
        """
        B = x.shape[1]
        
        # Ensure cond_vec is (B, D)
        if cond_vec.dim() == 1:
            cond_vec = cond_vec.unsqueeze(0).expand(B, -1)
        elif cond_vec.dim() == 2:
            if cond_vec.shape[1] == B:
                cond_vec = cond_vec.t()
        elif cond_vec.dim() > 2:
            cond_vec = cond_vec.view(B, -1)
        
        # Self-attention
        x_norm = self.adapt_ln_attn(x, cond_vec)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        
        gate_attn = torch.sigmoid(self.gate_mlp_attn(cond_vec))  # (B, 1)
        if gate_attn.dim() == 1:
            gate_attn = gate_attn.unsqueeze(-1)
        elif gate_attn.dim() > 2:
            gate_attn = gate_attn.view(B, -1)[:, :1]
        elif gate_attn.shape[1] > 1:
            gate_attn = gate_attn[:, :1]
        x = x + gate_attn.unsqueeze(0) * attn_out

        # Feed-forward
        x_norm2 = self.adapt_ln_ff(x, cond_vec)
        ff_out = self.ff(x_norm2)
        
        gate_ff = torch.sigmoid(self.gate_mlp_ff(cond_vec))  # (B, 1)
        if gate_ff.dim() == 1:
            gate_ff = gate_ff.unsqueeze(-1)
        elif gate_ff.dim() > 2:
            gate_ff = gate_ff.view(B, -1)[:, :1]
        elif gate_ff.shape[1] > 1:
            gate_ff = gate_ff[:, :1]
        x = x + gate_ff.unsqueeze(0) * ff_out

        return x

class ActionTransformerEncoder(nn.Module):
    def __init__(self,
                 action_number,
                 dim_model,
                 nhead,
                 num_layers,
                 dim_feedforward,
                 dropout_p,
                 activation="gelu") -> None:
        super().__init__()
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.input_embedder = nn.Linear(action_number, dim_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model,
                                                    nhead=nhead,
                                                    dim_feedforward=dim_feedforward,
                                                    dropout_p=dropout_p,
                                                    activation=activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers
        )

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = self.input_embedder(x)
        x = self.positional_encoder(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        x = torch.mean(x, dim=1, keepdim=True)
        return x
    

class LanguageEncoder(nn.Module):
    def __init__(self, dim_output, dim_input, **kwargs):
        super().__init__()
        self.dim_model = dim_output

        self.embedding_input1 = nn.Sequential(
            nn.Linear(dim_input, dim_output),
            nn.SiLU(inplace=False),
            nn.Linear(dim_output, dim_output),
        )

        self.embedding_input2 = nn.Sequential(
            nn.Linear(dim_output, dim_output),
            nn.SiLU(inplace=False),
            nn.Linear(dim_output, dim_output),
        )

        self.positional_encoder = PositionalEncoding(
            dim_model=dim_output, dropout_p=0.1, max_len=5000
        )

        self.embed_pi = ProgressIndicatorEmbedding(dim_output, self.positional_encoder)

    def forward(self, x, pi=None, need_pi=None):
        # x.shape: [b, 1, 768]

        x = self.embedding_input1(x)
        
        if pi is not None and need_pi is not None:
            pi_emb = self.embed_pi(pi)
            pi_emb = pi_emb / np.sqrt(self.dim_model // 2)
            not_need_pi = torch.logical_not(need_pi)
            pi_emb[not_need_pi] = 0.
            x = x + pi_emb
        
        x = self.embedding_input2(x)
        return x

class GoalEncoder(nn.Module):
    def __init__(self, mode, dim_output, **kwargs):
        super().__init__()

        self.mode = mode
        if mode == 'pelvis':
            self.embedding_input = nn.Sequential(nn.Linear(2, dim_output),
                                                    nn.SiLU(inplace=False),
                                                    nn.Linear(dim_output, dim_output))
        elif mode == 'hand':
            self.embedding_input = nn.Sequential(nn.Linear(3, dim_output),
                                                    nn.SiLU(inplace=False),
                                                    nn.Linear(dim_output, dim_output))

    def forward(self, x):
        # Flatten to (B, -1)
        B = x.shape[0]
        x = x.reshape(B, -1)
        
        # Extract relevant coordinates
        if self.mode == 'pelvis':
            x = x[:, [0, 2]] if x.shape[1] >= 3 else x[:, :2]
        else:  # hand
            x = x[:, :3] if x.shape[1] >= 3 else x
        
        x = self.embedding_input(x)  # (B, D_model)
        if x.dim() == 2:
            return x.unsqueeze(1)
        elif x.dim() == 3:
            return x.mean(dim=1, keepdim=True) if x.shape[1] != 1 else x
        return x