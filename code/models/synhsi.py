import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from vit_pytorch import ViT
from tqdm import tqdm
from utils import *


class Sampler:
    def __init__(self, device, mask_ind, emb_f, batch_size, channel, auto_regre_num, timesteps, **kwargs):
        self.device = device
        self.mask_ind = mask_ind
        self.emb_f = emb_f
        self.batch_size = batch_size
        self.channel = channel
        self.auto_regre_num = auto_regre_num
        self.timesteps = timesteps
        self.motion_len = kwargs.get('motion_len', None)
        self.scene_type = kwargs.get('scene_type', None)
        self.get_scheduler()

    def set_dataset_and_model(self, dataset, model):
        self.dataset = dataset
        if dataset.load_scene:
            self.grid = dataset.create_meshgrid(batch_size=self.batch_size).to(self.device)
        self.model = model
        nb_voxels = dataset.nb_voxels
        self.occ_idx = torch.arange(0, nb_voxels[1], 1).to(self.device)

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
            with torch.no_grad():
                x_orig = transform_points(self.dataset.denormalize_torch(x_noisy), mat)
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

        # 2. Set timesteps based on mode
        if mode == 't2m':
            t_text_val = 0 # Language is clean
            # We only need to denoise motion
            loop_timesteps = reversed(range(0, self.timesteps))
        elif mode == 'joint':
            t_text_val = -1 # Use same timestep as motion
            loop_timesteps = reversed(range(0, self.timesteps))
        else:
            # m2t (motion-to-text) and unconditional modes
            # would require different logic here.
            # For this example, we'll focus on t2m and joint.
            raise NotImplementedError(f"Mode {mode} not implemented for sampling.")

        # 3. Apply auto-regressive fixed points
        if self.auto_regre_num > 0:
            self.set_fixed_points(noisy_motion, None, fixed_points, mat, joint_id=self.mask_ind, fix_mode=True, fix_goal=False)

        # Store intermediate results
        imgs = []
        occs = []
        
        # 4. Run the diffusion loop
        for i in tqdm(loop_timesteps, desc=f'sampling loop ({mode})', total=self.timesteps):
            t_motion = torch.full((self.batch_size,), i, device=device, dtype=torch.long)
            
            if t_text_val == 0:
                t_text = torch.full((self.batch_size,), 0, device=device, dtype=torch.long)
                lang_payload = conditions['text_emb']
            else:
                t_text = t_motion.clone()
                lang_payload = noisy_lang

            # Get denoised predictions for one step
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
        
        # 2. Get betas and alphas for denoising
        betas_t_motion = extract(self.betas, t_motion, x_motion.shape)
        sqrt_one_minus_alphas_cumprod_t_motion = extract(self.sqrt_one_minus_alphas_cumprod, t_motion, x_motion.shape)
        sqrt_recip_alphas_t_motion = extract(self.sqrt_recip_alphas, t_motion, x_motion.shape)
        
        betas_t_lang = extract(self.betas, t_text, x_lang.shape)
        sqrt_one_minus_alphas_cumprod_t_lang = extract(self.sqrt_one_minus_alphas_cumprod, t_text, x_lang.shape)
        sqrt_recip_alphas_t_lang = extract(self.sqrt_recip_alphas, t_text, x_lang.shape)

        # 3. Get scene occupancy grid
        if self.dataset.load_scene:
            x_orig = transform_points(self.dataset.denormalize_torch(x_motion), mat)
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


class Unet(nn.Module):
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

        if self.scene_type == 'plane':
            vit_channels = 1
        elif self.scene_type == 'occ':
            vit_channels = nb_voxels[1]
        elif self.scene_type == 'plane_two':
            vit_channels = 2
        elif self.scene_type == 'occ_two':
            vit_channels = 2*nb_voxels[1]

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
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.embedding_input = nn.Linear(dim_input, dim_model)
        
        if self.load_language:
            self.embedding_language_input = nn.Linear(language_feature_dim, dim_model)

        if self.load_hand_goal:
            self.embedding_hand_goal = GoalEncoder(mode='hand', dim_output=dim_model)

        if self.load_pelvis_goal:
            self.embedding_pelvis_goal = GoalEncoder(mode='pelvis', dim_output=dim_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model,
                                                   nhead=num_heads,
                                                   dim_feedforward=dim_model,
                                                   dropout=dropout_p,
                                                   activation="gelu")

        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers
        )

        self.out = nn.Linear(dim_model, dim_output)

        if self.load_language:
            self.out_lang = nn.Linear(dim_model, language_feature_dim)

        self.embed_timestep = TimestepEmbedder(self.dim_model, self.positional_encoder)
        
        if self.load_language:
            self.embed_timestep_lang = TimestepEmbedder(self.dim_model, self.positional_encoder)

    def forward(self, x_motion, noisy_lang_emb, cond, t_motion, t_text, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir):
        t_motion_emb = self.embed_timestep(t_motion)
        t_lang_emb = self.embed_timestep_lang(t_text)

        # Embed Context Conditions
        if not self.load_scene:
            scene_emb = torch.zeros_like(t_motion_emb)
        else:
            scene_emb = self.scene_embedding(cond).reshape(-1, 1, self.dim_model)
            not_need_scene = torch.logical_not(need_scene)
            scene_emb[not_need_scene] = 0.
        
        if not self.load_hand_goal:
            hand_goal_emb = torch.zeros_like(t_motion_emb)
        else:
            hand_goal_emb = self.embedding_hand_goal(hand_goal)
            is_not_pick = torch.logical_not(is_pick)
            hand_goal_emb[is_not_pick] = 0.

        if not self.load_pelvis_goal:
            pelvis_goal_emb = torch.zeros_like(t_motion_emb)
        else:
            pelvis_goal_emb = self.embedding_pelvis_goal(pelvis_goal)
            not_need_pelvis_dir = torch.logical_not(need_pelvis_dir)
            pelvis_goal_emb[not_need_pelvis_dir] = 0.

        scene_emb = t_motion_emb + scene_emb
        hand_goal_emb = t_motion_emb + hand_goal_emb
        pelvis_goal_emb = t_motion_emb + pelvis_goal_emb
        
        scene_emb = scene_emb.permute(1, 0, 2)
        hand_goal_emb = hand_goal_emb.permute(1, 0, 2)
        pelvis_goal_emb = pelvis_goal_emb.permute(1, 0, 2)

        x_motion = x_motion.permute(1, 0, 2)
        x_motion_tokens = self.embedding_input(x_motion) * math.sqrt(self.dim_model)
        
        lang_token = self.embedding_language_input(noisy_lang_emb) # (B, 1, D_model)
        lang_token = lang_token + t_lang_emb
        lang_token = lang_token.permute(1, 0, 2)

        x = torch.cat((scene_emb, hand_goal_emb, pelvis_goal_emb, lang_token, x_motion_tokens), dim=0)
        
        x = self.positional_encoder(x)
        x = self.transformer(x)

        lang_output_token = x[3:4]
        motion_output_tokens = x[4:]
        
        pred_motion_noise = self.out(motion_output_tokens)
        pred_motion_noise = pred_motion_noise.permute(1, 0, 2) # (B, W, D_output)

        pred_lang_noise = self.out_lang(lang_output_token)
        pred_lang_noise = pred_lang_noise.permute(1, 0, 2) # (B, 1, D_lang)

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
        return self.time_embed(self.sequence_pos_encoder.pos_encoding[timesteps])


class ProgressIndicatorEmbedding(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

    def forward(self, timesteps):
        return self.sequence_pos_encoder.pos_encoding[timesteps]


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
        # x.shape: [b, 3]
        if self.mode == 'pelvis':
            x = x[..., [0, 2]]
        x = self.embedding_input(x)
        x = x.reshape(-1, 1, x.shape[-1])
        return x