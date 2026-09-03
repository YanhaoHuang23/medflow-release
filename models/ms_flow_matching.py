import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_tokens(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


def _clone_state_dict(state: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if state is None:
        return None

    cloned: Dict[str, object] = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.detach().clone()
        else:
            cloned[key] = value
    return cloned


class KDEPrior:
    def __init__(
        self,
        u_samples: torch.Tensor,
        bandwidth_factor: float = 1.0,
        max_centers: int = 2000,
    ):
        if u_samples.ndim != 2:
            raise ValueError(f'Expected [N, D] latent samples, got shape={tuple(u_samples.shape)}')

        self.centers = self._select_centers(u_samples.detach().float(), max_centers)
        self.num_samples = self.centers.shape[0]
        self.dim = self.centers.shape[1]
        self.device = self.centers.device
        self.bandwidth = self._estimate_bandwidth(self.centers, bandwidth_factor)

    @staticmethod
    def _select_centers(u_samples: torch.Tensor, max_centers: int) -> torch.Tensor:
        if max_centers <= 0 or u_samples.shape[0] <= max_centers:
            return u_samples

        idx = torch.linspace(
            0,
            u_samples.shape[0] - 1,
            steps=max_centers,
            device=u_samples.device,
        ).round().long()
        idx = torch.unique(idx, sorted=True)
        return u_samples[idx]

    @staticmethod
    def _estimate_bandwidth(centers: torch.Tensor, bandwidth_factor: float) -> float:
        if centers.shape[0] < 2:
            return max(float(bandwidth_factor), 1e-4)

        subset = centers[: min(2000, centers.shape[0])]
        dists = torch.cdist(subset, subset)
        dists.fill_diagonal_(float('inf'))
        min_dists = dists.min(dim=1).values
        finite = min_dists[torch.isfinite(min_dists)]
        if finite.numel() == 0:
            return max(float(bandwidth_factor), 1e-4)
        avg_nn_dist = finite.mean().item()
        return max(avg_nn_dist * float(bandwidth_factor), 1e-4)

    def sample(
        self,
        n_samples: int,
        noise_scale: float = 1.0,
        center_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if center_indices is None:
            indices = torch.randint(0, self.num_samples, (n_samples,), device=self.device)
        else:
            if center_indices.ndim != 1 or center_indices.shape[0] != n_samples:
                raise ValueError(
                    f'Expected center_indices shape=({n_samples},), got {tuple(center_indices.shape)}'
                )
            indices = center_indices.to(device=self.device, dtype=torch.long)
            if torch.any(indices < 0) or torch.any(indices >= self.num_samples):
                raise ValueError('center_indices out of KDE center range')
        base = self.centers[indices]
        noise = torch.randn(n_samples, self.dim, device=self.device, dtype=self.centers.dtype)
        return base + noise * (self.bandwidth * float(noise_scale))


class ScalarTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() > 1:
            t = t.reshape(t.shape[0], -1)[:, 0]
        return self.net(t[:, None])


class DiTBlock1D(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ada_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6),
        )
        nn.init.zeros_(self.ada_mod[-1].weight)
        nn.init.zeros_(self.ada_mod[-1].bias)

    def _modulate(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + scale[:, None, :]) + shift[:, None, :]

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada_mod(cond_vec).chunk(6, dim=-1)

        x_norm = self._modulate(self.ln1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + gate_msa[:, None, :] * attn_out

        y = self._modulate(self.ln2(x), shift_mlp, scale_mlp)
        y = self.mlp(y)
        x = x + gate_mlp[:, None, :] * y
        return x


class ClassSpecificAdapter(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, bottleneck_dim: Optional[int] = None):
        super().__init__()
        self.num_classes = int(num_classes)
        bottleneck_dim = int(bottleneck_dim or max(hidden_dim // 4, 16))
        self.adapters = nn.ModuleList()
        for _ in range(self.num_classes):
            adapter = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, bottleneck_dim),
                nn.GELU(),
                nn.Linear(bottleneck_dim, hidden_dim),
            )
            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)
            self.adapters.append(adapter)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        if labels is None or self.num_classes <= 0:
            return x
        labels = labels.to(device=x.device, dtype=torch.long).view(-1)
        out = torch.zeros_like(x)
        valid_any = False
        for class_id in labels.unique(sorted=True):
            class_int = int(class_id.item())
            if class_int < 0 or class_int >= self.num_classes:
                continue
            mask = labels == class_int
            if mask.any():
                out[mask] = self.adapters[class_int](x[mask])
                valid_any = True
        if not valid_any:
            return x
        return x + out


class FlowCEBackbone1D(nn.Module):
    def __init__(
        self,
        num_codes: int,
        code_dim: int,
        hidden_dim: int,
        depth: int,
        n_heads: int,
        dropout: float,
        max_seq_len: int,
        backbone: str,
        num_classes: int = 0,
        use_context: bool = False,
        class_specific_output_head: bool = False,
        class_specific_adapter: bool = False,
        label_conditioning_mode: str = 'add',
        token_type_conditioning: str = 'none',
        scale_id: int = 0,
        num_scales: int = 1,
    ):
        super().__init__()
        self.num_codes = int(num_codes)
        self.backbone = backbone
        self.num_classes = int(num_classes)
        self.use_context = bool(use_context)
        self.class_specific_output_head = bool(class_specific_output_head and self.num_classes > 0)
        self.class_specific_adapter = bool(class_specific_adapter and self.num_classes > 0)
        self.label_conditioning_mode = label_conditioning_mode
        self.token_type_conditioning = token_type_conditioning
        self.scale_id = int(scale_id)
        self.num_scales = max(int(num_scales), 1)
        if self.label_conditioning_mode not in {'add', 'film'}:
            raise ValueError(f'Unsupported label_conditioning_mode: {self.label_conditioning_mode}')
        if self.token_type_conditioning not in {'none', 'class', 'class_scale'}:
            raise ValueError(f'Unsupported token_type_conditioning: {self.token_type_conditioning}')

        init_global = torch.randn(1, 1, code_dim)
        self.global_token = nn.Parameter(_normalize_tokens(init_global))

        self.in_proj = nn.Linear(code_dim, hidden_dim)
        self.time_embed = ScalarTimeEmbedding(hidden_dim)
        self.label_embed = nn.Embedding(self.num_classes, hidden_dim) if self.num_classes > 0 else None
        token_type_count = self.num_classes
        if self.token_type_conditioning == 'class_scale':
            token_type_count = self.num_classes * self.num_scales
        self.token_type_embed = (
            nn.Embedding(token_type_count, hidden_dim)
            if self.token_type_conditioning in {'class', 'class_scale'} and token_type_count > 0
            else None
        )
        self.label_film = nn.Linear(hidden_dim, hidden_dim * 2) if self.label_conditioning_mode == 'film' and self.num_classes > 0 else None
        if self.label_film is not None:
            nn.init.zeros_(self.label_film.weight)
            nn.init.zeros_(self.label_film.bias)
        self.context_proj = nn.Linear(code_dim, hidden_dim) if self.use_context else None
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len + 1, hidden_dim))

        if backbone == 'dit1d':
            self.blocks = nn.ModuleList([DiTBlock1D(hidden_dim, n_heads, dropout) for _ in range(depth)])
            self.encoder = None
        elif backbone == 'transformer1d':
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation='gelu',
            )
            self.blocks = None
            self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, self.num_codes)
        self.class_out_proj = nn.ModuleList(
            [nn.Linear(hidden_dim, self.num_codes) for _ in range(self.num_classes)]
        ) if self.class_specific_output_head else None
        self.class_adapters = nn.ModuleList(
            [ClassSpecificAdapter(hidden_dim, self.num_classes) for _ in range(depth)]
        ) if self.class_specific_adapter else None

    def _apply_label_conditioning(self, time_vec: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        if self.label_embed is None:
            return time_vec
        label_vec = torch.zeros_like(time_vec)
        if labels is not None:
            labels = labels.to(device=time_vec.device, dtype=torch.long)
            valid = labels >= 0
            if valid.any():
                label_vec[valid] = self.label_embed(labels[valid])
        if self.label_conditioning_mode == 'add':
            return time_vec + label_vec
        if self.label_conditioning_mode == 'film':
            if self.label_film is None:
                return time_vec
            scale, shift = self.label_film(label_vec).chunk(2, dim=-1)
            return time_vec * (1.0 + scale) + shift
        raise ValueError(f'Unsupported label_conditioning_mode: {self.label_conditioning_mode}')

    def _output_logits(self, h: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        logits = self.out_proj(h)
        if self.class_out_proj is None or labels is None:
            return logits
        labels = labels.to(device=h.device, dtype=torch.long).view(-1)
        for class_id in labels.unique(sorted=True):
            class_int = int(class_id.item())
            if class_int < 0 or class_int >= self.num_classes:
                continue
            mask = labels == class_int
            if mask.any():
                logits[mask] = self.class_out_proj[class_int](h[mask])
        return logits

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = z_t.shape
        global_token = _normalize_tokens(self.global_token).expand(batch_size, -1, -1)
        tokens = torch.cat([global_token, z_t], dim=1)

        h = self.in_proj(tokens)
        h = h + self.pos_embed[:, :seq_len + 1, :]
        if self.token_type_embed is not None and labels is not None:
            labels_local = labels.to(device=h.device, dtype=torch.long).view(-1)
            valid = (labels_local >= 0) & (labels_local < self.num_classes)
            if valid.any():
                if self.token_type_conditioning == 'class_scale':
                    type_ids = labels_local * self.num_scales + self.scale_id
                else:
                    type_ids = labels_local
                type_vec = torch.zeros(batch_size, h.shape[-1], device=h.device, dtype=h.dtype)
                type_vec[valid] = self.token_type_embed(type_ids[valid])
                h[:, 1:, :] = h[:, 1:, :] + type_vec[:, None, :]
        time_vec = self.time_embed(t)
        time_vec = self._apply_label_conditioning(time_vec, labels)
        if self.context_proj is not None:
            if context is None:
                context_vec = torch.zeros_like(time_vec)
            else:
                context_vec = self.context_proj(context.to(device=z_t.device, dtype=z_t.dtype))
            time_vec = time_vec + context_vec

        if self.backbone == 'dit1d':
            for block_idx, block in enumerate(self.blocks):
                h = block(h, time_vec)
                if self.class_adapters is not None:
                    h = self.class_adapters[block_idx](h, labels)
        else:
            h = h + time_vec[:, None, :]
            h = self.encoder(h)
            if self.class_adapters is not None:
                h = self.class_adapters[0](h, labels)

        h = self.out_norm(h)
        logits = self._output_logits(h, labels)
        return logits[:, 1:, :]


class FlowCEScaleModel(nn.Module):
    def __init__(
        self,
        num_codes: int,
        seq_len: int,
        code_dim: int,
        variant: str,
        num_train_samples: Optional[int],
        latent_rank: int,
        latent_noise_std: float,
        t_scheduler: str,
        backbone: str,
        hidden_dim: int,
        depth: int,
        n_heads: int,
        dropout: float,
        max_seq_len: int,
        train_mixup_prob: float,
        train_mixup_alpha: float,
        structure_loss_weight: float,
        senior_mean_reg_weight: float,
        senior_std_reg_weight: float,
        senior_sample_noise_std: float,
        source_prior_mode: str = 'learned',
        num_classes: int = 0,
        use_context: bool = False,
        class_specific_output_head: bool = False,
        class_specific_adapter: bool = False,
        label_conditioning_mode: str = 'add',
        token_type_conditioning: str = 'none',
        scale_id: int = 0,
        num_scales: int = 1,
    ):
        super().__init__()
        self.num_codes = int(num_codes)
        self.seq_len = int(seq_len)
        self.code_dim = int(code_dim)
        self.variant = variant
        self.latent_rank = int(latent_rank)
        self.latent_noise_std = float(latent_noise_std)
        self.t_scheduler = t_scheduler
        self.train_mixup_prob = float(train_mixup_prob)
        self.train_mixup_alpha = float(train_mixup_alpha)
        self.structure_loss_weight = float(structure_loss_weight)
        self.senior_mean_reg_weight = float(senior_mean_reg_weight)
        self.senior_std_reg_weight = float(senior_std_reg_weight)
        self.senior_sample_noise_std = float(senior_sample_noise_std)
        self.source_prior_mode = str(source_prior_mode)
        if self.source_prior_mode not in {'learned', 'gaussian'}:
            raise ValueError(f'Unsupported source_prior_mode: {self.source_prior_mode}')
        self.num_classes = int(num_classes)

        self.backbone = FlowCEBackbone1D(
            num_codes=num_codes,
            code_dim=code_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            backbone=backbone,
            num_classes=self.num_classes,
            use_context=use_context,
            class_specific_output_head=class_specific_output_head,
            class_specific_adapter=class_specific_adapter,
            label_conditioning_mode=label_conditioning_mode,
            token_type_conditioning=token_type_conditioning,
            scale_id=scale_id,
            num_scales=num_scales,
        )

        self.latent_to_tokens = nn.Parameter(torch.randn(self.latent_rank, self.seq_len * self.code_dim) * 0.01)

        if self.variant != 'senior':
            raise ValueError('Only CE-FM senior variant is supported.')
        if num_train_samples is None or int(num_train_samples) <= 0:
            raise ValueError('senior CE-FM requires num_train_samples > 0')
        self.sample_latents = nn.Parameter(torch.randn(int(num_train_samples), self.latent_rank) * 0.01)
        self._kde_cache: Dict[Tuple[str, float, int], KDEPrior] = {}
        self._last_train_mixup_state: Optional[Dict[str, object]] = None
        self._last_sampling_state: Optional[Dict[str, object]] = None

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._kde_cache = {}
            self._last_train_mixup_state = None
            self._last_sampling_state = None
        return self

    def _schedule_t(self, t: torch.Tensor) -> torch.Tensor:
        if self.t_scheduler == 'linear':
            return t
        if self.t_scheduler == 'cosine':
            return 1.0 - torch.cos((t ** 2.0) * 0.5 * math.pi)
        raise ValueError(f'Unsupported t_scheduler: {self.t_scheduler}')

    def _latent_to_state(self, latents: torch.Tensor) -> torch.Tensor:
        flat = torch.matmul(latents, self.latent_to_tokens)
        z0 = flat.view(-1, self.seq_len, self.code_dim)
        return _normalize_tokens(z0)

    def _sample_senior_train_latents(self, sample_ids: torch.Tensor) -> torch.Tensor:
        if self.source_prior_mode == 'gaussian':
            return torch.randn(
                sample_ids.shape[0],
                self.latent_rank,
                device=sample_ids.device,
                dtype=self.sample_latents.dtype,
            )
        latents = self.sample_latents[sample_ids]
        if self.training and self.latent_noise_std > 0:
            latents = latents + self.latent_noise_std * torch.randn_like(latents)
        return latents

    def _senior_regularization_terms(
        self,
        sample_ids: torch.Tensor,
        z1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.source_prior_mode == 'gaussian':
            zero = z1.new_zeros(())
            return zero, zero, zero
        u_batch = self.sample_latents[sample_ids]
        z1_flat = z1.reshape(z1.shape[0], -1)

        real_dist = torch.cdist(z1_flat, z1_flat, p=2)
        real_dist = real_dist / (real_dist.mean() + 1e-6)

        u_dist = torch.cdist(u_batch, u_batch, p=2)
        u_dist = u_dist / (u_dist.mean() + 1e-6)

        structure_loss = F.mse_loss(u_dist, real_dist)
        mean_loss = self.sample_latents.mean(dim=0).abs().mean()
        std_loss = (self.sample_latents.std(dim=0) - 1.0).abs().mean()
        return mean_loss, std_loss, structure_loss

    def _maybe_apply_senior_mixup(
        self,
        local_indices: torch.Tensor,
        z1: torch.Tensor,
        sample_ids: torch.Tensor,
        dtype: torch.dtype,
        mixup_state: Optional[Dict[str, object]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[float]]:
        latents = self._sample_senior_train_latents(sample_ids)
        z0 = self._latent_to_state(latents.to(dtype))
        self._last_train_mixup_state = _clone_state_dict(mixup_state)

        if mixup_state is None or not bool(mixup_state.get('use_mixup', False)):
            return z0, z1, None, None

        perm = mixup_state.get('perm')
        lam = mixup_state.get('lam')
        if not isinstance(perm, torch.Tensor) or lam is None:
            raise ValueError('Invalid shared mixup_state for senior training')
        perm = perm.to(device=local_indices.device, dtype=torch.long)
        lam = float(lam)

        if self.source_prior_mode == 'gaussian':
            z0_mix = _normalize_tokens(torch.randn_like(z1))
        else:
            u_current = self.sample_latents[sample_ids]
            u_perm = self.sample_latents[sample_ids[perm]]
            u_mix = lam * u_current + (1.0 - lam) * u_perm
            if self.latent_noise_std > 0:
                u_mix = u_mix + self.latent_noise_std * torch.randn_like(u_mix)
            z0_mix = self._latent_to_state(u_mix.to(dtype))
        z1_mix = lam * z1 + (1.0 - lam) * z1[perm]
        permuted_indices = local_indices[perm]
        return z0_mix, z1_mix, permuted_indices, lam

    def _get_kde_prior(
        self,
        device: torch.device,
        kde_bandwidth_factor: float,
        kde_max_centers: int,
    ) -> KDEPrior:
        key = (str(device), float(kde_bandwidth_factor), int(kde_max_centers))
        if key not in self._kde_cache:
            samples = self.sample_latents.detach().to(device=device, dtype=torch.float32)
            self._kde_cache[key] = KDEPrior(
                samples,
                bandwidth_factor=kde_bandwidth_factor,
                max_centers=kde_max_centers,
            )
        return self._kde_cache[key]

    def _sample_senior_mixup_latents(
        self,
        batch_size: int,
        noise_scale: float,
        device: torch.device,
        dtype: torch.dtype,
        sampler_state: Optional[Dict[str, object]] = None,
    ) -> torch.Tensor:
        if sampler_state is not None:
            idx_a = sampler_state.get('idx_a')
            idx_b = sampler_state.get('idx_b')
            alpha = sampler_state.get('alpha')
            if not isinstance(idx_a, torch.Tensor) or not isinstance(idx_b, torch.Tensor) or not isinstance(alpha, torch.Tensor):
                raise ValueError('Invalid shared sampler_state for senior mixup sampling')
            idx_a = idx_a.to(device=device, dtype=torch.long)
            idx_b = idx_b.to(device=device, dtype=torch.long)
            alpha = alpha.to(device=device, dtype=dtype)
        else:
            idx_a = torch.randint(0, self.sample_latents.shape[0], (batch_size,), device=device)
            idx_b = torch.randint(0, self.sample_latents.shape[0], (batch_size,), device=device)
            alpha = torch.rand(batch_size, 1, device=device, dtype=dtype)

        u_a = self.sample_latents[idx_a].to(device=device, dtype=dtype)
        u_b = self.sample_latents[idx_b].to(device=device, dtype=dtype)
        latents = alpha * u_a + (1.0 - alpha) * u_b

        if self.senior_sample_noise_std > 0:
            latents = latents + noise_scale * self.senior_sample_noise_std * torch.randn_like(latents)
        return latents

    def _sample_senior_kde_latents(
        self,
        batch_size: int,
        noise_scale: float,
        device: torch.device,
        dtype: torch.dtype,
        kde_bandwidth_factor: float,
        kde_max_centers: int,
        sampler_state: Optional[Dict[str, object]] = None,
    ) -> torch.Tensor:
        prior = self._get_kde_prior(device, kde_bandwidth_factor, kde_max_centers)
        center_indices = None
        if sampler_state is not None:
            center_indices = sampler_state.get('center_indices')
            if center_indices is not None and not isinstance(center_indices, torch.Tensor):
                raise ValueError('Invalid shared sampler_state for senior KDE sampling')
        return prior.sample(batch_size, noise_scale=noise_scale, center_indices=center_indices).to(dtype=dtype)

    def sample_inference_initial_state(
        self,
        batch_size: int,
        noise_scale: float,
        device: torch.device,
        dtype: torch.dtype,
        senior_sampler: Optional[str] = None,
        kde_bandwidth_factor: float = 1.0,
        kde_max_centers: int = 2000,
        sampler_state: Optional[Dict[str, object]] = None,
    ) -> torch.Tensor:
        if senior_sampler is None:
            raise ValueError('senior sampling requires explicit senior_sampler=mixup|kde|gaussian')
        self._last_sampling_state = _clone_state_dict(sampler_state)
        if senior_sampler == 'gaussian' or self.source_prior_mode == 'gaussian':
            z0 = torch.randn(batch_size, self.seq_len, self.code_dim, device=device, dtype=dtype)
            return _normalize_tokens(z0)
        if senior_sampler == 'mixup':
            latents = self._sample_senior_mixup_latents(
                batch_size,
                noise_scale,
                device,
                dtype,
                sampler_state=sampler_state,
            )
        elif senior_sampler == 'kde':
            latents = self._sample_senior_kde_latents(
                batch_size,
                noise_scale,
                device,
                dtype,
                kde_bandwidth_factor,
                kde_max_centers,
                sampler_state=sampler_state,
            )
        else:
            raise ValueError(f'Unsupported senior_sampler: {senior_sampler}')
        return self._latent_to_state(latents)

    @staticmethod
    def _sample_contrastive_negatives(
        labels: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        neg_idx = torch.zeros(batch_size, device=device, dtype=torch.long)
        valid = torch.zeros(batch_size, device=device, dtype=torch.bool)
        if batch_size < 2:
            return neg_idx, valid

        if mode == 'random_nonself':
            offsets = torch.randint(1, batch_size, (batch_size,), device=device)
            neg_idx = (torch.arange(batch_size, device=device) + offsets) % batch_size
            valid.fill_(True)
            return neg_idx, valid

        if mode == 'different_label':
            if labels is None:
                raise ValueError('--contrastive-negative-mode different_label requires labels.')
            labels = labels.to(device=device, dtype=torch.long).view(-1)
            if labels.shape[0] != batch_size:
                raise ValueError(f'Expected labels shape=({batch_size},), got {tuple(labels.shape)}')
            for class_id in labels.unique(sorted=True):
                anchor_mask = labels == class_id
                pool = torch.nonzero(labels != class_id, as_tuple=False).flatten()
                if pool.numel() == 0:
                    continue
                count = int(anchor_mask.sum().item())
                neg_idx[anchor_mask] = pool[torch.randint(0, pool.numel(), (count,), device=device)]
                valid[anchor_mask] = True
            return neg_idx, valid

        raise ValueError(f'Unsupported contrastive negative mode: {mode}')

    def _compute_contrastive_loss(
        self,
        anchor_repr: torch.Tensor,
        positive_repr: torch.Tensor,
        negative_pool_repr: torch.Tensor,
        sample_labels: Optional[torch.Tensor],
        sample_weights: Optional[torch.Tensor],
        negative_mode: str,
        objective: str,
        margin: float,
        temperature: float,
    ) -> Tuple[torch.Tensor, float]:
        batch_size = anchor_repr.shape[0]
        device = anchor_repr.device
        neg_idx, valid = self._sample_contrastive_negatives(
            sample_labels,
            batch_size=batch_size,
            device=device,
            mode=negative_mode,
        )
        if not valid.any():
            return anchor_repr.new_zeros(()), 0.0

        negative_repr = negative_pool_repr[neg_idx]
        pos_dist = (anchor_repr - positive_repr).pow(2).mean(dim=(1, 2))
        neg_dist = (anchor_repr - negative_repr).pow(2).mean(dim=(1, 2))
        if objective == 'margin':
            loss_by_sample = F.relu(float(margin) + pos_dist - neg_dist)
        elif objective == 'delta':
            loss_by_sample = pos_dist - float(temperature) * neg_dist
        else:
            raise ValueError(f'Unsupported contrastive objective: {objective}')

        loss_by_sample = loss_by_sample[valid]
        if sample_weights is not None:
            weights = sample_weights.to(device=device, dtype=loss_by_sample.dtype)[valid]
            loss = (loss_by_sample * weights).sum() / weights.sum().clamp_min(1e-6)
        else:
            loss = loss_by_sample.mean()
        valid_frac = float(valid.float().mean().detach().item())
        return loss, valid_frac

    def compute_loss(
        self,
        local_indices: torch.Tensor,
        codebook: torch.Tensor,
        sample_ids: Optional[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        sample_labels: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
        mixup_state: Optional[Dict[str, object]] = None,
        dynamic_loss_weight: float = 0.0,
        dynamic_loss_components: Optional[Sequence[str]] = None,
        context: Optional[torch.Tensor] = None,
        contrastive_flow_weight: float = 0.0,
        contrastive_representation: str = 'token',
        contrastive_negative_mode: str = 'random_nonself',
        contrastive_objective: str = 'margin',
        contrastive_margin: float = 1.0,
        contrastive_temperature: float = 1.0,
        balanced_fm_loss: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size = local_indices.shape[0]
        device = local_indices.device
        dtype = codebook.dtype

        z1 = F.embedding(local_indices.long(), codebook)

        if sample_ids is None:
            raise ValueError('sample_ids are required for senior CE-FM training')
        mean_loss, std_loss, structure_loss = self._senior_regularization_terms(sample_ids, z1)
        z0, z1, permuted_indices, lam = self._maybe_apply_senior_mixup(
            local_indices,
            z1,
            sample_ids,
            dtype,
            mixup_state=mixup_state,
        )
        total_regularization = (
            self.senior_mean_reg_weight * mean_loss
            + self.senior_std_reg_weight * std_loss
            + self.structure_loss_weight * structure_loss
        )

        t = torch.rand(batch_size, device=device, dtype=dtype)
        t = self._schedule_t(t)
        t_tokens = t[:, None, None]
        z_t = _normalize_tokens((1.0 - t_tokens) * z0 + t_tokens * z1)
        logits = self.backbone(z_t, t, labels=labels, context=context)

        logits_flat = logits.reshape(-1, self.num_codes)
        ce_loss = F.cross_entropy(
            logits_flat,
            local_indices.reshape(-1).long(),
            reduction='none',
        )
        if permuted_indices is not None and lam is not None:
            ce_perm = F.cross_entropy(
                logits_flat,
                permuted_indices.reshape(-1).long(),
                reduction='none',
            )
            ce_loss = lam * ce_loss + (1.0 - lam) * ce_perm
        ce_loss_by_token = ce_loss.reshape(batch_size, -1)
        ce_loss_by_sample = ce_loss_by_token.mean(dim=1)
        balanced_class_count = 0
        if balanced_fm_loss and sample_labels is not None:
            labels_for_balance = sample_labels.to(device=device, dtype=torch.long)
            class_losses = []
            for class_id in labels_for_balance.unique(sorted=True):
                mask = labels_for_balance == int(class_id.item())
                if mask.any():
                    class_losses.append(ce_loss_by_sample[mask].mean())
            if class_losses:
                ce_loss = torch.stack(class_losses).mean()
                balanced_class_count = len(class_losses)
            else:
                ce_loss = ce_loss_by_sample.mean()
        elif sample_weights is not None:
            weights = sample_weights.to(device=device, dtype=ce_loss_by_sample.dtype)
            ce_loss = (ce_loss_by_sample * weights).sum() / weights.sum().clamp_min(1e-6)
        else:
            ce_loss = ce_loss_by_sample.mean()

        dynamic_loss = torch.zeros((), device=device, dtype=ce_loss.dtype)
        if dynamic_loss_weight > 0 and self.seq_len > 1:
            components = set(dynamic_loss_components or [])
            probs = F.softmax(logits, dim=-1)
            z_pred = _normalize_tokens(torch.matmul(probs, codebook))
            pred_diff = z_pred[:, 1:, :] - z_pred[:, :-1, :]
            real_diff = z1[:, 1:, :] - z1[:, :-1, :]
            if 'diff_mean' in components:
                dynamic_loss = dynamic_loss + (pred_diff.abs().mean() - real_diff.abs().mean()).abs()
            if 'diff_std' in components:
                dynamic_loss = dynamic_loss + (pred_diff.std(unbiased=False) - real_diff.std(unbiased=False)).abs()
            if 'feat_std' in components:
                dynamic_loss = dynamic_loss + (
                    z_pred.std(dim=(0, 1), unbiased=False) - z1.std(dim=(0, 1), unbiased=False)
                ).abs().mean()

        contrastive_loss = torch.zeros((), device=device, dtype=ce_loss.dtype)
        contrastive_valid_frac = 0.0
        if contrastive_flow_weight > 0:
            probs = F.softmax(logits, dim=-1)
            z_pred = _normalize_tokens(torch.matmul(probs, codebook))
            if contrastive_representation == 'token':
                anchor_repr = z_pred
                positive_repr = z1
                negative_pool_repr = z1
            elif contrastive_representation == 'velocity':
                anchor_repr = z_pred - z_t
                positive_repr = z1 - z_t
                negative_pool_repr = z1 - z_t
            else:
                raise ValueError(f'Unsupported contrastive representation: {contrastive_representation}')
            contrastive_loss, contrastive_valid_frac = self._compute_contrastive_loss(
                anchor_repr=anchor_repr,
                positive_repr=positive_repr,
                negative_pool_repr=negative_pool_repr,
                sample_labels=sample_labels,
                sample_weights=sample_weights,
                negative_mode=contrastive_negative_mode,
                objective=contrastive_objective,
                margin=contrastive_margin,
                temperature=contrastive_temperature,
            )

        total_loss = (
            ce_loss
            + total_regularization
            + float(dynamic_loss_weight) * dynamic_loss
            + float(contrastive_flow_weight) * contrastive_loss
        )

        pred_indices = logits.argmax(dim=-1)
        correct_by_sample = (pred_indices == local_indices).float().mean(dim=1)
        accuracy = correct_by_sample.mean()

        metrics: Dict[str, float] = {
            'ce_loss': float(ce_loss.detach().item()),
            'accuracy': float(accuracy.detach().item()),
            'mean_loss': float(mean_loss.detach().item()),
            'std_loss': float(std_loss.detach().item()),
            'structure_loss': float(structure_loss.detach().item()),
            'dynamic_loss': float(dynamic_loss.detach().item()),
            'contrastive_loss': float(contrastive_loss.detach().item()),
            'contrastive_valid_frac': contrastive_valid_frac,
            'contrastive_representation_velocity': float(contrastive_representation == 'velocity'),
            'balanced_class_count': float(balanced_class_count),
          }
        if sample_labels is not None:
            sample_labels = sample_labels.to(device=device, dtype=torch.long)
            for class_id in sorted(sample_labels.unique().detach().cpu().tolist()):
                mask = sample_labels == int(class_id)
                if mask.any():
                    metrics[f'ce_y{int(class_id)}'] = float(ce_loss_by_sample[mask].detach().mean().item())
                    metrics[f'acc_y{int(class_id)}'] = float(correct_by_sample[mask].detach().mean().item())
        return total_loss, metrics

    def _guided_logits(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        labels: Optional[torch.Tensor],
        label_guidance_scale: float,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cond_logits = self.backbone(z, t, labels=labels, context=context)
        if self.backbone.label_embed is None or labels is None or abs(label_guidance_scale - 1.0) < 1e-8:
            return cond_logits
        uncond_logits = self.backbone(z, t, labels=None, context=context)
        return uncond_logits + float(label_guidance_scale) * (cond_logits - uncond_logits)

    @staticmethod
    def _apply_token_manifold_guidance(
        logits: torch.Tensor,
        labels: Optional[torch.Tensor],
        token_manifold_bias: Optional[torch.Tensor],
        token_manifold_guidance_weight: float,
        token_manifold_guidance_target_label: int,
        token_transition_bias: Optional[torch.Tensor] = None,
        token_transition_guidance_weight: float = 0.0,
        token_cross_scale_bias: Optional[torch.Tensor] = None,
        token_cross_scale_guidance_weight: float = 0.0,
        current_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (
            token_manifold_bias is None
            and token_transition_bias is None
            and token_cross_scale_bias is None
        ):
            return logits
        if labels is None or (
            float(token_manifold_guidance_weight) == 0.0
            and float(token_transition_guidance_weight) == 0.0
            and float(token_cross_scale_guidance_weight) == 0.0
        ):
            return logits
        labels = labels.to(device=logits.device, dtype=torch.long).view(-1)
        guided = logits.clone()

        if token_manifold_bias is not None and float(token_manifold_guidance_weight) != 0.0:
            bias = token_manifold_bias.to(device=logits.device, dtype=logits.dtype)
            if bias.ndim == 1:
                if bias.shape[0] != logits.shape[-1]:
                    raise ValueError(
                        f'token_manifold_bias must have shape=({logits.shape[-1]},), '
                        f'got {tuple(bias.shape)}'
                    )
                mask = labels == int(token_manifold_guidance_target_label)
                if mask.any():
                    guided[mask] = guided[mask] + float(token_manifold_guidance_weight) * bias.view(1, 1, -1)
            elif bias.ndim == 2:
                if bias.shape[1] != logits.shape[-1]:
                    raise ValueError(
                        f'classwise token_manifold_bias must have shape=(num_classes,{logits.shape[-1]}), '
                        f'got {tuple(bias.shape)}'
                    )
                valid = (labels >= 0) & (labels < bias.shape[0])
                if valid.any():
                    guided[valid] = guided[valid] + float(token_manifold_guidance_weight) * bias[labels[valid]].unsqueeze(1)
            else:
                raise ValueError(f'token_manifold_bias must be 1D or 2D, got {tuple(bias.shape)}')

        if (
            token_transition_bias is not None
            and float(token_transition_guidance_weight) != 0.0
            and current_indices is not None
            and logits.shape[1] > 1
        ):
            trans_bias = token_transition_bias.to(device=logits.device, dtype=logits.dtype)
            current_indices = current_indices.to(device=logits.device, dtype=torch.long)
            if current_indices.shape[:2] != logits.shape[:2]:
                raise ValueError(
                    f'current_indices must have shape={tuple(logits.shape[:2])}, '
                    f'got {tuple(current_indices.shape)}'
                )
            prev_idx = current_indices[:, :-1].clamp(min=0, max=logits.shape[-1] - 1)
            if trans_bias.ndim == 2:
                if trans_bias.shape != (logits.shape[-1], logits.shape[-1]):
                    raise ValueError(
                        f'token_transition_bias must have shape=({logits.shape[-1]},{logits.shape[-1]}), '
                        f'got {tuple(trans_bias.shape)}'
                    )
                mask = labels == int(token_manifold_guidance_target_label)
                if mask.any():
                    guided[mask, 1:, :] = guided[mask, 1:, :] + float(token_transition_guidance_weight) * trans_bias[prev_idx[mask]]
            elif trans_bias.ndim == 3:
                if trans_bias.shape[1:] != (logits.shape[-1], logits.shape[-1]):
                    raise ValueError(
                        f'classwise token_transition_bias must have shape=(num_classes,{logits.shape[-1]},{logits.shape[-1]}), '
                        f'got {tuple(trans_bias.shape)}'
                    )
                valid = (labels >= 0) & (labels < trans_bias.shape[0])
                if valid.any():
                    class_bias = trans_bias[labels[valid]]
                    gathered = torch.gather(
                        class_bias,
                        1,
                        prev_idx[valid].unsqueeze(-1).expand(-1, -1, logits.shape[-1]),
                    )
                    guided[valid, 1:, :] = guided[valid, 1:, :] + float(token_transition_guidance_weight) * gathered
            else:
                raise ValueError(f'token_transition_bias must be 2D or 3D, got {tuple(trans_bias.shape)}')

        if token_cross_scale_bias is not None and float(token_cross_scale_guidance_weight) != 0.0:
            cross_bias = token_cross_scale_bias.to(device=logits.device, dtype=logits.dtype)
            if cross_bias.shape != logits.shape:
                raise ValueError(
                    f'token_cross_scale_bias must have shape={tuple(logits.shape)}, '
                    f'got {tuple(cross_bias.shape)}'
                )
            guided = guided + float(token_cross_scale_guidance_weight) * cross_bias
        return guided

    @staticmethod
    def _apply_additive_token_bias(
        logits: torch.Tensor,
        labels: Optional[torch.Tensor],
        token_bias: Optional[torch.Tensor],
        weight: float,
        target_label: int,
        name: str = 'token_bias',
    ) -> torch.Tensor:
        if token_bias is None or float(weight) == 0.0 or labels is None:
            return logits
        bias = token_bias.to(device=logits.device, dtype=logits.dtype)
        if bias.ndim != 1 or bias.shape[0] != logits.shape[-1]:
            raise ValueError(f'{name} must have shape=({logits.shape[-1]},), got {tuple(bias.shape)}')
        labels = labels.to(device=logits.device, dtype=torch.long).view(-1)
        mask = labels == int(target_label)
        if not mask.any():
            return logits
        guided = logits.clone()
        guided[mask] = guided[mask] + float(weight) * bias.view(1, 1, -1)
        return guided

    @staticmethod
    def _scheduled_guidance_weight(
        target_weight: float,
        step_idx: int,
        flow_steps: int,
        schedule: str,
        warmup_frac: float,
    ) -> float:
        target_weight = float(target_weight)
        if target_weight == 0.0:
            return 0.0
        if schedule == 'constant':
            return target_weight
        if schedule not in {'late_linear', 'late_cosine'}:
            raise ValueError(f'Unsupported token manifold guidance schedule: {schedule}')
        flow_steps = max(int(flow_steps), 1)
        warmup_frac = min(max(float(warmup_frac), 0.0), 0.95)
        progress = float(step_idx + 1) / float(flow_steps)
        if progress <= warmup_frac:
            return 0.0
        ramp = (progress - warmup_frac) / max(1.0 - warmup_frac, 1e-6)
        ramp = min(max(ramp, 0.0), 1.0)
        if schedule == 'late_cosine':
            ramp = 0.5 - 0.5 * math.cos(math.pi * ramp)
        return target_weight * ramp

    def _velocity_from_logits(
        self,
        z: torch.Tensor,
        logits: torch.Tensor,
        codebook: torch.Tensor,
        temperature: float,
        t_value: torch.Tensor,
    ) -> torch.Tensor:
        probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
        mu_t = torch.matmul(probs, codebook)
        mu_t = _normalize_tokens(mu_t)
        denom = (1.0 - t_value[:, None, None]).clamp_min(1e-5)
        return (mu_t - z) / denom

    @torch.no_grad()
    def sample(
        self,
        codebook: torch.Tensor,
        quantizer: nn.Module,
        batch_size: int,
        flow_steps: int,
        solver: str,
        temperature: float,
        noise_scale: float,
        device: torch.device,
        dtype: torch.dtype,
        senior_sampler: Optional[str] = None,
        labels: Optional[torch.Tensor] = None,
        label_guidance_scale: float = 1.0,
        token_manifold_bias: Optional[torch.Tensor] = None,
        token_manifold_guidance_weight: float = 0.0,
        token_manifold_guidance_target_label: int = 1,
        token_transition_bias: Optional[torch.Tensor] = None,
        token_transition_guidance_weight: float = 0.0,
        token_cross_scale_bias: Optional[torch.Tensor] = None,
        token_cross_scale_guidance_weight: float = 0.0,
        utility_token_bias: Optional[torch.Tensor] = None,
        utility_token_guidance_weight: float = 0.0,
        utility_token_guidance_target_label: int = 1,
        utility_token_transition_bias: Optional[torch.Tensor] = None,
        utility_token_transition_weight: float = 0.0,
        utility_token_cross_scale_bias: Optional[torch.Tensor] = None,
        utility_token_cross_scale_weight: float = 0.0,
        token_manifold_guidance_schedule: str = 'constant',
        token_manifold_guidance_warmup_frac: float = 0.35,
        kde_bandwidth_factor: float = 1.0,
        kde_max_centers: int = 2000,
        sampler_state: Optional[Dict[str, object]] = None,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.sample_inference_initial_state(
            batch_size=batch_size,
            noise_scale=noise_scale,
            device=device,
            dtype=dtype,
            senior_sampler=senior_sampler,
            kde_bandwidth_factor=kde_bandwidth_factor,
            kde_max_centers=kde_max_centers,
            sampler_state=sampler_state,
        )
        t_span = torch.linspace(0.0, 1.0, flow_steps + 1, device=device, dtype=dtype)
        t_span = self._schedule_t(t_span)

        for step_idx in range(flow_steps):
            t0 = t_span[step_idx].expand(batch_size)
            t1 = t_span[step_idx + 1].expand(batch_size)
            dt = t_span[step_idx + 1] - t_span[step_idx]
            guidance_weight = self._scheduled_guidance_weight(
                token_manifold_guidance_weight,
                step_idx,
                flow_steps,
                token_manifold_guidance_schedule,
                token_manifold_guidance_warmup_frac,
            )
            transition_weight = self._scheduled_guidance_weight(
                token_transition_guidance_weight,
                step_idx,
                flow_steps,
                token_manifold_guidance_schedule,
                token_manifold_guidance_warmup_frac,
            )
            cross_scale_weight = self._scheduled_guidance_weight(
                token_cross_scale_guidance_weight,
                step_idx,
                flow_steps,
                token_manifold_guidance_schedule,
                token_manifold_guidance_warmup_frac,
            )
            utility_transition_weight = self._scheduled_guidance_weight(
                utility_token_transition_weight,
                step_idx,
                flow_steps,
                token_manifold_guidance_schedule,
                token_manifold_guidance_warmup_frac,
            )
            utility_cross_scale_weight = self._scheduled_guidance_weight(
                utility_token_cross_scale_weight,
                step_idx,
                flow_steps,
                token_manifold_guidance_schedule,
                token_manifold_guidance_warmup_frac,
            )

            if solver == 'euler':
                logits = self._guided_logits(z, t0, labels, label_guidance_scale, context=context)
                current_indices = None
                if (
                    (token_transition_bias is not None and transition_weight != 0.0)
                    or (utility_token_transition_bias is not None and utility_transition_weight != 0.0)
                ):
                    current_indices = quantizer.quantize(z.reshape(-1, self.code_dim)).reshape(batch_size, self.seq_len).long()
                logits = self._apply_token_manifold_guidance(
                    logits,
                    labels,
                    token_manifold_bias,
                    guidance_weight,
                    token_manifold_guidance_target_label,
                    token_transition_bias=token_transition_bias,
                    token_transition_guidance_weight=transition_weight,
                    token_cross_scale_bias=token_cross_scale_bias,
                    token_cross_scale_guidance_weight=cross_scale_weight,
                    current_indices=current_indices,
                )
                logits = self._apply_additive_token_bias(
                    logits,
                    labels,
                    utility_token_bias,
                    utility_token_guidance_weight,
                    utility_token_guidance_target_label,
                    name='utility_token_bias',
                )
                logits = self._apply_token_manifold_guidance(
                    logits,
                    labels,
                    None,
                    0.0,
                    utility_token_guidance_target_label,
                    token_transition_bias=utility_token_transition_bias,
                    token_transition_guidance_weight=utility_transition_weight,
                    token_cross_scale_bias=utility_token_cross_scale_bias,
                    token_cross_scale_guidance_weight=utility_cross_scale_weight,
                    current_indices=current_indices,
                )
                velocity = self._velocity_from_logits(z, logits, codebook, temperature, t0)
                z = _normalize_tokens(z + dt * velocity)
            elif solver == 'heun':
                logits_1 = self._guided_logits(z, t0, labels, label_guidance_scale, context=context)
                current_indices_1 = None
                if (
                    (token_transition_bias is not None and transition_weight != 0.0)
                    or (utility_token_transition_bias is not None and utility_transition_weight != 0.0)
                ):
                    current_indices_1 = quantizer.quantize(z.reshape(-1, self.code_dim)).reshape(batch_size, self.seq_len).long()
                logits_1 = self._apply_token_manifold_guidance(
                    logits_1,
                    labels,
                    token_manifold_bias,
                    guidance_weight,
                    token_manifold_guidance_target_label,
                    token_transition_bias=token_transition_bias,
                    token_transition_guidance_weight=transition_weight,
                    token_cross_scale_bias=token_cross_scale_bias,
                    token_cross_scale_guidance_weight=cross_scale_weight,
                    current_indices=current_indices_1,
                )
                logits_1 = self._apply_additive_token_bias(
                    logits_1,
                    labels,
                    utility_token_bias,
                    utility_token_guidance_weight,
                    utility_token_guidance_target_label,
                    name='utility_token_bias',
                )
                logits_1 = self._apply_token_manifold_guidance(
                    logits_1,
                    labels,
                    None,
                    0.0,
                    utility_token_guidance_target_label,
                    token_transition_bias=utility_token_transition_bias,
                    token_transition_guidance_weight=utility_transition_weight,
                    token_cross_scale_bias=utility_token_cross_scale_bias,
                    token_cross_scale_guidance_weight=utility_cross_scale_weight,
                    current_indices=current_indices_1,
                )
                k1 = self._velocity_from_logits(z, logits_1, codebook, temperature, t0)
                z_pred = _normalize_tokens(z + dt * k1)
                logits_2 = self._guided_logits(z_pred, t1, labels, label_guidance_scale, context=context)
                current_indices_2 = None
                if (
                    (token_transition_bias is not None and transition_weight != 0.0)
                    or (utility_token_transition_bias is not None and utility_transition_weight != 0.0)
                ):
                    current_indices_2 = quantizer.quantize(z_pred.reshape(-1, self.code_dim)).reshape(batch_size, self.seq_len).long()
                logits_2 = self._apply_token_manifold_guidance(
                    logits_2,
                    labels,
                    token_manifold_bias,
                    guidance_weight,
                    token_manifold_guidance_target_label,
                    token_transition_bias=token_transition_bias,
                    token_transition_guidance_weight=transition_weight,
                    token_cross_scale_bias=token_cross_scale_bias,
                    token_cross_scale_guidance_weight=cross_scale_weight,
                    current_indices=current_indices_2,
                )
                logits_2 = self._apply_additive_token_bias(
                    logits_2,
                    labels,
                    utility_token_bias,
                    utility_token_guidance_weight,
                    utility_token_guidance_target_label,
                    name='utility_token_bias',
                )
                logits_2 = self._apply_token_manifold_guidance(
                    logits_2,
                    labels,
                    None,
                    0.0,
                    utility_token_guidance_target_label,
                    token_transition_bias=utility_token_transition_bias,
                    token_transition_guidance_weight=utility_transition_weight,
                    token_cross_scale_bias=utility_token_cross_scale_bias,
                    token_cross_scale_guidance_weight=utility_cross_scale_weight,
                    current_indices=current_indices_2,
                )
                k2 = self._velocity_from_logits(z_pred, logits_2, codebook, temperature, t1)
                z = _normalize_tokens(z + 0.5 * dt * (k1 + k2))
            else:
                raise ValueError(f'Unsupported solver: {solver}')

        flat = z.reshape(-1, self.code_dim)
        local_indices = quantizer.quantize(flat).reshape(batch_size, self.seq_len).long()
        return z, local_indices


class MultiScaleFlowMatching(nn.Module):
    def __init__(
        self,
        nb_code: Sequence[int],
        patch_num: Sequence[int],
        code_dim: int,
        fm_backbone: str = 'dit1d',
        flow_path: str = 'ot',
        num_train_samples: Optional[int] = None,
        latent_rank: int = 128,
        latent_noise_std: float = 0.01,
        t_scheduler: str = 'cosine',
        hidden_dim: int = 512,
        depth: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        train_mixup_prob: float = 0.5,
        train_mixup_alpha: float = 1.0,
        structure_loss_weight: float = 10.0,
        senior_mean_reg_weight: float = 0.1,
        senior_std_reg_weight: float = 0.1,
        senior_sample_noise_std: float = 0.01,
        source_prior_mode: str = 'learned',
        num_classes: int = 0,
        cross_scale_conditioning: str = 'none',
        class_specific_output_head: bool = False,
        class_specific_adapter: bool = False,
        label_conditioning_mode: str = 'add',
        token_type_conditioning: str = 'none',
    ):
        super().__init__()
        self.nb_code = list(nb_code)
        self.patch_num = list(patch_num)
        self.code_dim = int(code_dim)
        self.flow_path = flow_path
        self.train_mixup_prob = float(train_mixup_prob)
        self.train_mixup_alpha = float(train_mixup_alpha)
        self.num_classes = int(num_classes)
        self.cross_scale_conditioning = cross_scale_conditioning
        self.class_specific_output_head = bool(class_specific_output_head and self.num_classes > 0)
        self.class_specific_adapter = bool(class_specific_adapter and self.num_classes > 0)
        self.label_conditioning_mode = label_conditioning_mode
        self.token_type_conditioning = token_type_conditioning
        self.source_prior_mode = str(source_prior_mode)
        if self.class_specific_output_head and self.class_specific_adapter:
            raise ValueError('--class-specific-output-head and --class-specific-adapter are mutually exclusive.')
        if self.source_prior_mode not in {'learned', 'gaussian'}:
            raise ValueError(f'Unsupported source_prior_mode: {self.source_prior_mode}')
        if self.cross_scale_conditioning not in {'none', 'ctf'}:
            raise ValueError(f'Unsupported cross_scale_conditioning: {self.cross_scale_conditioning}')
        if self.token_type_conditioning not in {'none', 'class', 'class_scale'}:
            raise ValueError(f'Unsupported token_type_conditioning: {self.token_type_conditioning}')
        self.offsets = self._build_offsets(self.nb_code)
        self._last_train_mixup_state: Optional[Dict[str, object]] = None
        self._last_sampling_state: Optional[Dict[str, object]] = None
        self._train_local_indices: Optional[List[torch.Tensor]] = None
        self._train_labels: Optional[torch.Tensor] = None
        self._token_manifold_biases: Optional[List[torch.Tensor]] = None
        self._token_transition_biases: Optional[List[Optional[torch.Tensor]]] = None
        self._token_cross_scale_biases: Optional[List[Optional[torch.Tensor]]] = None
        self._utility_token_biases: Optional[List[torch.Tensor]] = None
        self._utility_token_transition_biases: Optional[List[Optional[torch.Tensor]]] = None
        self._utility_token_cross_scale_biases: Optional[List[Optional[torch.Tensor]]] = None

        max_seq_len = max(self.patch_num)
        num_scales = len(self.patch_num)
        self.scale_models = nn.ModuleList(
            [
                FlowCEScaleModel(
                    num_codes=nc,
                    seq_len=pn,
                    code_dim=self.code_dim,
                    variant='senior',
                    num_train_samples=num_train_samples,
                    latent_rank=latent_rank,
                    latent_noise_std=latent_noise_std,
                    t_scheduler=t_scheduler,
                    backbone=fm_backbone,
                    hidden_dim=hidden_dim,
                    depth=depth,
                    n_heads=n_heads,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                    train_mixup_prob=train_mixup_prob,
                    train_mixup_alpha=train_mixup_alpha,
                    structure_loss_weight=structure_loss_weight,
                    senior_mean_reg_weight=senior_mean_reg_weight,
                    senior_std_reg_weight=senior_std_reg_weight,
                    senior_sample_noise_std=senior_sample_noise_std,
                    source_prior_mode=self.source_prior_mode,
                    num_classes=self.num_classes,
                    use_context=(self.cross_scale_conditioning == 'ctf' and scale_idx > 0),
                    class_specific_output_head=self.class_specific_output_head,
                    class_specific_adapter=self.class_specific_adapter,
                    label_conditioning_mode=self.label_conditioning_mode,
                    token_type_conditioning=self.token_type_conditioning,
                    scale_id=scale_idx,
                    num_scales=num_scales,
                )
                for scale_idx, (nc, pn) in enumerate(zip(self.nb_code, self.patch_num))
            ]
        )

    @staticmethod
    def _build_offsets(nb_code: Sequence[int]) -> List[int]:
        offsets = [0]
        running = 0
        for nc in nb_code:
            running += int(nc)
            offsets.append(running)
        return offsets

    def _require_per_scale_quantizers(self, quantizers: Sequence[nn.Module]) -> Sequence[nn.Module]:
        if not isinstance(quantizers, (list, tuple, nn.ModuleList)):
            raise TypeError(
                'CE-FM expects per-scale quantizers from Stage-1, '
                f'but got type={type(quantizers)}.'
            )
        if len(quantizers) != len(self.patch_num):
            raise ValueError(
                'quantizer count mismatch: '
                f'got {len(quantizers)} quantizers for {len(self.patch_num)} scales'
            )
        return quantizers

    def split_global_indices(self, code_indices: torch.Tensor) -> List[torch.Tensor]:
        level_indices = []
        start = 0
        for k, pn in enumerate(self.patch_num):
            idx = code_indices[:, start:start + pn] - self.offsets[k]
            level_indices.append(idx.long())
            start += pn
        return level_indices

    def join_local_indices(self, level_indices: Sequence[torch.Tensor]) -> torch.Tensor:
        global_indices = []
        for k, idx in enumerate(level_indices):
            global_indices.append(idx.long() + self.offsets[k])
        return torch.cat(global_indices, dim=1)

    def set_training_code_indices(self, code_indices: torch.Tensor) -> None:
        """Cache training tokens for coarse-to-fine nearest-context sampling.

        This cache is intentionally non-persistent so existing checkpoints and
        the original shared-context sampler remain unchanged.
        """
        with torch.no_grad():
            self._train_local_indices = [idx.detach().cpu().long() for idx in self.split_global_indices(code_indices.cpu())]

    def set_training_labels(self, labels: torch.Tensor) -> None:
        labels = labels.detach().cpu().long().view(-1)
        expected = self.scale_models[0].sample_latents.shape[0]
        if labels.shape[0] != expected:
            raise ValueError(f'Expected {expected} training labels, got {labels.shape[0]}')
        self._train_labels = labels

    def _extract_codebooks(self, quantizers: Sequence[nn.Module]) -> List[torch.Tensor]:
        quantizers = self._require_per_scale_quantizers(quantizers)
        codebooks = []
        for quantizer in quantizers:
            codebook = getattr(quantizer, 'codebook', None)
            if codebook is None:
                raise AttributeError(f'Quantizer {type(quantizer)} does not expose `.codebook`.')
            codebooks.append(_normalize_tokens(codebook.detach()))
        return codebooks

    @staticmethod
    def _build_ctf_context(previous_embeds: Sequence[torch.Tensor]) -> Optional[torch.Tensor]:
        if not previous_embeds:
            return None
        pooled = [emb.mean(dim=1) for emb in previous_embeds]
        return torch.stack(pooled, dim=0).mean(dim=0)

    def dequantize_levels_from_indices(self, code_indices: torch.Tensor, quantizers: Sequence[nn.Module]) -> List[torch.Tensor]:
        codebooks = self._extract_codebooks(quantizers)
        return [
            F.embedding(local_idx.long(), codebook)
            for local_idx, codebook in zip(self.split_global_indices(code_indices), codebooks)
        ]

    def requantize_levels(self, level_embeds: List[torch.Tensor], quantizers: Sequence[nn.Module]) -> torch.Tensor:
        quantizers = self._require_per_scale_quantizers(quantizers)
        level_indices = []
        for quantizer, emb in zip(quantizers, level_embeds):
            batch_size, seq_len, dim = emb.shape
            flat = emb.reshape(batch_size * seq_len, dim)
            idx = quantizer.quantize(flat).reshape(batch_size, seq_len).long()
            level_indices.append(idx)
        return self.join_local_indices(level_indices)

    def _build_shared_train_mixup_state(
        self,
        batch_size: int,
        device: torch.device,
        labels: Optional[torch.Tensor] = None,
        class_aware: bool = False,
    ) -> Dict[str, object]:
        state: Dict[str, object] = {'use_mixup': False, 'perm': None, 'lam': None}
        if not self.training or self.train_mixup_prob <= 0:
            self._last_train_mixup_state = _clone_state_dict(state)
            return state

        if torch.rand(1).item() >= self.train_mixup_prob:
            self._last_train_mixup_state = _clone_state_dict(state)
            return state

        alpha = max(self.train_mixup_alpha, 1e-6)
        lam = float(np.random.beta(alpha, alpha))
        lam = max(lam, 1.0 - lam)
        if class_aware:
            if labels is None:
                raise ValueError('class-aware mixup requires labels.')
            labels = labels.to(device=device, dtype=torch.long).view(-1)
            if labels.shape[0] != batch_size:
                raise ValueError(f'Expected labels shape=({batch_size},), got {tuple(labels.shape)}')
            perm = torch.arange(batch_size, device=device)
            for class_id in labels.unique(sorted=True):
                mask = torch.nonzero(labels == class_id, as_tuple=False).flatten()
                if mask.numel() > 1:
                    perm[mask] = mask[torch.randperm(mask.numel(), device=device)]
        else:
            perm = torch.randperm(batch_size, device=device)
        state = {
            'use_mixup': True,
            'perm': perm,
            'lam': lam,
        }
        self._last_train_mixup_state = _clone_state_dict(state)
        return state

    def _build_shared_sampling_state(
        self,
        batch_size: int,
        senior_sampler: str,
        device: torch.device,
        dtype: torch.dtype,
        kde_bandwidth_factor: float,
        kde_max_centers: int,
        labels: Optional[torch.Tensor] = None,
        label_conditioned_prior: bool = False,
    ) -> Dict[str, object]:
        if senior_sampler == 'gaussian':
            if label_conditioned_prior:
                raise ValueError('label_conditioned_prior is not defined for senior_sampler=gaussian.')
            state: Dict[str, object] = {'sampler': 'gaussian'}
            self._last_sampling_state = _clone_state_dict(state)
            return state

        if senior_sampler == 'mixup':
            num_train_samples = self.scale_models[0].sample_latents.shape[0]
            if label_conditioned_prior:
                if labels is None:
                    raise ValueError('label_conditioned_prior requires sampling labels.')
                if self._train_labels is None:
                    raise ValueError('label_conditioned_prior requires cached training labels via set_training_labels().')
                train_labels = self._train_labels.to(device=device)
                labels = labels.to(device=device, dtype=torch.long).view(-1)
                idx_a = torch.empty(batch_size, device=device, dtype=torch.long)
                idx_b = torch.empty(batch_size, device=device, dtype=torch.long)
                for class_id in labels.unique(sorted=True):
                    mask = labels == class_id
                    pool = torch.nonzero(train_labels == class_id, as_tuple=False).flatten()
                    if pool.numel() == 0:
                        raise ValueError(f'No training latents available for label={int(class_id.item())}')
                    idx_a[mask] = pool[torch.randint(0, pool.numel(), (int(mask.sum().item()),), device=device)]
                    idx_b[mask] = pool[torch.randint(0, pool.numel(), (int(mask.sum().item()),), device=device)]
            else:
                idx_a = torch.randint(0, num_train_samples, (batch_size,), device=device)
                idx_b = torch.randint(0, num_train_samples, (batch_size,), device=device)
            state: Dict[str, object] = {
                'sampler': 'mixup',
                'idx_a': idx_a,
                'idx_b': idx_b,
                'alpha': torch.rand(batch_size, 1, device=device, dtype=dtype),
            }
            self._last_sampling_state = _clone_state_dict(state)
            return state

        if senior_sampler == 'kde':
            if label_conditioned_prior:
                raise ValueError('label_conditioned_prior currently supports senior_sampler=mixup only.')
            priors = [
                scale_model._get_kde_prior(device, kde_bandwidth_factor, kde_max_centers)
                for scale_model in self.scale_models
            ]
            num_centers = min(prior.num_samples for prior in priors)
            state = {
                'sampler': 'kde',
                'center_indices': torch.randint(0, num_centers, (batch_size,), device=device),
            }
            self._last_sampling_state = _clone_state_dict(state)
            return state

        raise ValueError(f'Unsupported senior_sampler: {senior_sampler}')

    def set_token_manifold_biases(self, biases: Optional[Sequence[torch.Tensor]]) -> None:
        if biases is None:
            self._token_manifold_biases = None
            return
        if len(biases) != len(self.scale_models):
            raise ValueError(f'Expected {len(self.scale_models)} token manifold biases, got {len(biases)}')
        checked: List[torch.Tensor] = []
        for scale_idx, (bias, num_codes) in enumerate(zip(biases, self.nb_code)):
            bias = torch.as_tensor(bias, dtype=torch.float32).detach().cpu()
            if bias.ndim == 1:
                valid_shape = bias.shape[0] == int(num_codes)
            elif bias.ndim == 2:
                valid_shape = bias.shape[1] == int(num_codes)
            else:
                valid_shape = False
            if not valid_shape:
                raise ValueError(
                    f'Expected token manifold bias for scale {scale_idx} with shape=({num_codes},) '
                    f'or (num_classes,{num_codes}), '
                    f'got {tuple(bias.shape)}'
                )
            checked.append(bias)
        self._token_manifold_biases = checked

    def set_utility_token_biases(self, biases: Optional[Sequence[torch.Tensor]]) -> None:
        if biases is None:
            self._utility_token_biases = None
            return
        if len(biases) != len(self.scale_models):
            raise ValueError(f'Expected {len(self.scale_models)} utility token biases, got {len(biases)}')
        checked: List[torch.Tensor] = []
        for scale_idx, (bias, num_codes) in enumerate(zip(biases, self.nb_code)):
            bias = torch.as_tensor(bias, dtype=torch.float32).detach().cpu()
            if bias.ndim != 1 or bias.shape[0] != int(num_codes):
                raise ValueError(
                    f'Expected utility token bias for scale {scale_idx} with shape=({num_codes},), '
                    f'got {tuple(bias.shape)}'
                )
            checked.append(bias)
        self._utility_token_biases = checked

    def set_utility_token_transition_biases(self, biases: Optional[Sequence[Optional[torch.Tensor]]]) -> None:
        if biases is None:
            self._utility_token_transition_biases = None
            return
        if len(biases) != len(self.scale_models):
            raise ValueError(f'Expected {len(self.scale_models)} utility token transition biases, got {len(biases)}')
        checked: List[Optional[torch.Tensor]] = []
        for scale_idx, (bias, num_codes) in enumerate(zip(biases, self.nb_code)):
            if bias is None:
                checked.append(None)
                continue
            bias = torch.as_tensor(bias, dtype=torch.float32).detach().cpu()
            if bias.ndim != 2 or bias.shape != (int(num_codes), int(num_codes)):
                raise ValueError(
                    f'Expected utility transition bias for scale {scale_idx} with shape=({num_codes},{num_codes}), '
                    f'got {tuple(bias.shape)}'
                )
            checked.append(bias)
        self._utility_token_transition_biases = checked

    def set_utility_token_cross_scale_biases(self, biases: Optional[Sequence[Optional[torch.Tensor]]]) -> None:
        if biases is None:
            self._utility_token_cross_scale_biases = None
            return
        if len(biases) != len(self.scale_models):
            raise ValueError(f'Expected {len(self.scale_models)} utility token cross-scale biases, got {len(biases)}')
        checked: List[Optional[torch.Tensor]] = []
        for scale_idx, bias in enumerate(biases):
            if scale_idx == 0:
                checked.append(None)
                continue
            if bias is None:
                checked.append(None)
                continue
            parent_codes = int(self.nb_code[scale_idx - 1])
            child_codes = int(self.nb_code[scale_idx])
            bias = torch.as_tensor(bias, dtype=torch.float32).detach().cpu()
            if bias.ndim != 2 or bias.shape != (parent_codes, child_codes):
                raise ValueError(
                    f'Expected utility cross-scale bias for scale {scale_idx} with shape=({parent_codes},{child_codes}), '
                    f'got {tuple(bias.shape)}'
                )
            checked.append(bias)
        self._utility_token_cross_scale_biases = checked

    def set_token_transition_biases(self, biases: Optional[Sequence[Optional[torch.Tensor]]]) -> None:
        if biases is None:
            self._token_transition_biases = None
            return
        if len(biases) != len(self.scale_models):
            raise ValueError(f'Expected {len(self.scale_models)} token transition biases, got {len(biases)}')
        checked: List[Optional[torch.Tensor]] = []
        for scale_idx, (bias, num_codes) in enumerate(zip(biases, self.nb_code)):
            if bias is None:
                checked.append(None)
                continue
            bias = torch.as_tensor(bias, dtype=torch.float32).detach().cpu()
            if bias.ndim == 2:
                valid_shape = bias.shape == (int(num_codes), int(num_codes))
            elif bias.ndim == 3:
                valid_shape = bias.shape[1:] == (int(num_codes), int(num_codes))
            else:
                valid_shape = False
            if not valid_shape:
                raise ValueError(
                    f'Expected transition bias for scale {scale_idx} with shape=({num_codes},{num_codes}) '
                    f'or (num_classes,{num_codes},{num_codes}), got {tuple(bias.shape)}'
                )
            checked.append(bias)
        self._token_transition_biases = checked

    def set_token_cross_scale_biases(self, biases: Optional[Sequence[Optional[torch.Tensor]]]) -> None:
        if biases is None:
            self._token_cross_scale_biases = None
            return
        if len(biases) != len(self.scale_models):
            raise ValueError(f'Expected {len(self.scale_models)} token cross-scale biases, got {len(biases)}')
        checked: List[Optional[torch.Tensor]] = []
        for scale_idx, bias in enumerate(biases):
            if scale_idx == 0:
                checked.append(None)
                continue
            if bias is None:
                checked.append(None)
                continue
            parent_codes = int(self.nb_code[scale_idx - 1])
            child_codes = int(self.nb_code[scale_idx])
            bias = torch.as_tensor(bias, dtype=torch.float32).detach().cpu()
            if bias.ndim == 2:
                valid_shape = bias.shape == (parent_codes, child_codes)
            elif bias.ndim == 3:
                valid_shape = bias.shape[1:] == (parent_codes, child_codes)
            else:
                valid_shape = False
            if not valid_shape:
                raise ValueError(
                    f'Expected cross-scale bias for scale {scale_idx} with shape=({parent_codes},{child_codes}) '
                    f'or (num_classes,{parent_codes},{child_codes}), got {tuple(bias.shape)}'
                )
            checked.append(bias)
        self._token_cross_scale_biases = checked

    def _build_cross_scale_dynamic_bias(
        self,
        scale_idx: int,
        parent_indices: torch.Tensor,
        labels: Optional[torch.Tensor],
        target_label: int,
        batch_size: int,
        child_seq_len: int,
        child_num_codes: int,
        device: torch.device,
        dtype: torch.dtype,
        bias_store: Optional[Sequence[Optional[torch.Tensor]]] = None,
    ) -> Optional[torch.Tensor]:
        bias_store = self._token_cross_scale_biases if bias_store is None else bias_store
        if bias_store is None or scale_idx <= 0:
            return None
        matrix = bias_store[scale_idx]
        if matrix is None:
            return None
        if labels is None:
            return None
        matrix = matrix.to(device=device, dtype=dtype)
        labels = labels.to(device=device, dtype=torch.long).view(-1)
        parent_indices = parent_indices.to(device=device, dtype=torch.long)
        if parent_indices.ndim != 2 or parent_indices.shape[0] != batch_size:
            raise ValueError(
                f'Expected parent_indices shape=({batch_size}, parent_seq_len), got {tuple(parent_indices.shape)}'
            )
        parent_seq_len = parent_indices.shape[1]
        out = torch.zeros(batch_size, child_seq_len, child_num_codes, device=device, dtype=dtype)
        for child_pos in range(child_seq_len):
            parent_pos = min((child_pos * parent_seq_len) // child_seq_len, parent_seq_len - 1)
            parent_token = parent_indices[:, parent_pos]
            if matrix.ndim == 2:
                mask = labels == int(target_label)
                if mask.any():
                    out[mask, child_pos, :] = matrix[parent_token[mask].clamp(0, matrix.shape[0] - 1)]
            else:
                valid = (labels >= 0) & (labels < matrix.shape[0])
                if valid.any():
                    class_matrix = matrix[labels[valid]]
                    gathered = torch.gather(
                        class_matrix,
                        1,
                        parent_token[valid].clamp(0, matrix.shape[1] - 1).view(-1, 1, 1).expand(-1, 1, child_num_codes),
                    ).squeeze(1)
                    out[valid, child_pos, :] = gathered
        return out

    def _nearest_context_state(
        self,
        previous_indices: Sequence[torch.Tensor],
        batch_size: int,
        senior_sampler: str,
        device: torch.device,
        dtype: torch.dtype,
        target_scale: int,
    ) -> Dict[str, object]:
        if self._train_local_indices is None:
            raise ValueError('ctf_nearest sampling requires cached training tokens via set_training_code_indices().')
        if senior_sampler != 'mixup':
            raise ValueError('ctf_nearest currently supports senior_sampler=mixup only.')
        if target_scale <= 0:
            raise ValueError('target_scale must be > 0 for coarse-to-fine context.')

        memories = [self._train_local_indices[k].to(device=device) for k in range(target_scale)]
        train_prev = torch.cat(memories, dim=1)
        query_prev = torch.cat([idx.to(device=device).long() for idx in previous_indices], dim=1)

        nearest_ids = []
        chunk = 64
        for start in range(0, batch_size, chunk):
            query = query_prev[start:start + chunk]
            distances = (query[:, None, :] != train_prev[None, :, :]).float().mean(dim=-1)
            nearest_ids.append(distances.argmin(dim=1))
        nearest_ids = torch.cat(nearest_ids, dim=0).long()
        return {
            'sampler': 'mixup',
            'idx_a': nearest_ids,
            'idx_b': nearest_ids,
            'alpha': torch.ones(batch_size, 1, device=device, dtype=dtype),
        }

    def compute_loss(
        self,
        code_indices: torch.Tensor,
        quantizers: Sequence[nn.Module],
        sample_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
        label_dropout_prob: float = 0.0,
        class_aware_mixup: bool = False,
        dynamic_loss_weight: float = 0.0,
        dynamic_loss_components: Optional[Sequence[str]] = None,
        contrastive_flow_weight: float = 0.0,
        contrastive_representation: str = 'token',
        contrastive_negative_mode: str = 'random_nonself',
        contrastive_objective: str = 'margin',
        contrastive_margin: float = 1.0,
        contrastive_temperature: float = 1.0,
        balanced_fm_loss: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if self.flow_path != 'ot':
            raise ValueError(f'Unsupported flow path: {self.flow_path}')
        if sample_ids is None:
            raise ValueError('sample_ids are required for CE-FM senior training')
        if self.num_classes > 0 and labels is None:
            raise ValueError('Conditional CE-FM requires labels.')
        if labels is not None:
            labels = labels.to(device=code_indices.device, dtype=torch.long)
        model_labels = labels
        sample_weights = None
        if labels is not None:
            if class_weights is not None:
                class_weights = class_weights.to(device=code_indices.device, dtype=torch.float32)
                sample_weights = class_weights[labels]
            if self.training and label_dropout_prob > 0:
                drop_mask = torch.rand(labels.shape, device=labels.device) < float(label_dropout_prob)
                if drop_mask.any():
                    model_labels = labels.clone()
                    model_labels[drop_mask] = -1

        local_indices = self.split_global_indices(code_indices)
        codebooks = self._extract_codebooks(quantizers)
        real_level_embeds = [
            F.embedding(idx.long(), codebook)
            for idx, codebook in zip(local_indices, codebooks)
        ]
        train_mixup_labels = labels if (self.num_classes > 0 and class_aware_mixup) else None
        mixup_state = self._build_shared_train_mixup_state(
            code_indices.shape[0],
            code_indices.device,
            labels=train_mixup_labels,
            class_aware=bool(class_aware_mixup and self.num_classes > 0),
        )

        total_loss = 0.0
        metrics: Dict[str, float] = {}

        for k, (scale_model, idx, codebook) in enumerate(zip(self.scale_models, local_indices, codebooks)):
            context = None
            if self.cross_scale_conditioning == 'ctf' and k > 0:
                context = self._build_ctf_context(real_level_embeds[:k])
            loss_k, scale_metrics = scale_model.compute_loss(
                idx,
                codebook,
                sample_ids,
                labels=model_labels,
                sample_labels=labels,
                sample_weights=sample_weights,
                mixup_state=mixup_state,
                dynamic_loss_weight=dynamic_loss_weight,
                dynamic_loss_components=dynamic_loss_components,
                context=context,
                contrastive_flow_weight=contrastive_flow_weight,
                contrastive_representation=contrastive_representation,
                contrastive_negative_mode=contrastive_negative_mode,
                contrastive_objective=contrastive_objective,
                contrastive_margin=contrastive_margin,
                contrastive_temperature=contrastive_temperature,
                balanced_fm_loss=balanced_fm_loss,
            )
            total_loss = total_loss + loss_k
            metrics[f'loss_s{k}'] = float(loss_k.detach().item())
            metrics[f'ce_s{k}'] = scale_metrics['ce_loss']
            metrics[f'acc_s{k}'] = scale_metrics['accuracy']
            metrics[f'mean_s{k}'] = scale_metrics['mean_loss']
            metrics[f'std_s{k}'] = scale_metrics['std_loss']
            metrics[f'structure_s{k}'] = scale_metrics['structure_loss']
            metrics[f'dynamic_s{k}'] = scale_metrics['dynamic_loss']
            metrics[f'contrastive_s{k}'] = scale_metrics['contrastive_loss']
            metrics[f'contrastive_valid_frac_s{k}'] = scale_metrics['contrastive_valid_frac']
            metrics[f'contrastive_velocity_s{k}'] = scale_metrics['contrastive_representation_velocity']
            metrics[f'balanced_class_count_s{k}'] = scale_metrics['balanced_class_count']
            for metric_name, metric_value in scale_metrics.items():
                if metric_name.startswith(('ce_y', 'acc_y')):
                    metrics[f'{metric_name}_s{k}'] = metric_value

        total_loss = total_loss / float(len(self.scale_models))
        if labels is not None:
            for metric_prefix in ('ce_y', 'acc_y'):
                class_ids = sorted(labels.unique().detach().cpu().tolist())
                for class_id in class_ids:
                    values = [
                        metrics[f'{metric_prefix}{int(class_id)}_s{k}']
                        for k in range(len(self.scale_models))
                        if f'{metric_prefix}{int(class_id)}_s{k}' in metrics
                    ]
                    if values:
                        metrics[f'{metric_prefix}{int(class_id)}'] = float(sum(values) / len(values))
        metrics['loss_total'] = float(total_loss.detach().item())
        return total_loss, metrics

    def compute_fm_loss(
        self,
        code_indices: torch.Tensor,
        quantizers: Sequence[nn.Module],
        sample_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
        label_dropout_prob: float = 0.0,
        class_aware_mixup: bool = False,
        dynamic_loss_weight: float = 0.0,
        dynamic_loss_components: Optional[Sequence[str]] = None,
        contrastive_flow_weight: float = 0.0,
        contrastive_negative_mode: str = 'random_nonself',
        contrastive_objective: str = 'margin',
        contrastive_margin: float = 1.0,
        contrastive_temperature: float = 1.0,
        balanced_fm_loss: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        return self.compute_loss(
            code_indices,
            quantizers,
            sample_ids,
            labels=labels,
            class_weights=class_weights,
            label_dropout_prob=label_dropout_prob,
            class_aware_mixup=class_aware_mixup,
            dynamic_loss_weight=dynamic_loss_weight,
            dynamic_loss_components=dynamic_loss_components,
            contrastive_flow_weight=contrastive_flow_weight,
            contrastive_negative_mode=contrastive_negative_mode,
            contrastive_objective=contrastive_objective,
            contrastive_margin=contrastive_margin,
            contrastive_temperature=contrastive_temperature,
            balanced_fm_loss=balanced_fm_loss,
        )

    @torch.no_grad()
    def sample_token_indices(
        self,
        batch_size: int,
        quantizers: Sequence[nn.Module],
        flow_steps: int = 30,
        solver: str = 'euler',
        sample_temperature: float = 0.9,
        noise_scale: float = 1.0,
        senior_sampler: Optional[str] = None,
        kde_bandwidth_factor: float = 1.0,
        kde_max_centers: int = 2000,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        labels: Optional[torch.Tensor] = None,
        label_guidance_scale: float = 1.0,
        token_manifold_guidance_weight: float = 0.0,
        token_manifold_guidance_target_label: int = 1,
        token_manifold_transition_weight: float = 0.0,
        token_manifold_cross_scale_weight: float = 0.0,
        utility_token_guidance_weight: float = 0.0,
        utility_token_guidance_target_label: int = 1,
        utility_token_transition_weight: float = 0.0,
        utility_token_cross_scale_weight: float = 0.0,
        token_manifold_guidance_schedule: str = 'constant',
        token_manifold_guidance_warmup_frac: float = 0.35,
        sampling_mode: str = 'shared_context',
        label_conditioned_prior: bool = False,
    ) -> torch.Tensor:
        if senior_sampler is None:
            raise ValueError('senior sampling requires explicit senior_sampler=mixup|kde|gaussian')

        device = device if device is not None else next(self.parameters()).device
        if self.num_classes > 0:
            if labels is None:
                raise ValueError('Conditional CE-FM sampling requires labels.')
            labels = labels.to(device=device, dtype=torch.long)
            if labels.ndim == 0:
                labels = labels.repeat(batch_size)
            if labels.shape[0] != batch_size:
                raise ValueError(f'Expected labels shape=({batch_size},), got {tuple(labels.shape)}')
        quantizers = self._require_per_scale_quantizers(quantizers)
        codebooks = self._extract_codebooks(quantizers)
        if sampling_mode not in {'shared_context', 'ctf_nearest'}:
            raise ValueError(f'Unsupported sampling_mode: {sampling_mode}')
        sampler_state = self._build_shared_sampling_state(
            batch_size=batch_size,
            senior_sampler=senior_sampler,
            device=device,
            dtype=dtype,
            kde_bandwidth_factor=kde_bandwidth_factor,
            kde_max_centers=kde_max_centers,
            labels=labels,
            label_conditioned_prior=label_conditioned_prior,
        )

        local_indices = []
        local_embeds = []
        for scale_idx, (scale_model, codebook, quantizer) in enumerate(zip(self.scale_models, codebooks, quantizers)):
            state_k = sampler_state
            if sampling_mode == 'ctf_nearest' and scale_idx > 0:
                state_k = self._nearest_context_state(
                    previous_indices=local_indices,
                    batch_size=batch_size,
                    senior_sampler=senior_sampler,
                    device=device,
                    dtype=dtype,
                    target_scale=scale_idx,
                )
            context = None
            if self.cross_scale_conditioning == 'ctf' and scale_idx > 0:
                context = self._build_ctf_context(local_embeds)
            token_manifold_bias = None
            if self._token_manifold_biases is not None:
                token_manifold_bias = self._token_manifold_biases[scale_idx]
            utility_token_bias = None
            if self._utility_token_biases is not None:
                utility_token_bias = self._utility_token_biases[scale_idx]
            utility_token_transition_bias = None
            if self._utility_token_transition_biases is not None:
                utility_token_transition_bias = self._utility_token_transition_biases[scale_idx]
            token_transition_bias = None
            if self._token_transition_biases is not None:
                token_transition_bias = self._token_transition_biases[scale_idx]
            token_cross_scale_bias = None
            utility_token_cross_scale_bias = None
            if scale_idx > 0 and local_indices:
                token_cross_scale_bias = self._build_cross_scale_dynamic_bias(
                    scale_idx=scale_idx,
                    parent_indices=local_indices[scale_idx - 1],
                    labels=labels,
                    target_label=token_manifold_guidance_target_label,
                    batch_size=batch_size,
                    child_seq_len=int(self.patch_num[scale_idx]),
                    child_num_codes=int(self.nb_code[scale_idx]),
                    device=device,
                    dtype=dtype,
                )
                if self._utility_token_cross_scale_biases is not None:
                    utility_token_cross_scale_bias = self._build_cross_scale_dynamic_bias(
                        scale_idx=scale_idx,
                        parent_indices=local_indices[scale_idx - 1],
                        labels=labels,
                        target_label=utility_token_guidance_target_label,
                        batch_size=batch_size,
                        child_seq_len=int(self.patch_num[scale_idx]),
                        child_num_codes=int(self.nb_code[scale_idx]),
                        device=device,
                        dtype=dtype,
                        bias_store=self._utility_token_cross_scale_biases,
                    )
            _, idx = scale_model.sample(
                codebook=codebook.to(device=device, dtype=dtype),
                quantizer=quantizer,
                batch_size=batch_size,
                flow_steps=flow_steps,
                solver=solver,
                temperature=sample_temperature,
                noise_scale=noise_scale,
                device=device,
                dtype=dtype,
                senior_sampler=senior_sampler,
                labels=labels,
                label_guidance_scale=label_guidance_scale,
                token_manifold_bias=token_manifold_bias,
                token_manifold_guidance_weight=token_manifold_guidance_weight,
                token_manifold_guidance_target_label=token_manifold_guidance_target_label,
                token_transition_bias=token_transition_bias,
                token_transition_guidance_weight=token_manifold_transition_weight,
                token_cross_scale_bias=token_cross_scale_bias,
                token_cross_scale_guidance_weight=token_manifold_cross_scale_weight,
                utility_token_bias=utility_token_bias,
                utility_token_guidance_weight=utility_token_guidance_weight,
                utility_token_guidance_target_label=utility_token_guidance_target_label,
                utility_token_transition_bias=utility_token_transition_bias,
                utility_token_transition_weight=utility_token_transition_weight,
                utility_token_cross_scale_bias=utility_token_cross_scale_bias,
                utility_token_cross_scale_weight=utility_token_cross_scale_weight,
                token_manifold_guidance_schedule=token_manifold_guidance_schedule,
                token_manifold_guidance_warmup_frac=token_manifold_guidance_warmup_frac,
                kde_bandwidth_factor=kde_bandwidth_factor,
                kde_max_centers=kde_max_centers,
                sampler_state=state_k,
                context=context,
            )
            local_indices.append(idx)
            local_embeds.append(F.embedding(idx.long(), codebook.to(device=device, dtype=dtype)))

        return self.join_local_indices(local_indices)
