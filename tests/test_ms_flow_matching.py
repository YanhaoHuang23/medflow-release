import unittest

import torch
import torch.nn.functional as F

from models.ms_flow_matching import MultiScaleFlowMatching


class DummyQuantizer:
    def __init__(self, num_codes: int, code_dim: int):
        self.codebook = F.normalize(torch.randn(num_codes, code_dim), dim=-1)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        sim = torch.matmul(x, self.codebook.t())
        return torch.argmax(sim, dim=1)

    def dequantize(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.codebook)


class FlowMatchingTests(unittest.TestCase):
    def setUp(self):
        self.nb_code = [8, 10, 12]
        self.patch_num = [2, 4, 6]
        self.code_dim = 16
        self.batch_size = 3
        self.num_train_samples = 11
        self.quantizers = [DummyQuantizer(nc, self.code_dim) for nc in self.nb_code]

        idx_levels = []
        offset = 0
        for nc, pn in zip(self.nb_code, self.patch_num):
            idx_levels.append(torch.randint(low=0, high=nc, size=(self.batch_size, pn)) + offset)
            offset += nc
        self.global_idx = torch.cat(idx_levels, dim=1)
        self.sample_ids = torch.tensor([0, 1, 2], dtype=torch.long)

    def _build_model(self, **overrides) -> MultiScaleFlowMatching:
        kwargs = dict(
            nb_code=self.nb_code,
            patch_num=self.patch_num,
            code_dim=self.code_dim,
            fm_backbone='dit1d',
            num_train_samples=self.num_train_samples,
            latent_rank=8,
            hidden_dim=32,
            depth=2,
            n_heads=4,
            dropout=0.0,
            train_mixup_prob=0.5,
            train_mixup_alpha=1.0,
            structure_loss_weight=10.0,
            senior_mean_reg_weight=0.1,
            senior_std_reg_weight=0.1,
            senior_sample_noise_std=0.01,
        )
        kwargs.update(overrides)
        return MultiScaleFlowMatching(**kwargs)

    def test_split_and_join_indices(self):
        model = self._build_model()
        level_indices = model.split_global_indices(self.global_idx)
        self.assertEqual(len(level_indices), len(self.patch_num))
        for idx, pn, nc in zip(level_indices, self.patch_num, self.nb_code):
            self.assertEqual(tuple(idx.shape), (self.batch_size, pn))
            self.assertTrue(torch.all(idx >= 0))
            self.assertTrue(torch.all(idx < nc))

        rebuilt = model.join_local_indices(level_indices)
        self.assertTrue(torch.equal(rebuilt, self.global_idx))

    def test_senior_loss_contains_structure_terms(self):
        model = self._build_model()
        loss, metrics = model.compute_loss(
            code_indices=self.global_idx,
            quantizers=self.quantizers,
            sample_ids=self.sample_ids,
        )
        self.assertTrue(torch.isfinite(loss).item())
        self.assertIn('loss_total', metrics)
        for k in range(len(self.patch_num)):
            self.assertIn(f'loss_s{k}', metrics)
            self.assertIn(f'ce_s{k}', metrics)
            self.assertIn(f'acc_s{k}', metrics)
            self.assertIn(f'mean_s{k}', metrics)
            self.assertIn(f'std_s{k}', metrics)
            self.assertIn(f'structure_s{k}', metrics)

    def test_conditional_loss_and_sampling_require_labels(self):
        model = self._build_model(num_classes=2)
        labels = torch.tensor([0, 1, 0], dtype=torch.long)
        loss, metrics = model.compute_loss(
            code_indices=self.global_idx,
            quantizers=self.quantizers,
            sample_ids=self.sample_ids,
            labels=labels,
        )
        self.assertTrue(torch.isfinite(loss).item())
        self.assertIn('loss_total', metrics)

        with self.assertRaises(ValueError):
            model.compute_loss(
                code_indices=self.global_idx,
                quantizers=self.quantizers,
                sample_ids=self.sample_ids,
            )

        model.eval()
        sampled = model.sample_token_indices(
            batch_size=2,
            quantizers=self.quantizers,
            flow_steps=4,
            solver='euler',
            sample_temperature=0.9,
            noise_scale=1.0,
            senior_sampler='mixup',
            kde_bandwidth_factor=1.0,
            kde_max_centers=8,
            device=torch.device('cpu'),
            labels=torch.tensor([0, 1], dtype=torch.long),
        )
        self.assertEqual(tuple(sampled.shape), (2, sum(self.patch_num)))

    def test_class_token_type_conditioning_loss_and_sampling(self):
        model = self._build_model(num_classes=2, token_type_conditioning='class')
        labels = torch.tensor([0, 1, 0], dtype=torch.long)
        loss, metrics = model.compute_loss(
            code_indices=self.global_idx,
            quantizers=self.quantizers,
            sample_ids=self.sample_ids,
            labels=labels,
        )
        self.assertTrue(torch.isfinite(loss).item())
        self.assertIn('loss_total', metrics)

        model.eval()
        sampled = model.sample_token_indices(
            batch_size=2,
            quantizers=self.quantizers,
            flow_steps=4,
            solver='euler',
            sample_temperature=0.9,
            noise_scale=1.0,
            senior_sampler='mixup',
            kde_bandwidth_factor=1.0,
            kde_max_centers=8,
            device=torch.device('cpu'),
            labels=torch.tensor([0, 1], dtype=torch.long),
        )
        self.assertEqual(tuple(sampled.shape), (2, sum(self.patch_num)))

    def test_class_scale_token_type_conditioning_loss_and_sampling(self):
        model = self._build_model(num_classes=2, token_type_conditioning='class_scale')
        labels = torch.tensor([0, 1, 0], dtype=torch.long)
        loss, metrics = model.compute_loss(
            code_indices=self.global_idx,
            quantizers=self.quantizers,
            sample_ids=self.sample_ids,
            labels=labels,
        )
        self.assertTrue(torch.isfinite(loss).item())
        self.assertIn('loss_total', metrics)

        model.eval()
        sampled = model.sample_token_indices(
            batch_size=2,
            quantizers=self.quantizers,
            flow_steps=4,
            solver='euler',
            sample_temperature=0.9,
            noise_scale=1.0,
            senior_sampler='mixup',
            kde_bandwidth_factor=1.0,
            kde_max_centers=8,
            device=torch.device('cpu'),
            labels=torch.tensor([0, 1], dtype=torch.long),
        )
        self.assertEqual(tuple(sampled.shape), (2, sum(self.patch_num)))

    def test_train_mixup_state_is_shared_across_scales(self):
        model = self._build_model(train_mixup_prob=1.0)
        model.train()
        loss, _ = model.compute_loss(
            code_indices=self.global_idx,
            quantizers=self.quantizers,
            sample_ids=self.sample_ids,
        )
        self.assertTrue(torch.isfinite(loss).item())

        states = [scale_model._last_train_mixup_state for scale_model in model.scale_models]
        self.assertTrue(all(state is not None for state in states))
        self.assertTrue(all(bool(state['use_mixup']) for state in states))

        base_state = states[0]
        for state in states[1:]:
            self.assertEqual(bool(state['use_mixup']), bool(base_state['use_mixup']))
            self.assertAlmostEqual(float(state['lam']), float(base_state['lam']), places=7)
            self.assertTrue(torch.equal(state['perm'], base_state['perm']))

    def test_senior_sampling_requires_explicit_sampler(self):
        model = self._build_model()
        model.eval()
        with self.assertRaises(ValueError):
            model.sample_token_indices(
                batch_size=2,
                quantizers=self.quantizers,
                flow_steps=4,
                solver='euler',
                sample_temperature=0.9,
                noise_scale=1.0,
                device=torch.device('cpu'),
            )

    def test_senior_sampling_is_seed_reproducible_for_mixup_and_kde(self):
        for sampler in ['mixup', 'kde']:
            model = self._build_model()
            model.eval()
            torch.manual_seed(2026)
            idx1 = model.sample_token_indices(
                batch_size=2,
                quantizers=self.quantizers,
                flow_steps=6,
                solver='euler',
                sample_temperature=0.9,
                noise_scale=1.0,
                senior_sampler=sampler,
                kde_bandwidth_factor=1.0,
                kde_max_centers=8,
                device=torch.device('cpu'),
            )
            torch.manual_seed(2026)
            idx2 = model.sample_token_indices(
                batch_size=2,
                quantizers=self.quantizers,
                flow_steps=6,
                solver='euler',
                sample_temperature=0.9,
                noise_scale=1.0,
                senior_sampler=sampler,
                kde_bandwidth_factor=1.0,
                kde_max_centers=8,
                device=torch.device('cpu'),
            )
            self.assertTrue(torch.equal(idx1, idx2))

    def test_mixup_sampling_state_is_shared_across_scales(self):
        model = self._build_model()
        model.eval()
        model.sample_token_indices(
            batch_size=2,
            quantizers=self.quantizers,
            flow_steps=4,
            solver='euler',
            sample_temperature=0.9,
            noise_scale=1.0,
            senior_sampler='mixup',
            kde_bandwidth_factor=1.0,
            kde_max_centers=8,
            device=torch.device('cpu'),
        )

        states = [scale_model._last_sampling_state for scale_model in model.scale_models]
        self.assertTrue(all(state is not None for state in states))
        self.assertTrue(all(state['sampler'] == 'mixup' for state in states))

        base_state = states[0]
        for state in states[1:]:
            self.assertTrue(torch.equal(state['idx_a'], base_state['idx_a']))
            self.assertTrue(torch.equal(state['idx_b'], base_state['idx_b']))
            self.assertTrue(torch.equal(state['alpha'], base_state['alpha']))

    def test_kde_sampling_state_is_shared_across_scales(self):
        model = self._build_model()
        model.eval()
        model.sample_token_indices(
            batch_size=2,
            quantizers=self.quantizers,
            flow_steps=4,
            solver='euler',
            sample_temperature=0.9,
            noise_scale=1.0,
            senior_sampler='kde',
            kde_bandwidth_factor=1.0,
            kde_max_centers=8,
            device=torch.device('cpu'),
        )

        states = [scale_model._last_sampling_state for scale_model in model.scale_models]
        self.assertTrue(all(state is not None for state in states))
        self.assertTrue(all(state['sampler'] == 'kde' for state in states))

        base_state = states[0]
        for state in states[1:]:
            self.assertTrue(torch.equal(state['center_indices'], base_state['center_indices']))

    def test_sampling_returns_valid_ranges(self):
        for sampler in ['mixup', 'kde']:
            model = self._build_model()
            model.eval()
            sampled = model.sample_token_indices(
                batch_size=2,
                quantizers=self.quantizers,
                flow_steps=4,
                solver='heun',
                sample_temperature=0.9,
                noise_scale=1.0,
                senior_sampler=sampler,
                kde_bandwidth_factor=1.0,
                kde_max_centers=8,
                device=torch.device('cpu'),
            )
            self.assertEqual(tuple(sampled.shape), (2, sum(self.patch_num)))

            local = model.split_global_indices(sampled)
            for idx, nc in zip(local, self.nb_code):
                self.assertTrue(torch.all(idx >= 0))
                self.assertTrue(torch.all(idx < nc))


if __name__ == '__main__':
    unittest.main()
