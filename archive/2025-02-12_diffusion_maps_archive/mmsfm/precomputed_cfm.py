from typing import Optional, Tuple

import numpy as np
import torch

from .diffusion_map_sampler import DiffusionMapTrajectorySampler


class PrecomputedConditionalFlowMatcher:
    """
    Lightweight Flow Matcher wrapper around a precomputed trajectory sampler.

    This class mirrors the interface expected by the training loop while
    delegating all sampling logic to the provided sampler. A Gaussian path
    schedule is provided so that x-pred / dynamic v-loss formulations can
    recover dot_sigma / sigma.
    """

    def __init__(
        self,
        sampler: DiffusionMapTrajectorySampler,
        sigma: float,
        diff_ref: str = 'whole',
        schedule_times: Optional[np.ndarray] = None,
        window_mode: str = 'interval',
        triplet_bucket: str = 'random',
        *,
        ratio_min: float = 1e-4,
        ratio_max: float = 100.0,
        t_clip_eps: float = 1e-4,
    ):
        self.sampler = sampler
        self.sigma = float(sigma)
        self.use_miniflow_sigma_t = diff_ref == 'miniflow'
        self.t0 = float(sampler.times[0])
        self.t1 = float(sampler.times[-1])
        self.schedule_times = np.asarray(
            sampler.times if schedule_times is None else schedule_times,
            dtype=float,
        )
        if self.schedule_times.ndim != 1:
            raise ValueError('schedule_times must be a 1D array.')
        if self.schedule_times.shape[0] < 2:
            raise ValueError('schedule_times must contain at least two time points.')
        if np.any(np.diff(self.schedule_times) <= 0):
            raise ValueError('schedule_times must be strictly increasing.')
        self.window_mode = str(window_mode)
        if self.window_mode not in {'interval', 'triplet'}:
            raise ValueError("window_mode must be one of {'interval', 'triplet'}.")
        self.triplet_bucket = str(triplet_bucket)
        if self.triplet_bucket not in {'random', 'left', 'right'}:
            raise ValueError("triplet_bucket must be one of {'random', 'left', 'right'}.")
        self.ratio_min = float(ratio_min)
        self.ratio_max = float(ratio_max)
        self.t_clip_eps = float(t_clip_eps)

    def _infer_interval_bounds(self, t: np.ndarray) -> np.ndarray:
        """Return per-sample (a,b) bounds by bucketizing against schedule_times."""
        idx = np.searchsorted(self.schedule_times, t, side='right') - 1
        idx = np.clip(idx, 0, self.schedule_times.shape[0] - 2)
        a = self.schedule_times[idx]
        b = self.schedule_times[idx + 1]
        return np.stack([a, b], axis=1).astype(np.float32)

    def _infer_triplet_bounds(self, t: np.ndarray) -> np.ndarray:
        """Return per-sample (a,b) bounds using overlapping triplet windows.

        Triplet windows match the TripletAgent convention: window k uses
        endpoints (schedule_times[k], schedule_times[k+2]).

        For a time t in interval i=[t_i, t_{i+1}], there are up to two valid
        triplet windows that contain it: k=i-1 and k=i. We select between them
        using `triplet_bucket` ('random', 'left', or 'right').
        """
        n_knots = self.schedule_times.shape[0]
        if n_knots < 3:
            return self._infer_interval_bounds(t)

        interval_idx = np.searchsorted(self.schedule_times, t, side='right') - 1
        interval_idx = np.clip(interval_idx, 0, n_knots - 2)
        max_start = n_knots - 3

        left_start = interval_idx - 1
        right_start = interval_idx

        if self.triplet_bucket == 'left':
            start_idx = np.clip(left_start, 0, max_start)
        elif self.triplet_bucket == 'right':
            start_idx = np.clip(right_start, 0, max_start)
        else:  # 'random'
            start_idx = right_start.copy()
            # Right edge: last interval can only map to the left-start window.
            right_oob = right_start > max_start
            start_idx[right_oob] = left_start[right_oob]
            # Left edge: first interval can only map to the right-start window (k=0).
            left_oob = left_start < 0
            start_idx[left_oob] = right_start[left_oob]

            both_valid = (~right_oob) & (~left_oob)
            if np.any(both_valid):
                choose_right = np.random.rand(int(np.sum(both_valid))) < 0.5
                start_idx[both_valid] = np.where(
                    choose_right,
                    right_start[both_valid],
                    left_start[both_valid],
                )

        start_idx = np.clip(start_idx, 0, max_start)
        a = self.schedule_times[start_idx]
        b = self.schedule_times[start_idx + 2]
        return np.stack([a, b], axis=1).astype(np.float32)

    def _infer_schedule_bounds(self, t: np.ndarray) -> np.ndarray:
        if self.window_mode == 'triplet':
            return self._infer_triplet_bounds(t)
        return self._infer_interval_bounds(t)

    def sample_location_and_conditional_flow(
        self,
        batch_size: int,
        return_noise: bool = False,
    ):
        """
        Return (t, x_t, u_t[, eps], ab) tuples for Flow/Score Matching training.

        When `return_noise` is True (SB training), we inject Gaussian noise into
        the clean trajectory sample and return the noise used to construct it.
        """
        t, xt_clean, vt = self.sampler.sample(batch_size)

        t_tensor = torch.from_numpy(t).float()
        if self.use_miniflow_sigma_t:
            ab = self._infer_schedule_bounds(t)
            a_tensor = torch.from_numpy(ab[:, 0]).float()
            b_tensor = torch.from_numpy(ab[:, 1]).float()
        else:
            a_tensor = self.t0
            b_tensor = self.t1
            ab = np.stack(
                [
                    np.full_like(t, self.t0, dtype=np.float32),
                    np.full_like(t, self.t1, dtype=np.float32),
                ],
                axis=1,
            )

        sigma_t = self.compute_sigma_t(t_tensor, a_tensor, b_tensor).cpu().numpy()
        sigma_ratio = self.compute_sigma_t_ratio(t_tensor, a_tensor, b_tensor).cpu().numpy()

        eps = None
        if return_noise:
            eps = np.random.randn(*xt_clean.shape).astype(np.float32)
            xt = xt_clean + sigma_t[:, None] * eps
        else:
            xt = xt_clean

        # Conditional flow under Gaussian path: (dot_sigma/sigma)(x - mu_t) + mu_t'
        ut = sigma_ratio[:, None] * (xt - xt_clean) + vt

        xt = xt.astype(np.float32)
        ut = ut.astype(np.float32)

        if return_noise:
            return t.astype(np.float32), xt, ut, eps, ab
        return t.astype(np.float32), xt, ut, ab

    def _canonicalize_bounds(self, t, a, b):
        if a is None or b is None:
            a = self.t0
            b = self.t1
        if not torch.is_tensor(a):
            a = torch.as_tensor(a, dtype=torch.float32, device=t.device)
        if not torch.is_tensor(b):
            b = torch.as_tensor(b, dtype=torch.float32, device=t.device)
        return a, b

    def _clipped_t(self, t: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Keep s in (eps, 1-eps) to avoid blowups near endpoints.
        eps = self.t_clip_eps
        return torch.clamp(t, a + eps, b - eps)

    def compute_sigma_t(self, t, a=None, b=None):
        if not torch.is_tensor(t):
            t = torch.as_tensor(t, dtype=torch.float32)
        a_t, b_t = self._canonicalize_bounds(t, a, b)
        t_eff = self._clipped_t(t, a_t, b_t)
        tscaled = (t_eff - a_t) / (b_t - a_t)
        return self.sigma * torch.sqrt(tscaled * (1 - tscaled))

    def compute_sigma_t_ratio(self, t, a=None, b=None):
        if not torch.is_tensor(t):
            t = torch.as_tensor(t, dtype=torch.float32)
        a_t, b_t = self._canonicalize_bounds(t, a, b)
        t_eff = self._clipped_t(t, a_t, b_t)
        if self.use_miniflow_sigma_t:
            tscaled = (t_eff - a_t) / (b_t - a_t)
            ratio = (1 - 2 * tscaled) / (2 * tscaled * (1 - tscaled) + 1e-8)
            ratio *= 1 / (b_t - a_t)
        else:
            ratio = (1 - 2 * t_eff) / (2 * t_eff * (1 - t_eff) + 1e-8)
        # Clamp by magnitude while preserving sign.
        # Using `torch.clamp(min=...)` would incorrectly force negative ratios
        # to become positive, which can explode x-pred targets (u_t / r).
        eps = self.ratio_min
        sign = torch.sign(ratio)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        ratio = torch.where(ratio.abs() < eps, sign * eps, ratio)
        ratio = torch.clamp(ratio, min=-self.ratio_max, max=self.ratio_max)
        return ratio

    def compute_lambda(self, t, ab=None):
        if ab is not None and self.use_miniflow_sigma_t:
            a = ab[:, 0]
            b = ab[:, 1]
        else:
            a = self.t0
            b = self.t1
        sigma_t = self.compute_sigma_t(t, a, b)
        return 2 * sigma_t / (self.sigma ** 2 + 1e-8)
