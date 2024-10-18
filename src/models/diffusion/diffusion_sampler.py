from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .denoiser import Denoiser


@dataclass
class DiffusionSamplerConfig:
    num_steps_denoising: int
    sigma_min: float = 2e-3
    sigma_max: float = 5.0
    rho: int = 7
    order: int = 1
    s_churn: float = 0.0
    s_tmin: float = 0.0
    s_tmax: float = float("inf")
    s_noise: float = 1.0
    s_cond: float = 0.0


class DiffusionSampler:
    def __init__(self, denoiser: Denoiser, cfg: DiffusionSamplerConfig) -> None:
        """
        Initializes the DiffusionSampler with a given denoiser and configuration.

        Args:
            denoiser (Denoiser): The denoising model to use.
            cfg (DiffusionSamplerConfig): Configuration parameters for sampling.
        """
        self.denoiser = denoiser
        self.cfg = cfg
        self.device = denoiser.device  # Cache device for reuse

        # Precompute sigmas for the diffusion process
        self.sigmas = build_sigmas(
            num_steps=cfg.num_steps_denoising,
            sigma_min=cfg.sigma_min,
            sigma_max=cfg.sigma_max,
            rho=cfg.rho,
            device=self.device,
        )

    @torch.no_grad()
    def sample(
        self,
        prev_obs: Tensor,
        prev_act: Optional[Tensor] = None
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Generates a sample using the diffusion process.

        Args:
            prev_obs (Tensor): The previous observations tensor of shape (b, t, c, h, w).
            prev_act (Optional[Tensor]): The previous actions tensor, if any.

        Returns:
            Tuple[Tensor, List[Tensor]]: The final sampled tensor and the trajectory of samples.
        """
        device = self.device  # Use cached device

        # Ensure input tensor is on the correct device and dtype
        prev_obs = prev_obs.to(device=device, dtype=torch.float32)
        b, t, c, h, w = prev_obs.size()
        prev_obs = prev_obs.reshape(b, t * c, h, w)

        # Initialize sigma conditioning if required
        sigma_cond = (
            torch.full(
                (b,),
                fill_value=self.cfg.s_cond,
                device=device,
                dtype=torch.float32
            )
            if self.cfg.s_cond > 0
            else None
        )

        # Initialize the latent variable with Gaussian noise
        x = torch.randn(b, c, h, w, device=device, dtype=torch.float32)
        trajectory = [x.clone()]

        # Pre-allocate noise buffer for performance
        noise_buffer = torch.empty_like(x)

        # Compute gamma threshold once if possible
        num_sigmas = len(self.sigmas)
        if num_sigmas > 1:
            gamma_threshold = min(
                self.cfg.s_churn / (num_sigmas - 1),
                2**0.5 - 1
            )
        else:
            gamma_threshold = 0.0

        # Precompute boolean mask for gamma application based on sigma range
        sigma_tensor = self.sigmas[:-1]  # Current sigma
        gamma_mask = (sigma_tensor >= self.cfg.s_tmin) & (sigma_tensor <= self.cfg.s_tmax)
        gamma_values = torch.full_like(sigma_tensor, gamma_threshold)
        gamma_values = torch.where(gamma_mask, gamma_values, torch.zeros_like(gamma_values))

        # Iterate through the sigma schedule
        for idx in range(num_sigmas - 1):
            sigma = self.sigmas[idx]
            next_sigma = self.sigmas[idx + 1]
            gamma = gamma_values[idx]
            sigma_hat = sigma * (gamma + 1)

            if gamma > 0.0:
                # Add controlled noise to the latent variable
                noise_buffer.normal_()
                eps = noise_buffer * self.cfg.s_noise
                # Ensure numerical stability by clamping the argument inside sqrt
                noise_scale = torch.sqrt(torch.clamp(sigma_hat ** 2 - sigma ** 2, min=0.0))
                x = x + eps * noise_scale

            if sigma_cond is not None:
                # Apply noise conditioning to the observations
                # It's safer to avoid modifying prev_obs in-place
                conditioned_obs = self.denoiser.apply_noise(
                    prev_obs,
                    sigma_cond,
                    sigma_offset_noise=0
                )
            else:
                conditioned_obs = prev_obs

            # Denoise the current latent variable
            denoised = self.denoiser.denoise(
                x,
                sigma,
                sigma_cond,
                conditioned_obs,
                prev_act
            )

            # Compute the update term
            d = (x - denoised) / sigma_hat
            dt = next_sigma - sigma_hat

            if self.cfg.order == 1 or next_sigma == 0:
                # Euler's method for first-order integration
                x = x + d * dt
            else:
                # Heun's method for second-order integration
                x_2 = x + d * dt

                if sigma_cond is not None:
                    # Apply noise conditioning to the observations for the next step
                    conditioned_obs_next = self.denoiser.apply_noise(
                        prev_obs,
                        sigma_cond,
                        sigma_offset_noise=0
                    )
                else:
                    conditioned_obs_next = prev_obs

                denoised_2 = self.denoiser.denoise(
                    x_2,
                    next_sigma,
                    sigma_cond,
                    conditioned_obs_next,
                    prev_act
                )
                d_2 = (x_2 - denoised_2) / next_sigma
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt

            # Append the current state to the trajectory
            trajectory.append(x.clone())

        return x, trajectory


def build_sigmas(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: int,
    device: torch.device
) -> Tensor:
    """
    Constructs a tensor of sigma values for the diffusion process.

    Args:
        num_steps (int): Number of denoising steps.
        sigma_min (float): Minimum sigma value.
        sigma_max (float): Maximum sigma value.
        rho (int): Exponent parameter controlling the schedule.
        device (torch.device): Device to create the tensor on.

    Returns:
        Tensor: A 1D tensor of sigma values.
    """
    # Compute inverse rho scaling
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)

    # Create a linear space in the scaled domain
    l = torch.linspace(0, 1, num_steps, device=device, dtype=torch.float32)
    sigmas = (max_inv_rho + l * (min_inv_rho - max_inv_rho)) ** rho

    # Append a zero to signify the final step
    sigmas = torch.cat(
        (sigmas, torch.zeros(1, device=device, dtype=torch.float32)),
        dim=0
    )

    return sigmas