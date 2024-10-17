from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

from coroutines import coroutine
from models.diffusion import Denoiser, DiffusionSampler, DiffusionSamplerConfig
from models.rew_end_model import RewEndModel

ResetOutput = Tuple[torch.FloatTensor, Dict[str, Any]]
StepOutput = Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]
InitialCondition = Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]


@dataclass
class WorldModelEnvConfig:
    horizon: int
    num_batches_to_preload: int
    diffusion_sampler_next_obs: DiffusionSamplerConfig
    diffusion_sampler_upsampling: Optional[DiffusionSamplerConfig] = None


class WorldModelEnv:
    def __init__(
        self,
        denoiser: Denoiser,
        upsampler: Optional[Denoiser],
        rew_end_model: Optional[RewEndModel],
        spawn_dir: Path,
        num_envs: int,
        seq_length: int,
        cfg: WorldModelEnvConfig,
        return_denoising_trajectory: bool = False,
    ) -> None:
        assert num_envs == 1  # for csgo only
        self.sampler_next_obs = DiffusionSampler(denoiser, cfg.diffusion_sampler_next_obs)
        self.sampler_upsampling = None if upsampler is None else DiffusionSampler(upsampler, cfg.diffusion_sampler_upsampling)
        self.rew_end_model = rew_end_model
        self.horizon = cfg.horizon
        self.return_denoising_trajectory = return_denoising_trajectory
        self.num_envs = num_envs
        self.generator_init = self.make_generator_init(spawn_dir, cfg.num_batches_to_preload)
        
        self.n_skip_next_obs = seq_length - self.sampler_next_obs.denoiser.cfg.inner_model.num_steps_conditioning
        self.n_skip_upsampling = None if upsampler is None else seq_length - self.sampler_upsampling.denoiser.cfg.inner_model.num_steps_conditioning

    @property
    def device(self) -> torch.device:
        return self.sampler_next_obs.denoiser.device

    @torch.no_grad()
    def reset(self, **kwargs) -> ResetOutput:
        # Gebruik asynchrone I/O of laad gegevens in batches
        obs, obs_full_res, act, next_act, (hx, cx) = self.generator_init.send(self.num_envs)
        self.obs_buffer = obs
        self.act_buffer = act
        self.next_act = next_act[0]
        self.obs_full_res_buffer = obs_full_res
        self.ep_len = torch.zeros(self.num_envs, dtype=torch.long, device=obs.device)
        self.hx_rew_end = hx
        self.cx_rew_end = cx
        obs_to_return = self.obs_buffer[:, -1] if self.sampler_upsampling is None else self.obs_full_res_buffer[:, -1]
        return obs_to_return, {}

    @torch.no_grad()
    def step(self, act: torch.LongTensor) -> StepOutput:
        # Zorg ervoor dat alle tensorbewerkingen op de GPU worden uitgevoerd
        act = act.to(self.device)
        self.act_buffer[:, -1] = act

        next_obs, denoising_trajectory = self.predict_next_obs()

        if self.sampler_upsampling is not None:
            next_obs_full, denoising_trajectory_upsampling = self.upsample_next_obs(next_obs)

        if self.rew_end_model is not None:
            rew, end = self.predict_rew_end(next_obs.unsqueeze(1))
        else:
            rew = torch.zeros(next_obs.size(0), dtype=torch.float32, device=self.device)
            end = torch.zeros(next_obs.size(0), dtype=torch.int64, device=self.device)
        
        # Gebruik in-place bewerkingen waar mogelijk
        self.ep_len.add_(1)
        trunc = (self.ep_len >= self.horizon).long()

        self.obs_buffer = self.obs_buffer.roll(-1, dims=1)
        self.act_buffer = self.act_buffer.roll(-1, dims=1)
        self.obs_buffer[:, -1] = next_obs

        if self.sampler_upsampling is not None:
            self.obs_full_res_buffer = self.obs_full_res_buffer.roll(-1, dims=1)
            self.obs_full_res_buffer[:, -1] = next_obs_full

        info = {}
        if self.return_denoising_trajectory:
            info["denoising_trajectory"] = torch.stack(denoising_trajectory, dim=1)
            
        if self.sampler_upsampling is not None:
            info["obs_low_res"] = next_obs
            if self.return_denoising_trajectory:
                info["denoising_trajectory_upsampling"] = torch.stack(denoising_trajectory_upsampling, dim=1)

        obs_to_return = self.obs_buffer[:, -1] if self.sampler_upsampling is None else self.obs_full_res_buffer[:, -1]
        return obs_to_return, rew, end, trunc, info

    @torch.no_grad()
    def predict_next_obs(self) -> Tuple[Tensor, List[Tensor]]:
        # Zorg ervoor dat de invoer op de juiste apparaat is
        obs_buffer = self.obs_buffer[:, self.n_skip_next_obs:].to(self.device)
        act_buffer = self.act_buffer[:, self.n_skip_next_obs:].to(self.device)
        return self.sampler_next_obs.sample(obs_buffer, act_buffer)

    @torch.no_grad()
    def upsample_next_obs(self, next_obs: Tensor) -> Tuple[Tensor, List[Tensor]]:
        low_res = F.interpolate(next_obs, scale_factor=self.sampler_upsampling.denoiser.cfg.upsampling_factor, mode="bilinear").unsqueeze(1) #switch to bilinear instead of bicubic for mps support
        return self.sampler_upsampling.sample(torch.cat((self.obs_full_res_buffer[:, self.n_skip_upsampling:], low_res), dim=1), None)

    @torch.no_grad()
    def predict_rew_end(self, next_obs: Tensor) -> Tuple[Tensor, Tensor]:
        logits_rew, logits_end, (self.hx_rew_end, self.cx_rew_end) = self.rew_end_model.predict_rew_end(
            self.obs_buffer[:, -1:],
            self.act_buffer[:, -1:],
            next_obs,
            (self.hx_rew_end, self.cx_rew_end),
        )
        rew = Categorical(logits=logits_rew).sample().squeeze(1) - 1.0  # in {-1, 0, 1}
        end = Categorical(logits=logits_end).sample().squeeze(1)
        return rew, end

    @coroutine
    def make_generator_init(
        self,
        spawn_dir: Path,
        num_batches_to_preload: int,
    ) -> Generator[InitialCondition, None, None]:
        num_dead = yield

        spawn_dirs = cycle(sorted(list(spawn_dir.iterdir())))

        while True:
            obs_, obs_full_res_, act_, next_act_, hx_, cx_ = [], [], [], [], [], []
            for _ in range(num_batches_to_preload):
                d = next(spawn_dirs)
                obs = torch.tensor(np.load(d / "low_res.npy"), dtype=torch.float32, device=self.device) / 255.0 * 2 - 1
                obs_full_res = torch.tensor(np.load(d / "full_res.npy"), dtype=torch.float32, device=self.device) / 255.0 * 2 - 1
                act = torch.tensor(np.load(d / "act.npy"), dtype=torch.long, device=self.device)
                next_act = torch.tensor(np.load(d / "next_act.npy"), dtype=torch.long, device=self.device)

                obs_.append(obs.unsqueeze(0))
                obs_full_res_.append(obs_full_res.unsqueeze(0))
                act_.append(act.unsqueeze(0))
                next_act_.append(next_act.unsqueeze(0))

            # Batch the data
            obs_ = torch.cat(obs_, dim=0)
            obs_full_res_ = torch.cat(obs_full_res_, dim=0) if self.sampler_upsampling else None
            act_ = torch.cat(act_, dim=0)
            next_act_ = torch.cat(next_act_, dim=0)

            if self.rew_end_model:
                with torch.no_grad():
                    _, _, (hx_batch, cx_batch) = self.rew_end_model.predict_rew_end(
                        obs_[:, :-1], act_[:, :-1], obs_[:, 1:]
                    )
                hx_ = hx_batch.unsqueeze(0)
                cx_ = cx_batch.unsqueeze(0)
            else:
                hx_, cx_ = None, None

            while num_dead > 0:
                batch_size = min(num_dead, obs_.size(0))
                obs = obs_[:batch_size]
                act = act_[:batch_size]
                next_act = next_act_[:batch_size]
                obs_full_res = obs_full_res_[:batch_size] if self.sampler_upsampling else None
                hx = hx_[:batch_size] if self.rew_end_model else None
                cx = cx_[:batch_size] if self.rew_end_model else None

                yield obs, obs_full_res, act, next_act, (hx, cx)
                obs_ = obs_[batch_size:]
                act_ = act_[batch_size:]
                next_act_ = next_act_[batch_size:]
                if self.sampler_upsampling:
                    obs_full_res_ = obs_full_res_[batch_size:]
                if self.rew_end_model:
                    hx_ = hx_[batch_size:]
                    cx_ = cx_[batch_size:]
                num_dead -= batch_size
