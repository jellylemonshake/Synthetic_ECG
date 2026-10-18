"""Shared helpers used by the ECG training scripts.

The original project kept the diffusion math and checkpoint handling in this
module.  Keeping them here gives the SSSD and DSAT implementations one
device-safe implementation that also works on CPU-only machines.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def calc_diffusion_step_embedding(
    diffusion_steps: torch.Tensor, diffusion_step_embed_dim_in: int
) -> torch.Tensor:
    """Encode each diffusion timestep with sinusoidal features."""
    if diffusion_step_embed_dim_in < 2 or diffusion_step_embed_dim_in % 2:
        raise ValueError("diffusion_step_embed_dim_in must be an even integer of at least 2.")

    half_dim = diffusion_step_embed_dim_in // 2
    frequencies = torch.exp(
        torch.arange(half_dim, device=diffusion_steps.device, dtype=torch.float32)
        * -(torch.log(torch.tensor(10000.0, device=diffusion_steps.device)) / (half_dim - 1))
    )
    angles = diffusion_steps.to(dtype=torch.float32) * frequencies
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)


def calc_diffusion_hyperparams(T: int, beta_0: float, beta_T: float) -> dict[str, Any]:
    """Create the linear diffusion schedule used by SSSD and DSAT."""
    beta = torch.linspace(beta_0, beta_T, T)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    beta_tilde = beta.clone()
    if T > 1:
        beta_tilde[1:] *= (1 - alpha_bar[:-1]) / (1 - alpha_bar[1:])
    return {
        "T": T,
        "Beta": beta,
        "Alpha": alpha,
        "Alpha_bar": alpha_bar,
        "Sigma": torch.sqrt(beta_tilde),
    }


def training_loss_label(net: torch.nn.Module, loss_fn: torch.nn.Module, batch, diffusion_hyperparams: dict[str, Any]) -> torch.Tensor:
    """Add noise at a random timestep and train the model to predict it."""
    audio, labels = batch
    batch_size = audio.shape[0]
    diffusion_steps = torch.randint(
        diffusion_hyperparams["T"], (batch_size, 1), device=audio.device
    )
    noise = torch.randn_like(audio)
    alpha_bar = diffusion_hyperparams["Alpha_bar"][diffusion_steps].view(batch_size, 1, 1)
    noised_audio = torch.sqrt(alpha_bar) * audio + torch.sqrt(1 - alpha_bar) * noise
    predicted_noise = net(noised_audio, labels, diffusion_steps)
    return loss_fn(predicted_noise, noise)


def config_filename(project: str, run_name: str) -> str:
    """Return a filesystem-safe name for the configuration saved with a run."""
    safe_project = "".join(char if char.isalnum() or char in "-_" else "_" for char in project)
    safe_run = "".join(char if char.isalnum() or char in "-_" else "_" for char in run_name)
    return f"{safe_project}_{safe_run}_config.json"


def find_run(config: dict[str, Any], experiment: dict[str, Any]) -> tuple[int, Any]:
    """Resume the latest run or create a fresh numbered run when requested."""
    checkpoint_root = Path(experiment["ckpt_directory"])
    run_numbers = []
    if checkpoint_root.is_dir():
        for path in checkpoint_root.glob("run_*"):
            try:
                run_numbers.append(int(path.name.removeprefix("run_")))
            except ValueError:
                continue
    latest_run = max(run_numbers, default=0)
    start_fresh = config.get("train", {}).get("ckpt_iter") == -1
    return (latest_run + 1 if start_fresh else max(latest_run, 1)), experiment.get("id")


def _latest_checkpoint(checkpoint_directory: str | os.PathLike[str]) -> int:
    try:
        values = [
            int(path.stem)
            for path in Path(checkpoint_directory).glob("*.pkl")
            if path.stem.isdigit()
        ]
    except OSError:
        return -1
    return max(values, default=-1)


def save_checkpoint(checkpoint_directory: str | os.PathLike[str], iteration: int, net, optimizer) -> Path:
    """Save model and optimizer state in the format used by the sampling code."""
    output = Path(checkpoint_directory) / f"{iteration}.pkl"
    if isinstance(net, tuple):
        generator, discriminator = net
        optimizer_generator, optimizer_discriminator = optimizer
        state = {
            "modelG_state_dict": generator.state_dict(),
            "modelD_state_dict": discriminator.state_dict(),
            "optimizerG_state_dict": optimizer_generator.state_dict(),
            "optimizerD_state_dict": optimizer_discriminator.state_dict(),
        }
    else:
        state = {"model_state_dict": net.state_dict(), "optimizer_state_dict": optimizer.state_dict()}
    torch.save(state, output)
    return output


def load_checkpoint(checkpoint_directory: str | os.PathLike[str], checkpoint_iteration, net, optimizer):
    """Load a requested checkpoint, returning a clean first-run state if none exists."""
    iteration = _latest_checkpoint(checkpoint_directory) if checkpoint_iteration == "max" else int(checkpoint_iteration)
    if iteration < 0:
        return -1, net, optimizer

    path = Path(checkpoint_directory) / f"{iteration}.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(net, tuple):
        generator, discriminator = net
        optimizer_generator, optimizer_discriminator = optimizer
        generator.load_state_dict(state["modelG_state_dict"])
        discriminator.load_state_dict(state["modelD_state_dict"])
        if "optimizerG_state_dict" in state:
            optimizer_generator.load_state_dict(state["optimizerG_state_dict"])
        if "optimizerD_state_dict" in state:
            optimizer_discriminator.load_state_dict(state["optimizerD_state_dict"])
    else:
        net.load_state_dict(state.get("model_state_dict", state))
        if isinstance(state, dict) and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
    return iteration, net, optimizer
