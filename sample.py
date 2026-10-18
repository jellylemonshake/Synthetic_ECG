"""Generate ECG samples from a trained SynthECG checkpoint.

This intentionally uses one process and selects CPU when CUDA is unavailable;
the former NCCL-only implementation could not run on ordinary workstations.
"""
import argparse
import importlib
import json
import os
from pathlib import Path

import numpy as np
import torch


MODEL_CONFIGS = {
    "wavegan": ("WaveGAN", "configs/config_cond_wavegan_star_ECG.json",
                "models.cond_wavegan_star", "CondWaveGANGenerator"),
    "pulse2pulse": ("Pulse2Pulse", "configs/config_cond_pulse2pulse_ECG.json",
                    "models.cond_pulse2pulse", "CondP2PGenerator"),
    "sssd_ecg": ("SSSD_ECG", "configs/config_SSSD_ECG.json", "models.SSSD_ECG", "SSSD_ECG"),
    "dsat_ecg": ("DSAT_ECG", "configs/config_DSAT_ECG.json", "models.DSAT_ECG", "DSAT_ECG"),
}


def natural_path_key(path):
    """Order run directories and numbered checkpoints numerically when possible."""
    try:
        return (0, int(path.stem.split("_")[-1]))
    except ValueError:
        return (1, path.stem)


def config_and_checkpoint(model_key, checkpoint, config_override=None):
    display_name, config_path, module_name, class_name = MODEL_CONFIGS[model_key]
    config_path = config_override or config_path
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    root = Path(config["experiment"]["ckpt_directory"])
    run_dirs = sorted(root.glob("run_*"), key=natural_path_key)
    if not run_dirs:
        raise FileNotFoundError(f"No runs found under {root}. Train {display_name} first.")
    run_dir = run_dirs[-1]
    candidates = sorted(run_dir.glob("*.pkl"), key=natural_path_key)
    if checkpoint is not None:
        candidates = [path for path in candidates if path.stem == str(checkpoint)]
    if not candidates:
        raise FileNotFoundError(f"No matching .pkl checkpoint found in {run_dir}")
    return display_name, config, module_name, class_name, run_dir, candidates[-1]


def build_model(model_key, config, module_name, class_name, device):
    if model_key in {"wavegan", "pulse2pulse"}:
        kwargs = {**config["model"], **config["generator"]}
    else:
        from utils.utils import calc_diffusion_hyperparams
        kwargs = {**config["model"], **calc_diffusion_hyperparams(**config["diffusion"])}
    model = getattr(importlib.import_module(module_name), class_name)(**kwargs).to(device)
    return model


def load_weights(model, path, model_key, device):
    state = torch.load(path, map_location=device, weights_only=False)
    if model_key in {"wavegan", "pulse2pulse"}:
        state = state.get("modelG_state_dict", state) if isinstance(state, dict) else state[0]
    elif isinstance(state, dict):
        state = state.get("model_state_dict", state)
    model.load_state_dict(state)


def generate(model, model_key, labels, batch_size):
    chunks = []
    for batch_labels in labels.split(batch_size):
        result = model.sample_trained_model(batch_labels.shape[0], labels=batch_labels)
        chunks.append(result[0] if isinstance(result, tuple) else result)
    return torch.cat(chunks)


def main():
    parser = argparse.ArgumentParser(description="Generate samples from a trained ECG generator")
    parser.add_argument("--model", default="all", choices=[*MODEL_CONFIGS, "all"], type=str.lower)
    parser.add_argument("--checkpoint", help="Checkpoint epoch/iteration; defaults to the latest.")
    parser.add_argument("--config", help="Configuration used for training (required for a non-default run).")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-directory", default="samples")
    parser.add_argument("--display-name", default=None, help="Override the model folder name shown in the output path.")
    args = parser.parse_args()
    if args.num_samples < 1 or args.batch_size < 1:
        parser.error("--num-samples and --batch-size must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_keys = MODEL_CONFIGS if args.model == "all" else [args.model]
    for key in model_keys:
        try:
            name, config, module_name, class_name, run_dir, checkpoint = config_and_checkpoint(key, args.checkpoint, args.config)
            model = build_model(key, config, module_name, class_name, device)
            load_weights(model, checkpoint, key, device)
            model.eval()
            classes = config["model"].get("label_embed_classes")
            if classes is None:
                classes = config["generator"]["label_embed_classes"]
            labels = torch.eye(classes, device=device)[torch.arange(args.num_samples, device=device) % classes]
            with torch.inference_mode():
                samples = generate(model, key, labels, args.batch_size).cpu().numpy()
            folder_name = args.display_name if args.display_name else name
            output = Path(args.output_directory) / folder_name / run_dir.name / checkpoint.stem
            output.mkdir(parents=True, exist_ok=True)
            np.save(output / "samples.npy", samples)
            np.save(output / "labels.npy", labels.cpu().numpy())
            print(f"{name}: wrote {samples.shape} to {output}")
        except FileNotFoundError as error:
            if args.model == "all":
                print(f"Skipping {key}: {error}")
            else:
                raise


if __name__ == "__main__":
    main()
