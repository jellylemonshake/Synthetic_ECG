"""Evaluate a saved ``samples.npy`` batch against a prepared PTB-XL split.

Unlike the legacy ``evaluate.py`` workflow, this script consumes the files
written by ``sample.py`` and by the Streamlit application directly.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

MODEL_LEAD_INDICES = [0, 1, 6, 7, 8, 9, 10, 11]  # I, II, V1-V6 in PTB-XL order

def load_array(path: Path, description: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return np.load(path)


def choose_split(data_directory: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    return "test" if (data_directory / "test_data.npy").is_file() else "val"


def paired_reference(fake_labels: np.ndarray, real_data: np.ndarray, real_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic same-class real reference for each generated ECG."""
    fake_classes = np.argmax(fake_labels, axis=1)
    real_classes = np.argmax(real_labels, axis=1)
    references = []
    available = []
    for index, class_id in enumerate(fake_classes):
        candidates = np.flatnonzero(real_classes == class_id)
        if not len(candidates):
            continue
        references.append(real_data[candidates[index % len(candidates)]])
        available.append(index)
    if not references:
        raise ValueError("None of the generated conditioning classes are present in the selected real-data split.")
    return np.asarray(references, dtype=np.float32), np.asarray(available)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a generated ECG batch against prepared PTB-XL data")
    parser.add_argument("--samples", required=True, help="Path to a generated samples.npy file")
    parser.add_argument("--labels", help="Path to labels.npy; defaults to a sibling of --samples")
    parser.add_argument("--data-directory", default="data/ptbxl")
    parser.add_argument("--split", choices=["auto", "val", "test"], default="auto")
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()

    samples_path = Path(args.samples)
    labels_path = Path(args.labels) if args.labels else samples_path.with_name("labels.npy")
    data_directory = Path(args.data_directory)
    fake_data = load_array(samples_path, "Generated samples").astype(np.float32, copy=False)
    fake_labels = load_array(labels_path, "Generated labels").astype(np.float32, copy=False)
    split = choose_split(data_directory, args.split)
    real_data = load_array(data_directory / f"{split}_data.npy", f"{split.title()} data").astype(np.float32, copy=False)
    real_labels = load_array(data_directory / f"{split}_labels.npy", f"{split.title()} labels").astype(np.float32, copy=False)

    if fake_data.ndim != 3 or real_data.ndim != 3:
        raise ValueError("Samples and reference data must have shape (batch, leads, timepoints).")
    if fake_labels.ndim != 2 or len(fake_labels) != len(fake_data):
        raise ValueError("labels.npy must have one 2-D conditioning row per generated sample.")
    if real_labels.ndim != 2 or len(real_labels) != len(real_data):
        raise ValueError("Prepared reference labels do not match reference data.")
    lead_selection = "none"
    if fake_data.shape[1] == 8 and real_data.shape[1] == 12:
        real_data = real_data[:, MODEL_LEAD_INDICES, :]
        lead_selection = "PTB-XL I, II, V1-V6"
    if fake_data.shape[1:] != real_data.shape[1:]:
        raise ValueError(f"Generated shape {fake_data.shape[1:]} does not match reference shape {real_data.shape[1:]}")
    if fake_labels.shape[1] != real_labels.shape[1]:
        raise ValueError("Generated and reference labels have different class counts.")

    reference, used_indices = paired_reference(fake_labels, real_data, real_labels)
    generated = fake_data[used_indices]
    difference = generated - reference
    rmse = np.sqrt(np.mean(np.square(difference), axis=(1, 2)))
    denominator = np.sqrt(np.sum(np.square(reference), axis=(1, 2)))
    prd = 100 * np.sqrt(np.sum(np.square(difference), axis=(1, 2))) / np.maximum(denominator, np.finfo(np.float32).eps)
    correlations = []
    for generated_ecg, reference_ecg in zip(generated, reference):
        correlations.append(float(np.corrcoef(generated_ecg.ravel(), reference_ecg.ravel())[0, 1]))

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "samples_file": str(samples_path),
        "reference_split": split,
        "reference_lead_selection": lead_selection,
        "generated_samples": int(len(fake_data)),
        "class_matched_samples": int(len(generated)),
        "rmse_mean": float(np.mean(rmse)),
        "rmse_std": float(np.std(rmse)),
        "prd_percent_mean": float(np.mean(prd)),
        "pearson_correlation_mean": float(np.nanmean(correlations)),
        "mean_amplitude_difference": float(np.mean(difference)),
        "std_amplitude_difference": float(np.std(generated) - np.std(reference)),
    }
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
        print(f"Report written to {output}")


if __name__ == "__main__":
    main()
