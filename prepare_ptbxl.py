"""Create the NumPy ECG splits consumed by the generative-model trainers.

The original PTB-XL files store signals as ``(time, leads)`` WFDB records.
This script selects the five super-diagnostic classes, applies the official
PTB-XL folds, standardises from the training split only, and writes the
channel-first arrays expected by this repository.
"""
import argparse
import ast
import os

import numpy as np
import pandas as pd
import wfdb
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

MODEL_LEAD_INDICES = [0, 1, 6, 7, 8, 9, 10, 11]  # I, II, V1-V6 in PTB-XL order

def diagnostic_classes(codes, statements):
    classes = set()
    for code in codes:
        if code in statements.index and statements.loc[code, "diagnostic"] == 1.0:
            value = statements.loc[code, "diagnostic_class"]
            if pd.notna(value):
                classes.add(value)
    return sorted(classes)


def load_split(frame, root):
    signals = []
    for filename in frame.filename_lr:
        signal, _ = wfdb.rdsamp(os.path.join(root, filename))
        # SynthECG configurations use the eight clinically standard leads
        # I, II, and V1-V6 rather than all 12 PTB-XL channels.
        signals.append(signal[:, MODEL_LEAD_INDICES].T.astype(np.float32, copy=False))
    return np.stack(signals)


def main():
    parser = argparse.ArgumentParser(description="Prepare PTB-XL arrays for SynthECG")
    parser.add_argument("--data-directory", default="data/ptbxl")
    parser.add_argument("--max-per-split", type=int, default=None,
                        help="Optional deterministic cap per split (useful for smoke tests).")
    args = parser.parse_args()

    root = args.data_directory
    database = pd.read_csv(os.path.join(root, "ptbxl_database.csv"), index_col="ecg_id")
    statements = pd.read_csv(os.path.join(root, "scp_statements.csv"), index_col=0)
    database["scp_codes"] = database.scp_codes.map(ast.literal_eval)
    database["labels"] = database.scp_codes.map(lambda codes: diagnostic_classes(codes, statements))
    database = database[database.labels.map(len) > 0].copy()

    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(database.labels)
    if labels.shape[1] != 5:
        raise RuntimeError(f"Expected 5 super-diagnostic classes, found {labels.shape[1]}: {mlb.classes_}")
    print("Classes:", ", ".join(mlb.classes_))

    split_folds = {"train": database.strat_fold <= 8, "val": database.strat_fold == 9,
                   "test": database.strat_fold == 10}
    splits = {}
    for name, mask in split_folds.items():
        indices = np.flatnonzero(mask.to_numpy())
        if args.max_per_split is not None:
            indices = indices[:args.max_per_split]
        splits[name] = (database.iloc[indices], labels[indices].astype(np.float32))

    train_raw = load_split(splits["train"][0], root)
    scaler = StandardScaler().fit(train_raw.reshape(-1, 1))
    for name, (frame, y) in splits.items():
        raw = train_raw if name == "train" else load_split(frame, root)
        x = scaler.transform(raw.reshape(-1, 1)).reshape(raw.shape).astype(np.float32)
        np.save(os.path.join(root, f"{name}_data.npy"), x)
        np.save(os.path.join(root, f"{name}_labels.npy"), y)
        print(f"{name}: data={x.shape}, labels={y.shape}")


if __name__ == "__main__":
    main()
