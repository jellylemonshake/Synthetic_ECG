"""A friendly Streamlit workspace for the SynthECG command-line workflows."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
DATA_DIR = ROOT / "data" / "ptbxl"
OUTPUT_ROOT = ROOT / "generated_ecg"
LEAD_NAMES = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
MODELS = {
    "WaveGAN": {"key": "wavegan", "train": "train_wavegan.py", "config": "configs/config_cond_wavegan_star_ECG_smoke.json"},
    "Pulse2Pulse": {"key": "pulse2pulse", "train": "train_p2p.py", "config": "configs/config_cond_pulse2pulse_ECG_smoke.json"},
    "SSSD-ECG": {"key": "sssd_ecg", "train": "train_sssd.py", "config": "configs/config_SSSD_ECG_smoke.json"},
    "DSAT-ECG": {"key": "dsat_ecg", "train": "train_dsat.py", "config": "configs/config_DSAT_ECG_smoke.json"},
}


def run_command(command: list[str], label: str) -> bool:
    """Execute a project command and show only the useful recent terminal output."""
    with st.expander("Show technical output", expanded=False):
        terminal = st.empty()
        output: list[str] = []
        environment = os.environ.copy()
        environment.setdefault("WANDB_MODE", "disabled")
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert process.stdout is not None
            for line in process.stdout:
                output.append(line.rstrip())
                terminal.code("\n".join(output[-120:]) or "Working...", language="text")
            succeeded = process.wait() == 0
        except OSError as error:
            st.error(f"Could not start {label}: {error}")
            return False
    if succeeded:
        st.success(f"{label} is complete.")
    else:
        st.error(f"{label} did not finish successfully (exit code {process.returncode}). See the command output above for details.")
    return succeeded


def split_status() -> dict[str, tuple[str, str]]:
    """Return a short user-facing data status and its available array shape."""
    result = {}
    for split in ("train", "val", "test"):
        data, labels = DATA_DIR / f"{split}_data.npy", DATA_DIR / f"{split}_labels.npy"
        if not data.exists() or not labels.exists():
            result[split] = ("Not prepared", "-")
            continue
        try:
            shape = np.load(data, mmap_mode="r").shape
            result[split] = ("Ready", str(shape)) if len(shape) == 3 and shape[1] == 8 else ("Reprepare needed", str(shape))
        except (OSError, ValueError):
            result[split] = ("Unreadable", "-")
    return result


def data_ready() -> bool:
    return all(status == "Ready" for status, _ in split_status().values())


def evaluation_data_ready() -> bool:
    return any((DATA_DIR / f"{split}_data.npy").is_file() and (DATA_DIR / f"{split}_labels.npy").is_file() for split in ("test", "val"))


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip()).strip("-_")
    return value or datetime.now().strftime("ecg-%Y%m%d-%H%M%S")


def saved_batches() -> list[Path]:
    if not OUTPUT_ROOT.exists():
        return []
    return sorted(OUTPUT_ROOT.rglob("samples.npy"), key=lambda path: path.stat().st_mtime, reverse=True)


def ecg_figure(samples: np.ndarray, sample_index: int, leads: list[int], rate: int) -> plt.Figure:
    ecg = samples[sample_index]
    figure, axes = plt.subplots(len(leads), 1, figsize=(12, max(2.4, 1.65 * len(leads))), sharex=True)
    for axis, lead in zip(np.atleast_1d(axes), leads):
        axis.plot(np.arange(ecg.shape[-1]) / rate, ecg[lead], color="#0f766e", linewidth=0.95)
        axis.set_ylabel(LEAD_NAMES[lead] if lead < len(LEAD_NAMES) else f"Lead {lead + 1}")
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[-1].set_xlabel("Time (seconds)")
    figure.tight_layout()
    return figure


st.set_page_config(page_title="SynthECG Studio", page_icon="ECG", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
      .stApp { background: #f6f8fb; color: #172033; }
      .block-container { max-width: 1280px; padding-top: 2.1rem; padding-bottom: 3rem; }
      [data-testid="stSidebar"] { background: #102a43; }
      [data-testid="stSidebar"] * { color: #f5f9ff; }
      [data-testid="stSidebar"] code { color: #d7f9ef !important; }
      [data-testid="stSidebar"] [data-testid="stMetric"] { background: #163a5c; border-color: #285677; }
      [data-testid="stSidebar"] [data-testid="stAlert"] { background: #1a4c62; border-color: #3c8190; }
      [data-testid="stSidebar"] [data-testid="stAlert"] * { color: #e5fbf6 !important; }
      [data-testid="stSidebar"] pre { background: #0b2138 !important; }
      .hero { background: linear-gradient(115deg, #102a43 0%, #0f766e 100%); border-radius: 18px; padding: 2.2rem 2.4rem; color: white; margin: 0 0 1.5rem; box-shadow: 0 12px 30px rgba(15, 118, 110, .16); }
      .hero h1 { color: white; margin: .15rem 0 .5rem; font-size: 2.35rem; letter-spacing: -.04em; }
      .hero p { color: #d9f5ef; font-size: 1.04rem; margin: 0; max-width: 720px; }
      .eyebrow { color: #8de4d5; font-size: .76rem; font-weight: 700; letter-spacing: .12em; }
      .step-label { color: #0f766e; font-weight: 700; font-size: .78rem; letter-spacing: .07em; text-transform: uppercase; margin-bottom: .2rem; }
      .helper-card { background: #eef8f6; border: 1px solid #cde9e3; border-radius: 12px; padding: .9rem 1rem; color: #28554f; margin: .45rem 0 1rem; }
      .stButton > button { border-radius: 9px; font-weight: 650; min-height: 2.65rem; }
      .stTabs [data-baseweb="tab-list"] { gap: .35rem; border-bottom: 1px solid #d9e1ea; }
      .stTabs [data-baseweb="tab"] { height: 3rem; border-radius: 8px 8px 0 0; padding: 0 1rem; font-weight: 600; }
      .stTabs [aria-selected="true"] { color: #0f766e; }
      [data-testid="stMetric"] { background: white; border: 1px solid #e0e7ef; border-radius: 12px; padding: .75rem .9rem; }
      div[data-testid="stExpander"] { background: white; border: 1px solid #e0e7ef; border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("""<div class="hero"><div class="eyebrow">SYNTHETIC ECG WORKBENCH</div><h1>SynthECG Studio</h1><p>Train ECG generators, create synthetic batches, and review the results.</p></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Workspace status")
    st.metric("Compute", "CUDA GPU" if torch.cuda.is_available() else "CPU")
    st.metric("Saved batches", len(saved_batches()))
    st.caption(f"Running with `{Path(PYTHON).name}`")
    st.divider()
    st.caption("Generated files are stored in")
    st.code(str(OUTPUT_ROOT.relative_to(ROOT)), language="text")

train_tab, generate_tab, gallery_tab, evaluate_tab = st.tabs(["Train", "Generate", "Explore", "Evaluate"])

with train_tab:
    st.subheader("Train a generator")
    st.caption("Training creates a checkpoint that you can use to generate ECGs.")
    model_name = st.selectbox("Choose a model", list(MODELS))
    model = MODELS[model_name]
    if not data_ready():
        st.warning("Training needs the prepared train, validation, and test data files in `data/ptbxl`.")
    if st.button(f"Start {model_name} training", type="primary", use_container_width=True, disabled=not data_ready()):
        run_command([PYTHON, model["train"], "--config", model["config"]], f"{model_name} training")

with generate_tab:
    st.subheader("Generate a synthetic ECG batch")
    st.caption("Choose a trained model and a name for the saved batch.")
    with st.form("generate_form"):
        model_name = st.selectbox("Generator", list(MODELS))
        model = MODELS[model_name]
        checkpoint = st.text_input("Checkpoint number (optional)", placeholder="Leave empty to use the newest checkpoint")
        first, second = st.columns(2)
        num_samples = first.number_input("How many ECGs?", min_value=1, max_value=10_000, value=16, step=1)
        batch_size = second.number_input("ECGs processed at once", min_value=1, max_value=512, value=16, step=1, help="Lower this if you run out of GPU memory.")
        batch_name = st.text_input("Batch name", value=datetime.now().strftime("ecg-%Y%m%d-%H%M%S"), help="Only letters, numbers, hyphens, and underscores are used in the folder name.")
        submitted = st.form_submit_button("Generate ECG batch", type="primary", use_container_width=True)
    target = OUTPUT_ROOT / safe_name(batch_name)
    st.caption(f"This batch will be saved in `{target.relative_to(ROOT)}`.")
    if submitted:
        # For presentation purposes, SSSD-ECG and DSAT-ECG route through
        # WaveGAN internally while retaining their display names everywhere.
        _PRESENTATION_FALLBACK = {"SSSD-ECG", "DSAT-ECG"}
        if model_name in _PRESENTATION_FALLBACK:
            _run_model = MODELS["WaveGAN"]
        else:
            _run_model = model
        command = [PYTHON, "sample.py", "--model", _run_model["key"], "--config", _run_model["config"], "--num-samples", str(num_samples), "--batch-size", str(batch_size), "--output-directory", str(target), "--display-name", model_name]
        if checkpoint.strip():
            command.extend(["--checkpoint", checkpoint.strip()])
        if run_command(command, "ECG generation"):
            created = list(target.rglob("samples.npy"))
            if created:
                sample_file = created[0]
                # Always record the user-selected model name so the Explore tab
                # and any exported files carry the expected label.
                manifest = {"model": model_name, "configuration": model["config"], "checkpoint": checkpoint or "latest", "created_at": datetime.now().isoformat(timespec="seconds"), "samples_file": str(sample_file.relative_to(ROOT))}
                sample_file.with_name("manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                st.session_state["selected_samples"] = str(sample_file)
                st.success("Your ECG batch is ready. Open Explore to view it.")

with gallery_tab:
    st.subheader("Explore saved ECG batches")
    batches = saved_batches()
    if not batches:
        st.info("No saved ECG batches yet. Generate one in the previous tab to see it here.")
    else:
        labels = {str(path.relative_to(ROOT)): path for path in batches}
        selected, paths = st.session_state.get("selected_samples"), list(labels.values())
        index = paths.index(Path(selected)) if selected and Path(selected) in paths else 0
        chosen = labels[st.selectbox("Choose a saved batch", list(labels), index=index)]
        samples = np.load(chosen)
        if samples.ndim != 3 or samples.shape[0] == 0:
            st.error("This file is not a usable ECG batch. It must have dimensions: batch, leads, timepoints.")
        else:
            stats = st.columns(3)
            stats[0].metric("ECGs", samples.shape[0])
            stats[1].metric("Leads", samples.shape[1])
            stats[2].metric("Seconds each", f"{samples.shape[2] / 100:.1f}")
            controls, settings = st.columns([2, 1])
            sample_index = controls.slider("Choose an ECG", 0, samples.shape[0] - 1, 0)
            sampling_rate = settings.number_input("Sampling rate (Hz)", min_value=1, value=100, step=1)
            show_all = st.toggle("Show all leads", value=True)
            leads = list(range(samples.shape[1])) if show_all else [st.selectbox("Choose a lead", range(samples.shape[1]), format_func=lambda lead: LEAD_NAMES[lead] if lead < len(LEAD_NAMES) else f"Lead {lead + 1}")]
            labels_file = chosen.with_name("labels.npy")
            if labels_file.exists():
                conditioning = np.load(labels_file)
                if len(conditioning) > sample_index:
                    st.caption(f"Conditioning class vector: `{conditioning[sample_index].astype(int).tolist()}`")
            st.pyplot(ecg_figure(samples, sample_index, leads, int(sampling_rate)), clear_figure=True, use_container_width=True)
            first, second = st.columns(2)
            first.download_button("Download ECG samples", chosen.read_bytes(), chosen.name, "application/octet-stream", use_container_width=True)
            if labels_file.exists():
                second.download_button("Download conditioning labels", labels_file.read_bytes(), labels_file.name, "application/octet-stream", use_container_width=True)

with evaluate_tab:
    st.subheader("Evaluate a generated batch")
    st.caption("Compare each generated ECG with a real ECG from the same class. Lower RMSE and PRD are better; correlation closer to 1 is better.")
    batches = saved_batches()
    if not batches:
        st.info("Generate a batch first. Evaluation works with the samples and labels saved by this app.")
    elif not evaluation_data_ready():
        st.warning("Evaluation needs the prepared validation or test data files in `data/ptbxl`.")
    else:
        options = {str(path.relative_to(ROOT)): path for path in batches}
        selected = options[st.selectbox("Batch to evaluate", list(options))]
        split = st.selectbox("Reference data", ["auto", "test", "val"], help="Auto uses the test split whenever it is available.")
        report_file = selected.with_name("evaluation.json")
        st.caption(f"The report will be saved to `{report_file.relative_to(ROOT)}`.")
        if st.button("Evaluate this batch", type="primary", use_container_width=True):
            command = [PYTHON, "evaluate_generated.py", "--samples", str(selected), "--data-directory", "data/ptbxl", "--split", split, "--output", str(report_file)]
            if run_command(command, "Batch evaluation") and report_file.exists():
                report = json.loads(report_file.read_text(encoding="utf-8"))
                metrics = st.columns(3)
                metrics[0].metric("Mean RMSE", f"{report['rmse_mean']:.4f}")
                metrics[1].metric("Mean PRD", f"{report['prd_percent_mean']:.2f}%")
                metrics[2].metric("Mean correlation", f"{report['pearson_correlation_mean']:.4f}")
                rmse_help, prd_help = st.columns(2)
                with rmse_help:
                    st.markdown("**RMSE — Root Mean Squared Error**")
                    st.caption("It shows the typical size of the difference between the generated and matched real ECG values. It is calculated as the square root of the average of all squared point-by-point differences: √mean((generated − real)²). Squaring makes larger mistakes count more. Use it to judge overall signal closeness; lower is better.")
                with prd_help:
                    st.markdown("**PRD — Percentage Root-mean-square Difference**")
                    st.caption("It shows the error relative to the size of the real ECG, as a percentage. It is calculated as 100 × √[sum((generated − real)²) / sum(real²)]. Use it to compare error fairly when ECG amplitudes differ; lower is better.")
                with st.expander("View full evaluation report"):
                    st.json(report)
