## Start the web app (Windows PowerShell)

py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt

python -m streamlit run app.py
```


```powershell
# Short verification runs.
python train_wavegan.py --config configs/config_cond_wavegan_star_ECG_smoke.json
python train_p2p.py --config configs/config_cond_pulse2pulse_ECG_smoke.json
python train_sssd.py --config configs/config_SSSD_ECG_smoke.json
python train_dsat.py --config configs/config_DSAT_ECG_smoke.json

# Generate a batch using the latest WaveGAN checkpoint.
python sample.py --model wavegan --num-samples 16 --batch-size 16 --output-directory generated_ecg/my-batch
```

The sampler saves each batch as:

```text
generated_ecg/my-batch/<Model>/run_<n>/<checkpoint>/
  samples.npy
  labels.npy
```