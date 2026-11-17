# DCGAN Streamlit Image Generator

This deployment app loads the supplied Keras DCGAN generator and creates 64×64 RGB images from random latent vectors.

## Files

- `app.py` — Streamlit inference application.
- `dcgan_generator.keras` — trained generator (required).
- `dcgan_discriminator.keras` — supplied discriminator, retained for future training/evaluation; not required by this inference app.
- `requirements.txt` — Python dependencies.
- `.python-version` — requests Python 3.11.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload all files from this folder, keeping the two `.keras` files in the repository root beside `app.py`.
2. In Streamlit Community Cloud, create a new app and select this repository.
3. Set the main file path to `app.py`.
4. In Advanced settings, select Python 3.11 if the option is available, then deploy.

Do not upload a local dataset for this image-generation app: generating images from the already-trained generator does not require the training dataset. The original training script (`dcgen.py`) was not included in the files available for this repair, so this package provides deployment of the trained generator for inference, not model retraining.

## Local run

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
streamlit run app.py
```

The generator supplied with this project has input shape `(None, 100)` and output shape `(None, 64, 64, 3)`. The app checks these assumptions at startup.
