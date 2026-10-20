# SAGE-EEG Reconstruction Code

Model definitions, hyperparameters, training, checkpoint testing and evaluation for SAGE-EEG and twelve baselines. Python 3.12, CUDA-enabled PyTorch 2.8 and RTX 5070 Ti were used for validation.

## Install

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

## Input

Supply prepared EEG windows at 128 Hz:

```text
data/<dataset>_X.npy          float32 [windows, channels, 256]
data/<dataset>_metadata.npz   subjects: fixed-width strings or integers [windows]
```

Dataset names are PhysioNet, EEG-MCD and DEAP; custom names also work. [EEG-MCD is our own dataset](https://huggingface.co/datasets/anonymous-588/MCD-EEG/tree/main). Classification labels are not used.

## Train and test

Run commands from this directory:

```bash
# Final SAGE-EEG, three subject folds:
python run.py train --models SAGE-EEG --datasets PhysioNet EEG-MCD DEAP

# All thirteen models, three subject folds:
python run.py train

# Preserve the original five/three-fold allocation:
python run.py train --original-folds

# Test saved checkpoints without optimization:
python run.py test --models SAGE-EEG
python run.py test --original-folds

# One dataset/fold with explicit batch size and paths:
python run.py train --models ConvDAE --datasets PhysioNet --folds 0 --batch-size 256 --data-dir data --output-dir results

# Aggregate saved evaluation metrics:
python run.py summarize --output-dir results
```

Training also tests selected checkpoints. Existing checkpoints are preserved; use a new output directory to retrain. All models use three-fold cross-validation by default (--n-splits 3). Use the same split settings for training and testing.

`model_parameters.json` lists model hyperparameters, batch sizes and optimization settings. Model constructors implement the architecture parameters. `--batch-size` overrides training batches; `--eval-batch-size` defaults to 256. Other options include `--epochs`, `--min-epochs`, `--patience`, `--lr` and `--seed`.

## Evaluation

Electrodes are randomly hidden for the entire window at 25%, 50% and 75% loss. Masks are nested and shared between models with the same subject split. Normalization and priors use training subjects only; validation selects checkpoints and calibrates the SAGE-EEG mixture.

Ten metrics: RMSE, MAE, NMSE, R2, PCC, SNR_dB, SpectralConvergence, LSD_dB, PSD_PCC and BandPowerRelativeError. Only hidden electrodes are scored, averaging channels within windows, windows within subjects, and then subjects. Fold summaries contain means and sample standard deviations. Metrics use train-standardized signals.

Outputs include checkpoints, normalization statistics, splits, training configuration, per-window metrics, `per_fold_metrics.csv` and `summary_mean_std.csv`. Trained weights are saved as `checkpoint.pt` during training.

## Files

- `models.py`: twelve baseline definitions and constructors.
- `field.py`, `gain_field.py`, `library.py`, `zero_expert.py`, `committee.py`: final SAGE-EEG model.
- `model_parameters.json`: hyperparameter reference for all models.
- `sage_training.py`, `calibration.py`: SAGE-EEG training and mixture calibration.
- `run.py`, `protocol.py`: training, checkpoint testing, splits and evaluation.
- `metrics.py`: ten reconstruction metrics.
- `gpvae_numerics.py`: stable GPVAE encoder adaptation.

Transformer, GPVAE, SAITS, TimesNet, iTransformer and TimeMixer use [PyPOTS 1.5](https://github.com/WenjieDu/PyPOTS) backbones adapted to missing-electrode reconstruction. Its BSD license is retained in `THIRD_PARTY_LICENSES.txt`.
