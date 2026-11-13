# RMAL-CRNN Supplementary Code

This anonymous package contains the code for the experiments and analyses reported with the submission. The public Kielce UAV recordings are available from Zenodo record 15190811; the Glasgow/ACSAC~2022 benchmark is cited in the manuscript.

## Contents

- `classification/`: feature extraction, baseline comparison, ablation, sensitivity, and plotting.
- `simulation/`: controlled corruption and mechanism analyses for GST, LMA, and FNSAR.
- `range/training/`: long-range acoustic preprocessing, channel selection, feature refinement, training, and evaluation.
- `range/distill/`: optional teacher-student distillation performed in PyTorch.
- `range/teacher/`: compact teacher checkpoint and recorded metrics used by the distillation scripts.
- `data/`: expected input layout; raw data are not redistributed.
- `outputs/`: generated results; created by the scripts when required.

## Environment

Python 3.10 or newer is recommended.

```text
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```

On Linux or macOS, activate the environment with the platform-specific command and use the same requirements file.

## Data layout

```text
data/
  audio/
    00/ ... 09/
  features/
    -5dB/ 0dB/ 5dB/ 10dB/
  range/
    90101.dat
    90102.dat
    dataset1/noise/90105.dat
```

The ten class folders contain one-second WAV files. Extracted feature arrays use shape `(T, 192)` with the order HHT, LOFAR, FFT, WPT, HHSA, and MFCC, each contributing 32 coordinates. The released class folders use the UAV-model labels listed in Appendix A; extracted features retain the corresponding SNR and class directory structure.

## Main commands

```text
python classification/features.py
python classification/compare.py
python classification/ablation.py
python classification/sweep.py
python classification/performance.py
python classification/sensitivity.py
```

Controlled simulation scripts are independent entry points, for example:

```text
python simulation/corruption.py
python simulation/interference.py
python simulation/affinity.py
python simulation/embedding.py
```

The long-range pipeline is run in this order:

```text
python range/training/selection.py
python range/training/features.py
python range/training/ranking.py
python range/training/evaluate.py
```

Distillation is optional. Run the following commands from `range/distill/` after the long-range feature files have been generated:

```text
python splits.py --features-dir ../../outputs/range/channels/features_refined
python cache.py
python train.py
python evaluate.py
```

No source file contains author names, affiliations, email addresses, private repository links, local user paths, or device-specific instructions.
