# Long-Range Experiment

The retained files cover the paper-relevant acoustic processing chain: multichannel decoding, six-domain feature construction, channel ranking, RMAL-CRNN training, independent-channel evaluation, and optional knowledge distillation.

The recordings are represented by three identifiers. `90101` and `90102` are positive UAV recordings; `90105` is background. The preprocessing uses 32 channels, excludes acquisition channels 9 and 15, and evaluates channel 20 independently after selecting 20 training channels. One-second windows are resampled to 12.8 kHz and represented by 192 features.

Device-specific installation, runtime conversion, quantization calibration, hardware benchmarking, and real-time inference files are excluded because they are not required to reproduce the manuscript's acoustic-learning results.
