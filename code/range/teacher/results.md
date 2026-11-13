# Teacher Record

The teacher uses six reliability-weighted descriptor blocks, four-head cross-domain attention, a 64-dimensional projection, rank-8 temporal refinement, two temporal convolution blocks, a bidirectional LSTM, and a two-class output layer.

The recorded run uses seed 2026, AdamW with learning rate `1.5e-3` and weight decay `1e-4`, batch size 128, 16 epochs, inverse-frequency class weights, and gradient clipping at 2. The checkpoint is selected by validation accuracy. Threshold selection is performed on validation data subject to the stated false-alarm constraint.

Numerical results, channel rankings, and the selected-channel list are stored in `metrics.json`, `ranking.json`, and `channels.json`. The checkpoint is `model.pt`.
