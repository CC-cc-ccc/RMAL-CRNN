# Teacher Protocol

The positive segments are 200 seconds from `90101` beginning at 76 seconds and the centered 200 seconds from `90102`. The negative segment is the final 400 seconds of `90105`. Each source is divided chronologically into 60% training, 20% validation, and 20% internal testing.

Channels 9 and 15 are excluded during preprocessing. The selected training channels are 7, 25, 29, 32, 19, 4, 10, 12, 26, 1, 30, 3, 31, 8, 6, 27, 24, 16, 13, and 22. Channel 20 is reserved for independent evaluation.

Each one-second waveform is resampled to 12.8 kHz. The feature order is HHT, LOFAR, FFT, WPT, HHSA, and MFCC, with 32 coordinates per domain. Standardization statistics are fitted only on the selected-channel training windows and then reused unchanged.

Raw recordings are excluded from this archive because of size and redistribution constraints. Place them under the `data/range/` layout documented in the package root.
