import os
import numpy as np
import librosa
import pywt
from scipy.signal import hilbert
from scipy.interpolate import CubicSpline
from PyEMD import EMD
from tqdm import tqdm
SOURCE_ROOT = 'data/audio'
OUTPUT_ROOT = 'data/features/-5dB'
SAMPLES_PER_CLASS = 500
SR = 44100
SNR_DB = -5
N_FFT = 4096
HOP_LENGTH = 1024
DIM = 32
HHT_N_IMF = 5
HHT_DS = 4
LOFAR_N_BANDS = DIM
FFT_N_BANDS = DIM
WAVELET_NAME = 'db4'
WAVELET_LEVEL = 5
HHSA_N_IMF_L1 = 4
HHSA_N_IMF_L2 = 3
HHSA_DS = 4
N_MFCC = DIM

def add_white_noise(signal, snr_db):
    signal_power = np.mean(signal ** 2)
    if signal_power == 0:
        return signal
    noise_power = signal_power / 10 ** (snr_db / 10)
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise

def extract_hht(y, sr, n_fft, hop_length, target_dim=32):
    ds = HHT_DS
    y_ds = y[::ds]
    sr_ds = sr // ds
    emd = EMD()
    emd.MAX_ITERATION = 10
    emd.FIXE = 6
    try:
        imfs = emd.emd(y_ds, max_imf=HHT_N_IMF)
    except Exception:
        imfs = np.zeros((1, len(y_ds)))
    if imfs.ndim == 1:
        imfs = imfs.reshape(1, -1)
    while len(imfs) < HHT_N_IMF:
        imfs = np.vstack([imfs, np.zeros_like(y_ds)])
    imfs = imfs[:HHT_N_IMF]
    ia_all = np.zeros((HHT_N_IMF, len(y_ds)))
    if_all = np.zeros((HHT_N_IMF, len(y_ds)))
    for k in range(HHT_N_IMF):
        analytic = hilbert(imfs[k])
        ia_all[k] = np.abs(analytic)
        ip = np.unwrap(np.angle(analytic))
        iff = np.clip(np.diff(ip) * sr_ds / (2 * np.pi), 0, sr_ds / 2)
        if_all[k, :len(iff)] = iff
    hop_ds = max(hop_length // ds, 1)
    nfft_ds = max(n_fft // ds, 1)
    T = max(1, 1 + (len(y_ds) - nfft_ds) // hop_ds)
    feat = np.zeros((T, target_dim), dtype=np.float32)
    for t in range(T):
        s = t * hop_ds
        e = min(s + nfft_ds, len(y_ds))
        if e <= s:
            continue
        for k in range(HHT_N_IMF):
            ia_seg = ia_all[k, s:e]
            if_seg = if_all[k, s:e]
            ie = ia_seg ** 2 + 1e-14
            p = ie / ie.sum()
            base = k * 6
            feat[t, base:base + 6] = [np.mean(ie), np.std(ia_seg), np.max(ia_seg), np.mean(if_seg), np.std(if_seg), -np.sum(p * np.log2(p + 1e-14))]
        all_ia_means = [np.mean(ia_all[k, s:e]) for k in range(HHT_N_IMF)]
        all_ia_max = [np.max(ia_all[k, s:e]) for k in range(HHT_N_IMF)]
        feat[t, 30] = np.std(all_ia_means)
        feat[t, 31] = max(all_ia_max) / (np.mean(all_ia_means) + 1e-14)
    return feat

def extract_lofar(y, sr, n_fft, hop_length, n_bands=32):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    freq_bins = S.shape[0]
    T = S.shape[1]
    freq_axis = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    min_freq = max(20.0, freq_axis[1])
    max_freq = sr / 2.0
    band_edges = np.logspace(np.log10(min_freq), np.log10(max_freq), n_bands + 1)
    feat = np.zeros((T, n_bands), dtype=np.float32)
    for b in range(n_bands):
        lo = band_edges[b]
        hi = band_edges[b + 1]
        idx = np.where((freq_axis >= lo) & (freq_axis < hi))[0]
        if len(idx) == 0:
            idx = [np.argmin(np.abs(freq_axis - (lo + hi) / 2))]
        feat[:, b] = np.mean(S[idx, :], axis=0)
    feat = 10 * np.log10(feat + 1e-14)
    feat -= np.mean(feat, axis=1, keepdims=True)
    return feat.T if feat.shape[0] == n_bands else feat

def extract_fft(y, sr, n_fft, hop_length, n_bands=32):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freq_bins = S.shape[0]
    T = S.shape[1]
    n_mag_bands = 24
    n_shape = n_bands - n_mag_bands
    feat = np.zeros((T, n_bands), dtype=np.float32)
    edges = np.linspace(0, freq_bins, n_mag_bands + 1, dtype=int)
    for b in range(n_mag_bands):
        lo, hi = (edges[b], edges[b + 1])
        if hi <= lo:
            hi = lo + 1
        feat[:, b] = np.mean(S[lo:hi, :], axis=0)
    freq_axis = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    for t in range(T):
        mag = S[:, t]
        mag_sum = mag.sum() + 1e-14
        p = mag / mag_sum
        feat[t, 24] = np.sum(freq_axis * p)
        centroid = feat[t, 24]
        feat[t, 25] = np.sqrt(np.sum(p * (freq_axis - centroid) ** 2))
        cumsum = np.cumsum(mag)
        feat[t, 26] = freq_axis[np.searchsorted(cumsum, 0.85 * cumsum[-1])]
        log_mag = np.log(mag + 1e-14)
        feat[t, 27] = np.exp(np.mean(log_mag)) / (np.mean(mag) + 1e-14)
        feat[t, 28] = -np.sum(p * np.log2(p + 1e-14))
        feat[t, 29] = np.polyfit(np.arange(freq_bins), mag, 1)[0]
        feat[t, 30] = np.max(mag) / (np.mean(mag) + 1e-14)
        denom = np.arange(1, freq_bins) + 1e-14
        feat[t, 31] = np.sum((mag[1:] - mag[0]) / denom) / mag_sum
    return feat

def extract_wpt_energy_single_frame(frame, level=5, wavelet='db4'):
    frame = np.array(frame).copy()
    min_len = 2 ** level
    if len(frame) < min_len:
        frame = np.pad(frame, (0, min_len - len(frame)))
    wp = pywt.WaveletPacket(data=frame, wavelet=wavelet, mode='symmetric', maxlevel=level)
    nodes = [node.path for node in wp.get_level(level, 'natural')]
    energies = []
    for node in nodes:
        data = wp[node].data
        eng = np.sum(data ** 2) / len(data) if len(data) > 0 else 0
        energies.append(eng)
    return np.array(energies, dtype=np.float32)

def extract_wpt(y, n_fft, hop_length, level=5, wavelet='db4'):
    frames = librosa.util.frame(y, frame_length=n_fft, hop_length=hop_length).T
    feat_list = []
    for frame in frames:
        energy = extract_wpt_energy_single_frame(frame, level=level, wavelet=wavelet)
        feat_list.append(energy)
    return np.array(feat_list, dtype=np.float32)

def _compute_envelope(imf):
    analytic = hilbert(imf)
    env = np.abs(analytic)
    return env

def _safe_emd(signal, max_imf=3):
    emd = EMD()
    emd.MAX_ITERATION = 10
    emd.FIXE = 6
    try:
        imfs = emd.emd(signal, max_imf=max_imf)
    except Exception:
        return np.zeros((1, len(signal)))
    if imfs is None or len(imfs) == 0:
        return np.zeros((1, len(signal)))
    if imfs.ndim == 1:
        imfs = imfs.reshape(1, -1)
    return imfs

def extract_hhsa(y, sr, n_fft, hop_length, target_dim=32):
    ds = HHSA_DS
    y_ds = y[::ds]
    sr_ds = sr // ds
    n_l1 = HHSA_N_IMF_L1
    n_l2 = HHSA_N_IMF_L2
    imfs_l1 = _safe_emd(y_ds, max_imf=n_l1)
    while len(imfs_l1) < n_l1:
        imfs_l1 = np.vstack([imfs_l1, np.zeros_like(y_ds)])
    imfs_l1 = imfs_l1[:n_l1]
    hop_ds = max(hop_length // ds, 1)
    nfft_ds = max(n_fft // ds, 1)
    T = max(1, 1 + (len(y_ds) - nfft_ds) // hop_ds)
    feat = np.zeros((T, target_dim), dtype=np.float32)
    envelopes = []
    l2_imfs_all = []
    l2_ia_all = []
    l2_if_all = []
    for j in range(n_l1):
        env = _compute_envelope(imfs_l1[j])
        envelopes.append(env)
        imfs_l2 = _safe_emd(env, max_imf=n_l2)
        while len(imfs_l2) < n_l2:
            imfs_l2 = np.vstack([imfs_l2, np.zeros(len(env))])
        imfs_l2 = imfs_l2[:n_l2]
        l2_imfs_all.append(imfs_l2)
        ia_l2 = np.zeros((n_l2, len(env)))
        if_l2 = np.zeros((n_l2, len(env)))
        for k in range(n_l2):
            analytic = hilbert(imfs_l2[k])
            ia_l2[k] = np.abs(analytic)
            ip = np.unwrap(np.angle(analytic))
            iff = np.clip(np.diff(ip) * sr_ds / (2 * np.pi), 0, sr_ds / 2)
            if_l2[k, :len(iff)] = iff
        l2_ia_all.append(ia_l2)
        l2_if_all.append(if_l2)
    for t in range(T):
        s = t * hop_ds
        e = min(s + nfft_ds, len(y_ds))
        if e <= s:
            continue
        for j in range(n_l1):
            for k in range(n_l2):
                ia_seg = l2_ia_all[j][k, s:e]
                if_seg = l2_if_all[j][k, s:e]
                idx = j * (n_l2 * 2) + k * 2
                feat[t, idx] = np.mean(ia_seg)
                feat[t, idx + 1] = np.mean(if_seg)
        for j in range(n_l1):
            env_seg = envelopes[j][s:e]
            std_env = np.std(env_seg)
            rms_env = np.sqrt(np.mean(env_seg ** 2) + 1e-14)
            feat[t, 24 + j] = std_env / rms_env
        env_means = [np.mean(envelopes[j][s:e]) for j in range(n_l1)]
        env_maxs = [np.max(envelopes[j][s:e]) for j in range(n_l1)]
        feat[t, 28] = np.std(env_means)
        feat[t, 29] = max(env_maxs) / (np.mean(env_means) + 1e-14)
        total_env_energy = sum((np.sum(envelopes[j][s:e] ** 2) for j in range(n_l1)))
        total_l2_energy = 0.0
        for j in range(n_l1):
            for k in range(n_l2):
                total_l2_energy += np.sum(l2_ia_all[j][k, s:e] ** 2)
        feat[t, 30] = total_l2_energy / (total_env_energy + 1e-14)
        feat[t, 31] = np.mean(feat[t, 24:28])
    return feat

def extract_mfcc(y, sr, n_fft, hop_length, n_mfcc=32):
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length).T.astype(np.float32)

def process_full_pipeline():
    np.random.seed(42)
    feature_names = ['HHT', 'LOFAR', 'FFT', 'WPT', 'HHSA', 'MFCC']
    feature_index_map = {}
    offset = 0
    for name in feature_names:
        feature_index_map[name] = (offset, offset + DIM)
        offset += DIM
    total_dim = offset
    print('=' * 65)
    print('  Comprehensive Feature Extraction Pipeline')
    print('  HHT + LOFAR + FFT + WPT + HHSA + MFCC')
    print('=' * 65)
    print(f'Total Feature Dimension: {total_dim}')
    for name in feature_names:
        lo, hi = feature_index_map[name]
        print(f'  {name:>5s}: [{lo:3d}, {hi:3d})')
    print(f'SNR: {SNR_DB} dB | SR: {SR} Hz | N_FFT: {N_FFT} | HOP: {HOP_LENGTH}')
    print(f'Samples per class: {SAMPLES_PER_CLASS}')
    print('-' * 65)
    target_labels = [f'{i:02d}' for i in range(10)]
    print(f'Target Classes: {target_labels}')
    for label in target_labels:
        src_dir = os.path.join(SOURCE_ROOT, label)
        save_dir = os.path.join(OUTPUT_ROOT, label)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(src_dir):
            print(f'[SKIP] Class {label}: source folder not found.')
            continue
        files = [f for f in os.listdir(src_dir) if f.lower().endswith('.wav')]
        if len(files) > SAMPLES_PER_CLASS:
            np.random.shuffle(files)
            selected_files = files[:SAMPLES_PER_CLASS]
        else:
            selected_files = files
        print(f'\nClass {label}: {len(files)} found, {len(selected_files)} selected')
        for fname in tqdm(selected_files, desc=f'Class {label}'):
            try:
                file_path = os.path.join(src_dir, fname)
                y, _ = librosa.load(file_path, sr=SR)
                if len(y) < N_FFT:
                    continue
                y_noisy = add_white_noise(y, SNR_DB)
                feat_hht = extract_hht(y_noisy, SR, N_FFT, HOP_LENGTH, DIM)
                feat_lofar = extract_lofar(y_noisy, SR, N_FFT, HOP_LENGTH, LOFAR_N_BANDS)
                feat_fft = extract_fft(y_noisy, SR, N_FFT, HOP_LENGTH, FFT_N_BANDS)
                feat_wpt = extract_wpt(y_noisy, N_FFT, HOP_LENGTH, WAVELET_LEVEL, WAVELET_NAME)
                feat_hhsa = extract_hhsa(y_noisy, SR, N_FFT, HOP_LENGTH, DIM)
                feat_mfcc = extract_mfcc(y_noisy, SR, N_FFT, HOP_LENGTH, N_MFCC)
                min_len = min(feat_hht.shape[0], feat_lofar.shape[0], feat_fft.shape[0], feat_wpt.shape[0], feat_hhsa.shape[0], feat_mfcc.shape[0])
                if min_len == 0:
                    continue
                feat_hht = feat_hht[:min_len]
                feat_lofar = feat_lofar[:min_len]
                feat_fft = feat_fft[:min_len]
                feat_wpt = feat_wpt[:min_len]
                feat_hhsa = feat_hhsa[:min_len]
                feat_mfcc = feat_mfcc[:min_len]
                combined = np.hstack([feat_hht, feat_lofar, feat_fft, feat_wpt, feat_hhsa, feat_mfcc]).astype(np.float32)
                np.save(os.path.join(save_dir, fname.replace('.wav', '.npy')), combined)
            except Exception as e:
                print(f'  [ERROR] {fname}: {e}')
    np.save(os.path.join(OUTPUT_ROOT, 'feature_indices.npy'), feature_index_map)
    print('\n' + '=' * 65)
    print('  All Done.')
    print(f'  Output shape per sample: (T, {total_dim})')
    print(f'  Feature index map saved to: feature_indices.npy')
    print('=' * 65)
if __name__ == '__main__':
    process_full_pipeline()
