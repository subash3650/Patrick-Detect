import parselmouth
import numpy as np
import math
from scipy.signal import welch
from scipy.stats import entropy


def safe_to_pitch(sound, time_step=0.01, pitch_floor=75, pitch_ceiling=600):
    """
    Safely extract pitch with fallback options.
    Only uses parameters that are actually supported by parselmouth.
    """
    try:
        # Correct parselmouth API - only supports time_step, pitch_floor, pitch_ceiling
        pitch = sound.to_pitch(time_step=time_step, 
                               pitch_floor=pitch_floor, 
                               pitch_ceiling=pitch_ceiling)
        return pitch
    except Exception as e1:
        try:
            # Fallback 1: Try Praat's AC method
            pitch = parselmouth.praat.call(sound, "To Pitch (ac)", time_step, pitch_floor, pitch_ceiling)
            return pitch
        except Exception as e2:
            try:
                # Fallback 2: Wider range
                pitch = sound.to_pitch(time_step=time_step, 
                                       pitch_floor=50, 
                                       pitch_ceiling=800)
                return pitch
            except Exception as e3:
                raise RuntimeError(f"pitch extraction failed: {str(e1)} | {str(e2)} | {str(e3)}")


def get_pitch_stats(sound):
    duration = sound.get_total_duration()
    if duration < 0.15:
        raise RuntimeError(f"audio too short for pitch extraction (duration={duration:.3f}s)")

    pitch = safe_to_pitch(sound)
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[np.isfinite(pitch_values)]
    pitch_values = pitch_values[pitch_values > 0]
    if len(pitch_values) == 0:
        raise RuntimeError("no voiced frames detected in pitch extraction")

    return {
        "Fo": float(np.mean(pitch_values)),
        "Fhi": float(np.max(pitch_values)),
        "Flo": float(np.min(pitch_values)),
        "duration": float(duration),
        "pitch_values": pitch_values  # Return for NHR calculation
    }


def get_jitter_shimmer_hnr(sound):
    try:
        pp = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 500)
    except Exception:
        pp = parselmouth.praat.call(sound, "To PointProcess (cc)", 75, 500)

    try:
        jitter_local = parselmouth.praat.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    except Exception:
        jitter_local = 0.0
    try:
        jitter_local_abs = parselmouth.praat.call(pp, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    except Exception:
        jitter_local_abs = 0.0
    try:
        jitter_rap = parselmouth.praat.call(pp, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    except Exception:
        jitter_rap = 0.0
    try:
        jitter_ppq5 = parselmouth.praat.call(pp, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    except Exception:
        jitter_ppq5 = 0.0
    try:
        jitter_ddp = parselmouth.praat.call(pp, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    except Exception:
        jitter_ddp = 0.0

    try:
        shimmer_local = parselmouth.praat.call([sound, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    except Exception:
        shimmer_local = 0.0
    try:
        shimmer_db = parselmouth.praat.call([sound, pp], "Get shimmer (dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    except Exception:
        shimmer_db = 0.0
    try:
        apq3 = parselmouth.praat.call([sound, pp], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    except Exception:
        apq3 = 0.0
    try:
        apq5 = parselmouth.praat.call([sound, pp], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    except Exception:
        apq5 = 0.0
    try:
        apq = parselmouth.praat.call([sound, pp], "Get shimmer (apq)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    except Exception:
        apq = 0.0
    try:
        dda = parselmouth.praat.call([sound, pp], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    except Exception:
        dda = 0.0

    try:
        harmonicity = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
    except Exception:
        hnr = 0.0

    return {
        "Jitter(%)": float(jitter_local),
        "Jitter_Abs": float(jitter_local_abs),
        "RAP": float(jitter_rap),
        "PPQ": float(jitter_ppq5),
        "Jitter:DDP": float(jitter_ddp),
        "Shimmer": float(shimmer_local),
        "Shimmer(dB)": float(shimmer_db),
        "Shimmer:APQ3": float(apq3),
        "Shimmer:APQ5": float(apq5),
        "MDVP:APQ": float(apq),
        "Shimmer:DDA": float(dda),
        "HNR": float(hnr)
    }


def compute_nhr(sound):
    """Noise-to-Harmonicity Ratio (inverse of HNR in dB)"""
    try:
        harmonicity = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
        # NHR = 1 / (10^(HNR/20))
        if hnr > -200:  # Avoid overflow
            nhr = 1.0 / (10.0 ** (hnr / 20.0))
        else:
            nhr = 0.0
        return float(nhr)
    except Exception:
        return 0.0


def compute_rpde(sound, order=2, delay=1):
    """Recurrence Period Density Entropy"""
    try:
        # Get the audio samples as time series
        samples = sound.values[0]  # Get first channel
        
        if len(samples) < order * delay + 1:
            return 0.0
        
        # Normalize samples
        samples = (samples - np.mean(samples)) / (np.std(samples) + 1e-10)
        
        # Create delay embedding
        N = len(samples) - (order - 1) * delay
        embedded = np.zeros((N, order))
        for i in range(order):
            embedded[:, i] = samples[i * delay:i * delay + N]
        
        # Compute recurrence matrix (Euclidean distance)
        recurrence = np.zeros((N, N))
        threshold = np.std(embedded) * 0.15  # Threshold for recurrence
        
        for i in range(N):
            for j in range(N):
                dist = np.sqrt(np.sum((embedded[i] - embedded[j]) ** 2))
                recurrence[i, j] = 1 if dist < threshold else 0
        
        # Compute period density (diagonal lines)
        periods = []
        for k in range(1, N):
            diagonal_recurrence = np.sum(np.diag(recurrence, k=k))
            if diagonal_recurrence > 0:
                periods.append(diagonal_recurrence)
        
        if len(periods) == 0:
            return 0.0
        
        # Entropy of period densities
        period_density = np.array(periods) / np.sum(periods)
        rpde = -np.sum(period_density * np.log(period_density + 1e-10))
        
        return float(rpde)
    except Exception:
        return 0.0


def compute_dfa(sound, order=1):
    """Detrended Fluctuation Analysis"""
    try:
        samples = sound.values[0]
        samples = (samples - np.mean(samples)) / (np.std(samples) + 1e-10)
        
        # Cumulative sum
        y = np.cumsum(samples)
        
        # Divide into segments
        N = len(y)
        segment_lengths = np.logspace(0.5, 3, 10, dtype=int)
        segment_lengths = segment_lengths[segment_lengths < N // 2]
        
        fluctuations = []
        for seg_len in segment_lengths:
            # Forward segments
            n_segments = N // seg_len
            F = 0
            for i in range(n_segments):
                start = i * seg_len
                end = start + seg_len
                # Fit polynomial
                x = np.arange(seg_len)
                coef = np.polyfit(x, y[start:end], order)
                fitted = np.polyval(coef, x)
                F += np.sum((y[start:end] - fitted) ** 2) / seg_len
            
            # Backward segments
            for i in range(n_segments):
                start = N - (i + 1) * seg_len
                end = start + seg_len
                x = np.arange(seg_len)
                coef = np.polyfit(x, y[start:end], order)
                fitted = np.polyval(coef, x)
                F += np.sum((y[start:end] - fitted) ** 2) / seg_len
            
            F = np.sqrt(F / (2 * n_segments))
            fluctuations.append(F)
        
        # Linear fit in log-log space
        log_seg_len = np.log(segment_lengths)
        log_F = np.log(fluctuations)
        dfa = np.polyfit(log_seg_len, log_F, 1)[0]
        
        return float(dfa)
    except Exception:
        return 0.0


def compute_spread(sound):
    """Spread1 and Spread2 using wavelet-based analysis"""
    try:
        samples = sound.values[0]
        samples = (samples - np.mean(samples)) / (np.std(samples) + 1e-10)
        
        # Simple approximation: use spectral spread
        freqs, psd = welch(samples, sound.sampling_frequency, nperseg=1024)
        
        # Remove DC component
        psd = psd[1:]
        freqs = freqs[1:]
        
        # Normalize
        psd = psd / np.sum(psd)
        
        # Spectral centroid
        centroid = np.sum(freqs * psd)
        
        # Spectral spread (standard deviation)
        spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * psd))
        
        # Approximate spread1 and spread2
        spread1 = spread - centroid
        spread2 = np.sqrt(np.sum(((freqs - centroid) ** 3) * psd)) / (spread ** 3 + 1e-10)
        
        return float(spread1), float(spread2)
    except Exception:
        return 0.0, 0.0


def compute_d2(sound):
    """Correlation Dimension (simplified)"""
    try:
        samples = sound.values[0]
        samples = (samples - np.mean(samples)) / (np.std(samples) + 1e-10)
        
        # Use decimated samples for speed
        step = max(1, len(samples) // 1000)
        samples = samples[::step]
        
        # Compute correlation integral
        max_dist = np.max(np.abs(np.diff(samples)))
        distances = np.logspace(-2, 0, 20) * max_dist
        
        C = []
        for dist in distances:
            count = 0
            N = len(samples)
            for i in range(N):
                for j in range(i + 1, N):
                    if abs(samples[i] - samples[j]) < dist:
                        count += 1
            C.append(count / (N * (N - 1) / 2))
        
        # Linear fit in log-log
        C = np.array(C)
        C = C[C > 0]
        if len(C) < 2:
            return 3.0
        
        log_dist = np.log(distances[-len(C):])
        log_C = np.log(C)
        d2 = np.polyfit(log_dist, log_C, 1)[0]
        
        return float(d2)
    except Exception:
        return 3.0


def compute_ppe(sound):
    """Pitch Period Entropy"""
    try:
        pitch = safe_to_pitch(sound)
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[np.isfinite(pitch_values)]
        pitch_values = pitch_values[pitch_values > 0]
        
        if len(pitch_values) < 10:
            return 0.0
        
        # Compute differences (pitch periods)
        periods = 1.0 / pitch_values
        periods = (periods - np.mean(periods)) / (np.std(periods) + 1e-10)
        
        # Histogram of normalized periods
        hist, _ = np.histogram(periods, bins=10, range=(-3, 3))
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]
        
        # Shannon entropy
        ppe = -np.sum(hist * np.log2(hist + 1e-10))
        
        return float(ppe)
    except Exception:
        return 0.0


def extract_from_wav(path):
    sound = parselmouth.Sound(path)
    duration = sound.get_total_duration()
    sr = sound.sampling_frequency

    pitch_stats = get_pitch_stats(sound)
    js = get_jitter_shimmer_hnr(sound)

    features = {}
    features['MDVP:Fo(Hz)'] = pitch_stats.get('Fo', 0.0)
    features['MDVP:Fhi(Hz)'] = pitch_stats.get('Fhi', 0.0)
    features['MDVP:Flo(Hz)'] = pitch_stats.get('Flo', 0.0)
    features['MDVP:Jitter(%)'] = js.get('Jitter(%)', 0.0)
    features['MDVP:Jitter(Abs)'] = js.get('Jitter_Abs', 0.0)
    features['MDVP:RAP'] = js.get('RAP', 0.0)
    features['MDVP:PPQ'] = js.get('PPQ', 0.0)
    features['Jitter:DDP'] = js.get('Jitter:DDP', 0.0)
    features['MDVP:Shimmer'] = js.get('Shimmer', 0.0)
    features['MDVP:Shimmer(dB)'] = js.get('Shimmer(dB)', 0.0)
    features['Shimmer:APQ3'] = js.get('Shimmer:APQ3', 0.0)
    features['Shimmer:APQ5'] = js.get('Shimmer:APQ5', 0.0)
    features['MDVP:APQ'] = js.get('MDVP:APQ', 0.0)
    features['Shimmer:DDA'] = js.get('Shimmer:DDA', 0.0)

    # Compute nonlinear features (previously zero)
    features['NHR'] = compute_nhr(sound)
    features['HNR'] = js.get('HNR', 0.0)
    features['RPDE'] = compute_rpde(sound, order=2, delay=1)
    features['DFA'] = compute_dfa(sound, order=1)
    
    spread1, spread2 = compute_spread(sound)
    features['spread1'] = spread1
    features['spread2'] = spread2
    
    features['D2'] = compute_d2(sound)
    features['PPE'] = compute_ppe(sound)

    features['_meta_duration'] = float(duration)
    features['_meta_samplerate'] = float(sr)

    return features
