import os
import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pyedflib as edf  # to read physio in MAHNOB-HCI
from scipy import signal
from scipy.interpolate import interp1d

from .utils import (
    rolling_mean,
    rolling_std,
    rolling_min,
    rolling_max,
    set_exact_frame,
)


def find_peaks(data, min_sep, window_len, threshold=2.0, axis=-1):
    """
    Detect smooth peaks by z-score and local max.

    :param data: Input data.
    :type data: list
    :param min_sep: Minimum separation between peaks.
    :type min_sep: int
    :param window_len: Length of the window.
    :type window_len: int
    :param threshold: Threshold for z-score. (default: 2.0)
    :type threshold: float, optional
    :param axis: Axis along which to operate. (default: -1)
    :type axis: int, optional
    :return: Boolean mask of detected peaks.
    :rtype: numpy.ndarray
    """
    r_max = rolling_max(data, min_sep, center=True, axis=axis)
    mask_localmax = np.isclose(r_max, data, atol=1e-10)
    r_mean = rolling_mean(data, window_len, center=True)
    r_std = rolling_std(data, window_len, center=True)
    mask_p_value = np.abs(data - r_mean) > (r_std * threshold)
    mask = np.logical_and(mask_localmax, mask_p_value)
    return mask


def find_spikes(
    data, min_sep, spike_width, window_len, threshold=1.5, axis=-1
):
    """
    Detect spike-like peaks by move_max-move_min, z-score, and local max.

    :param data: Input data.
    :type data: list
    :param min_sep: Minimum separation between peaks.
    :type min_sep: int
    :param spike_width: Width of the spike.
    :type spike_width: int
    :param window_len: Length of the window.
    :type window_len: int
    :param threshold: Threshold for z-score. (default: 1.5)
    :type threshold: float, optional
    :param axis: Axis along which to operate. (default: -1)
    :type axis: int, optional
    :return: Boolean mask of detected spikes.
    :rtype: numpy.ndarray
    """
    r_max = rolling_max(data, min_sep, center=True, axis=axis)
    mask_local_max = np.isclose(r_max, data, atol=1e-10)
    r_max = rolling_max(data, window_len=spike_width, center=True, axis=axis)
    r_min = rolling_min(data, window_len=spike_width, center=True, axis=axis)
    y = r_max - r_min
    y_mean = rolling_mean(y, window_len=window_len, center=True, axis=axis)
    y_std = rolling_std(y, window_len=window_len, center=True, axis=axis)
    mask_p_value = y > (y_mean + y_std * threshold)
    mask = np.logical_and(mask_local_max, mask_p_value)
    return mask


def estimate_hr_snr_by_fft(data_array, sampling_rate, num_harmonics=1, central_moment=True, axis=-1):
    """
    Estimate heart rate and signal-to-noise ratio (SNR) using Fast Fourier Transform (FFT).

    :param data_array: Input data array.
    :type data_array: numpy.ndarray
    :param sampling_rate: Sampling rate of the data array.
    :type sampling_rate: float
    :param num_harmonics: Number of harmonics to consider. (default: 1).
    :type num_harmonics: int, optional
    :param central_moment: Calculate central moment, default is True.
    :type central_moment: bool, optional
    :param axis: Axis along which the operation is applied. (default: -1).
    :type axis: int, optional
    :returns: Tuple containing heart rate and SNR arrays.
    :rtype: tuple

    References:
        - https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html
    """
    f, Pxx = signal.periodogram(
        data_array, fs=sampling_rate, window='boxcar', detrend=False, axis=axis)
    n = Pxx.shape[axis]
    if data_array.ndim > 1:
        if axis != -1 and axis != Pxx.ndim-1:
            Pxx = np.moveaxis(Pxx, axis, -1)
        reduced_shape = Pxx.shape[:-1]
        Pxx = Pxx.reshape(-1, Pxx.shape[-1])
        hr = np.empty(len(Pxx), dtype=np.float64)
        snr = np.empty(len(Pxx), dtype=np.float64)
        for i in range(len(Pxx)):
            pxx = Pxx[i]
            imax = np.argmax(pxx[1:]) + 1
            iL, iR = imax, imax
            while iL > 1 and pxx[iL - 1] < pxx[iL]:
                iL -= 1
            while iR + 1 < n and pxx[iR + 1] < pxx[iR]:
                iR += 1
            ptotal = np.sum(pxx[1:])
            psignal = np.sum(pxx[iL:iR+1])
            hr[i] = (np.sum(f[iL:iR+1] * pxx[iL:iR+1]) /
                     psignal) if central_moment else f[imax]
            for j in range(num_harmonics):
                fharm = hr[i] * (j + 2)
                if fharm > f[-1]:
                    break
                iL, iR = np.searchsorted(
                    f, fharm, side='left') - 1, np.searchsorted(f, fharm, side='right')
                while iL > 1 and pxx[iL - 1] < pxx[iL]:
                    iL -= 1
                while iR + 1 < n and pxx[iR + 1] < pxx[iR]:
                    iR += 1
                psignal += np.sum(pxx[iL:iR+1])
            snr[i] = psignal / (ptotal - psignal)
        hr = hr.reshape(reduced_shape)
        snr = snr.reshape(reduced_shape)
    else:  # Pxx.ndim == 1
        imax = np.argmax(Pxx[1:]) + 1
        iL, iR = imax, imax
        while iL > 1 and Pxx[iL - 1] < Pxx[iL]:
            iL -= 1
        while iR + 1 < n and Pxx[iR + 1] < Pxx[iR]:
            iR += 1
        psignal = np.sum(Pxx[iL:iR+1])
        ptotal = np.sum(Pxx[1:])
        hr = np.dot(f[iL:iR+1], Pxx[iL:iR+1]) / \
            psignal if central_moment else f[imax]
        for i in range(num_harmonics):
            fharm = hr * (i + 2)
            if fharm > f[-1]:
                break
            iL, iR = np.searchsorted(
                f, fharm, side='left') - 1, np.searchsorted(f, fharm, side='right')
            while iL > 1 and Pxx[iL - 1] < Pxx[iL]:
                iL -= 1
            while iR + 1 < n and Pxx[iR + 1] < Pxx[iR]:
                iR += 1
            psignal += np.sum(Pxx[iL:iR+1])
        snr = psignal / (ptotal - psignal)
    return hr, snr


def estimate_hr_by_peaks(time_peaks, time_start, time_end, unit=1.0):
    """
    Estimate heart rate by peaks.

    :param time_peaks: Ascending timestamps (in seconds) of each peak.
    :type time_peaks: list
    :param time_start: Timestamp of the start.
    :type time_start: float
    :param time_end: Timestamp of the end.
    :type time_end: float
    :param unit: Time unit (default: 1.0 seconds).
    :type unit: float, optional
    :returns: Estimated heart rate in Hz.
    :rtype: float
    """
    index_start = np.searchsorted(time_peaks, time_start, side='left')
    index_end = np.searchsorted(time_peaks, time_end, side='right')
    delta_time = np.diff(time_peaks[index_start:index_end])
    valid_deltas = np.abs(delta_time - np.mean(delta_time)
                          ) < np.std(delta_time) * 2.5
    heart_rate = unit / np.mean(delta_time[valid_deltas])
    return heart_rate


def plot_stft(signal_data, sampling_rate):
    """
    Plot Short-Time Fourier Transform (STFT) of the input signal.

    :param signal_data: Input signal data.
    :type signal_data: list
    :param sampling_rate: Sampling rate of the signal.
    :type sampling_rate: float
    :returns: Frequency array, time array, and STFT matrix.
    :rtype: tuple
    """
    freq, time, stft_matrix = signal.spectrogram(signal_data, sampling_rate)
    plt.pcolormesh(time, freq, stft_matrix)
    plt.gca().set_yscale('log')
    plt.ylim(top=freq[-1])
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (sec)')
    return freq, time, stft_matrix


def plot_cwt(signal_data, sampling_rate, wavelet='cmor1.5-1.0', total_scales=256, cmap='viridis'):
    """
    Plot Continuous Wavelet Transform (CWT) of the input signal.

    :param signal_data: Input signal data.
    :type signal_data: list
    :param sampling_rate: Sampling rate of the signal.
    :type sampling_rate: float
    :param wavelet: Wavelet type (default: 'cmor1.5-1.0').
    :type wavelet: str, optional
    :param total_scales: Number of scales (default: 256).
    :type total_scales: int, optional
    :param cmap: Colormap (default: 'viridis').
    :type cmap: str, optional
    :return: tuple of frequency array, time array, and CWT matrix.
    :rtype: tuple
    """
    scales = 2 * pywt.central_frequency(wavelet) * total_scales / \
        np.arange(total_scales, 1, -1, dtype=np.float64)
    cwt_matrix, freq = pywt.cwt(
        signal_data, scales, wavelet, 1.0 / sampling_rate)
    time = np.arange(len(signal_data), dtype=np.float64) * (1 / sampling_rate)
    plt.pcolormesh(time, freq, abs(cwt_matrix), cmap=cmap)
    plt.gca().set_yscale('log')
    plt.ylim(top=freq[0])
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (sec)')
    return freq, time, cwt_matrix


def plot_pearson_correlation(x_data, y_data):
    """
    Plot Pearson correlation coefficient.

    :param x_data: Input data for x-axis.
    :type x_data: list
    :param y_data: Input data for y-axis.
    :type y_data: list
    :return: tuple of Pearson correlation coefficient (r) and p-value.
    :rtype: tuple
    """
    x_array, y_array = np.asanyarray(x_data), np.asanyarray(y_data)
    r, p = stats.pearsonr(x_array, y_array)
    x_mean, y_mean = np.mean(x_array), np.mean(y_array)
    x_min, x_max = np.min(x_array), np.max(x_array)
    x_line = np.array([1.1 * x_min - 0.1 * x_max, 1.1 * x_max - 0.1 * x_min])
    y_line = np.array([y_mean + r * (x_line[0] - x_mean),
                      y_mean + r * (x_line[1] - x_mean)])
    plt.scatter(x_array, y_array)
    plt.plot(x_line, y_line, color='gray', linestyle='--')
    return r, p


def plot_bland_altman(x_data, y_data):
    """
    Plot Bland-Altman plot.

    :param x_data: Input data for x-axis.
    :type x_data: list
    :param y_data: Input data for y-axis.
    :type y_data: list
    :return: tuple of mean and difference arrays.
    :rtype: tuple
    """
    x_array, y_array = np.asanyarray(x_data), np.asanyarray(y_data)
    mean_array = (x_array + y_array) * 0.5
    diff_array = x_array - y_array
    mean_diff, std_diff = np.mean(diff_array), np.std(diff_array)
    plt.scatter(mean_array, diff_array)
    plt.axhline(mean_diff, color='gray', linestyle='--')
    plt.axhline(mean_diff + 1.96 * std_diff, color='gray', linestyle='--')
    plt.axhline(mean_diff - 1.96 * std_diff, color='gray', linestyle='--')
    return mean_array, diff_array


def load_mahnob_hci(video_path, physio_path, root_dir=None):
    """
    Load MAHNOB-HCI data.

    :param video_path: Path to video file.
    :type video_path: str
    :param physio_path: Path to physio data file.
    :type physio_path: str
    :param root_dir: Root directory for video and physio paths.
    :type root_dir: str, optional
    :return: A tuple containing the video capture object, ECG data, PPG data, start index, and end index.
    :rtype: tuple
    """
    if root_dir is not None:
        video_path, physio_path = os.path.join(
            root_dir, video_path), os.path.join(root_dir, physio_path)
    assert os.path.exists(video_path), f'File ({video_path}) does not exist'
    assert os.path.exists(physio_path), f'File ({physio_path}) does not exist'
    video_capture = cv2.VideoCapture(video_path)
    data_reader = edf.EdfReader(physio_path)
    start_index, end_index = 0, int(
        video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    set_exact_frame(video_capture, start_index)
    time_frame = np.arange(start_index) * \
        (1 / video_capture.get(cv2.CAP_PROP_FPS))
    time_ecg = np.arange(ecg_data.shape[1]) * \
        (1 / data_reader.getSampleFrequency(32))
    ecg_data = np.empty((3, data_reader.getNSamples()[0]), dtype=np.float64)
    ecg_data[0] = data_reader.readSignal(32)  # RA
    ecg_data[1] = data_reader.readSignal(33)  # LA, this is preferred
    ecg_data[2] = data_reader.readSignal(34)  # LL'
    # Aligned (resample) by linear interpolation
    # might be not accurate
    ecg_interp = interp1d(
        time_ecg, ecg_data, kind='linear', bounds_error=False)(time_frame)
    ppg_data = None  # no ppg
    # ecg and ppg are sync and aligned by preprocess
    return video_capture, ecg_interp, ppg_data, start_index, end_index


def load_proposed(video_path, physio_path, root_dir=None):
    """
    Load proposed data.

    :param video_path: Path to video file.
    :type video_path: str
    :param physio_path: Path to physio data file.
    :type physio_path: str
    :param root_dir: Root directory for video and physio paths.
    :type root_dir: str, optional
    :return: A tuple containing the video capture object, ECG data, PPG data, start index, and end index.
    :rtype: tuple
    """
    if root_dir is not None:
        video_path, physio_path = os.path.join(
            root_dir, video_path), os.path.join(root_dir, physio_path)
    assert os.path.exists(video_path), f'File ({video_path}) does not exist'
    assert os.path.exists(physio_path), f'File ({physio_path}) does not exist'
    video_capture = cv2.VideoCapture(video_path)
    data_array = np.loadtxt(physio_path, delimiter=',')  # csv
    start_index, end_index = int(data_array[0, 0]), int(data_array[-1, 0]) + 1
    set_exact_frame(video_capture, start_index)
    # shape = (3, N), NOM_ECG_ELEC_POTL_AVF, NOM_ECG_ELEC_POTL_V, NOM_ECG_ELEC_POTL_MCL
    ecg_data = data_array[:, 3:6].T  # ecg[0] is suggested
    # shape = (N,)
    ppg_data = data_array[:, 2]
    # ecg and ppg are sync and aligned by preprocess
    return video_capture, ecg_data, ppg_data, start_index, end_index
