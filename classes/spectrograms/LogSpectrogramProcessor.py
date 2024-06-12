import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
from scipy.signal import stft
from classes.spectrograms.SpectrogramProcessorBase import SpectrogramProcessorBase
from constants.constants import F_MIN, F_MAX, FIXED_SAMPLE_RATE, HOP_LENGTH, LOG_BASE, N_BANDS_PER_OCTAVE, N_FFT

class LogSpectrogramProcessor(SpectrogramProcessorBase):
    def __init__(self, f_min=F_MIN, f_max=F_MAX, log_base=LOG_BASE, n_bands_per_octave=N_BANDS_PER_OCTAVE, window_length=N_FFT, hop_length=HOP_LENGTH, target_sample_rate=FIXED_SAMPLE_RATE):
        self.f_min = f_min
        self.f_max = f_max
        self.log_base = log_base
        self.sample_rate = target_sample_rate
        self.n_bands_per_octave = n_bands_per_octave
        self.n_octaves = np.log2(f_max / f_min)
        self.n_frequency_bins = int(self.n_octaves * self.n_bands_per_octave) + 1
        self.window_length = window_length
        self.hop_length = hop_length

    @staticmethod
    def spectrogram_type():
        return 'log'

    def process(self, audio_path):
        try:
            y, original_sr = librosa.load(audio_path, sr=None)
            y_resampled = librosa.resample(y, orig_sr=original_sr, target_sr=self.sample_rate)

            stft = librosa.stft(y_resampled, n_fft=self.window_length, win_length=2048, window='hann', hop_length=self.hop_length)
            magnitude_spectrogram = np.abs(stft) ** 2
            
            # convert the linear spectrogram to a logarithmically spaced spectrogram
            frequencies = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.window_length)
            print("!!!!!!!!!! ", len(frequencies))
            log_frequencies = np.logspace(np.log2(self.f_min), np.log2(self.f_max), num=self.n_frequency_bins, base=self.log_base)
            
            # map the linear spectrogram to the logarithmically spaced frequency bins
            log_spectrogram = np.zeros((self.n_frequency_bins, magnitude_spectrogram.shape[1]))
            for i in range(1, len(log_frequencies)):
                freq_mask = []
                start_idx = 0
                stop_idx = 0
                for j in range (0, len(frequencies)):
                    if frequencies[j] >= log_frequencies[i-1] and frequencies[j] < log_frequencies[i]:
                        if start_idx != 0:
                            start_idx = j
                        freq_mask.append(True)
                    else:
                        stop_idx = j-1
                        freq_mask.append(False)
                # freq_mask = (frequencies >= log_frequencies[i-1]) & (frequencies < log_frequencies[i])
                print("=============== ", len(freq_mask))
                if np.any(freq_mask):
                    log_spectrogram[i, :] = np.mean(magnitude_spectrogram[start_idx:stop_idx+1, :], axis=0)
            print("//////////////// ", log_spectrogram.shape)
            
            spectrogram_transposed = log_spectrogram.T
            return spectrogram_transposed
        except Exception as e:
            print("Exception occurred: ", e)
            return []

    def plot_spectrogram(self, spectrogram, duration_s=None, cmap='gist_heat'):
        if duration_s is not None:
            num_frames_to_display = int(duration_s * self.sample_rate / self.hop_size_samples)
            spectrogram = spectrogram[:num_frames_to_display, :]

        # plot the logarithmically spaced spectrogram
        plt.figure(figsize=(10, 6))
        plt.imshow(abs(spectrogram.T),aspect='auto', origin='lower')
        plt.set_cmap(cmap)
        plt.xlabel('Time Steps')
        plt.ylabel('Freq bins')
        plt.show()

