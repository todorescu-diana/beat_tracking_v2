import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.feature
from constants.constants import FPS, HOP_LENGTH, N_FFT, N_MELS, NUM_BINS, FIXED_SAMPLE_RATE, F_MIN, F_MAX, WINDOW


class SpectrogramProcessor:
    def __init__(self, fps=FPS, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=N_FFT, window=WINDOW, f_min=F_MIN, f_max=F_MAX, n_mels=N_MELS, target_sample_rate=FIXED_SAMPLE_RATE):
        self.fps = fps
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels
        self.sample_rate = target_sample_rate
        self.mel_filterbank = librosa.filters.mel(sr=self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)

    def process(self, audio_path):
      try:
        y, original_sr = librosa.load(audio_path)
        y_resampled = librosa.resample(y, orig_sr=original_sr, target_sr=self.sample_rate)

        S = librosa.stft(y_resampled, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window)
        S_mel = np.dot(self.mel_filterbank, np.abs(S)**2)
        S_mel_db = librosa.power_to_db(S_mel, ref=np.max)

        return S_mel_db.T
      except Exception as e:
        print("Exception occurred: ", e)
        return []
      

    def plot_spectrogram(self, spectrogram, duration_s=None, cmap='gist_heat'):
        if duration_s is not None:
            num_frames_to_display = int(duration_s * self.sample_rate / self.hop_length)
            spectrogram = spectrogram[:num_frames_to_display, :]

        num_bins = spectrogram.shape[1]
        y_tick_positions = np.arange(0, num_bins, int(num_bins / 5))

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spectrogram.T, sr=self.sample_rate, hop_length=self.hop_length, x_axis='time',
                                 fmin=self.f_min, fmax=self.f_max)

        plt.yticks(y_tick_positions)
        plt.ylabel('Frequency [bins]')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Log-frequency power spectrogram')
        plt.set_cmap(cmap)
        plt.tight_layout()

        plt.show()