import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
from classes.spectrograms.SpectrogramProcessorBase import SpectrogramProcessorBase
from constants.constants import F_MIN, F_MAX, FIXED_SAMPLE_RATE, HOP_LENGTH, N_FFT

class DecibelSpectrogramProcessor(SpectrogramProcessorBase):
    def __init__(self, f_min=F_MIN, f_max=F_MAX, n_fft=N_FFT, hop_length=HOP_LENGTH, target_sample_rate=FIXED_SAMPLE_RATE):
        self.f_min = f_min
        self.f_max = f_max
        self.sample_rate = target_sample_rate
        self.window_length = n_fft
        self.hop_length = hop_length

    @staticmethod
    def spectrogram_type():
        return 'db'

    def process(self, audio_path):
        try:
            y, original_sr = librosa.load(audio_path, sr=None)
            y_resampled = librosa.resample(y, orig_sr=original_sr, target_sr=self.sample_rate)

            stft = librosa.stft(y_resampled, n_fft=self.window_length, win_length=2048, window='hann', hop_length=self.hop_length)
            magnitude_spectrogram = np.abs(stft) ** 2
            spectrogram_db = librosa.power_to_db(magnitude_spectrogram, ref=np.max)

            spectrogram_transposed = spectrogram_db.T
            return spectrogram_transposed
        except Exception as e:
            print("Exception occurred: ", e)
            return []

    def plot_spectrogram(self, spectrogram, duration_s=None, cmap='gist_heat'):
        if duration_s is not None:
            num_frames_to_display = int(duration_s * self.sample_rate / self.hop_length)
            spectrogram = spectrogram[:num_frames_to_display, :]

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spectrogram.T, sr=self.sample_rate, hop_length=self.hop_length, x_axis='time', y_axis='log',
                                 fmin=F_MIN, fmax=F_MAX)
        plt.set_cmap(cmap)
        plt.colorbar(label='dB')
        plt.ylabel('frequency [Hz]')
        plt.xlabel('time [s]')
        plt.title('dB log spectrogram')
        plt.ylim(self.f_min, self.f_max)
        plt.tight_layout()
        plt.show()

