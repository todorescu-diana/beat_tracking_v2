import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.feature
from constants.constants import FPS, NUM_BINS, FIXED_SAMPLE_RATE, FFT_SIZE, F_MIN, F_MAX


class SpectrogramProcessor:
    def __init__(self, fps=FPS, num_bins=NUM_BINS, target_sample_rate=FIXED_SAMPLE_RATE, n_fft=FFT_SIZE):
        self.fps = fps
        self.num_bins = num_bins
        self.sample_rate = target_sample_rate
        self.n_fft = n_fft
        self.hop_length = int(self.sample_rate / self.fps)
        self.filter_bank = librosa.filters.mel(sr=self.sample_rate, n_fft=self.n_fft, n_mels=self.num_bins,
                                                fmin=F_MIN, fmax=F_MAX)

    def process(self, audio_path):
      try:
        # load audio file using librosa
        y, original_sr = librosa.load(audio_path)

        # ---
        # Define the start and end times in seconds
        start_time = 60  # 1 minute in seconds
        end_time = 90    # 1 minute 30 seconds in seconds

        # Calculate the sample indices for the start and end times
        # start_sample = int(start_time * original_sr)
        # end_sample = int(end_time * original_sr)

        # # Extract the desired portion of the audio
        # y_segment = y[start_sample:end_sample]
        y_segment=y
        # ---

        # resample to the target sample rate
        y_resampled = librosa.resample(y_segment, orig_sr=original_sr, target_sr=self.sample_rate)

        # # calculate power spectrogram
        # spectrogram = np.abs(librosa.stft(y_resampled, n_fft=self.n_fft, hop_length=self.hop_length)) ** 2

        # # Apply mel filter bank
        # mel_spectrogram = np.dot(self.filter_bank, spectrogram)

        # # convert to decibels
        # spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # spectrogram_transposed = spectrogram_db.T

        # return spectrogram_transposed
        # Parameters
        n_fft = 2048  # FFT size
        hop_length = int(0.01 * 44100)  # 10 ms hop size
        win_length = n_fft  # Window length
        window = 'hann'  # Window shape
        fmin = 30  # Minimum frequency
        fmax = 10000  # Maximum frequency
        n_mels = 81  # Total number of Mel bands

        # Compute the Mel filterbank
        mel_filterbank = librosa.filters.mel(sr=self.sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

        # Compute the STFT
        S = librosa.stft(y_resampled, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)

        # Map the STFT to the Mel scale using the filterbank
        S_mel = np.dot(mel_filterbank, np.abs(S)**2)

        # Convert to decibels
        S_mel_db = librosa.power_to_db(S_mel, ref=np.max)

        return S_mel_db.T
      except Exception as e:
        print("Exception occurred: ", e)
        return []
      

    def plot_spectrogram(self, spectrogram, duration=None, cmap='gist_heat'):
        # calculate number of frames to display based on duration
        if duration is not None:
            num_frames_to_display = int(duration * self.sample_rate / self.hop_length)
            spectrogram = spectrogram[:num_frames_to_display, :]

        # determine the number of bins from the shape of the spectrogram
        num_bins = spectrogram.shape[1]

        # determine y-axis tick positions in increments of 20
        y_tick_positions = np.arange(0, num_bins, int(num_bins / 5))  # Adjusted to show 5 ticks

        # plot spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spectrogram.T, sr=self.sample_rate, hop_length=441, x_axis='time',
                                 fmin=F_MIN, fmax=F_MAX)

        # set y-axis ticks to represent the bins in increments of 20
        plt.yticks(y_tick_positions)
        # Set the label for the y-axis
        plt.ylabel('Frequency [bins]')

        # add colorbar
        plt.colorbar(format='%+2.0f dB')

        # set title
        plt.title('Log-frequency power spectrogram')

        plt.set_cmap(cmap)

        # adjust layout to prevent clipping of labels
        plt.tight_layout()

        # show the plot
        plt.show()