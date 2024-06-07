import sys
sys.path.append('')
import librosa
import librosa.display
import matplotlib.pyplot as plt

from constants.constants import F_MAX, F_MIN, FFT_SIZE, FIXED_SAMPLE_RATE, NUM_BINS

filter_bank = librosa.filters.mel(sr=FIXED_SAMPLE_RATE, n_fft=FFT_SIZE, n_mels=NUM_BINS,
                                        fmin=F_MIN, fmax=F_MAX)

plt.figure(figsize=(10, 4))
librosa.display.specshow(filter_bank, sr=FIXED_SAMPLE_RATE, hop_length=512, x_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Filterbanks')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Mel Filter Index')
plt.tight_layout()
plt.show()
