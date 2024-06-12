import sys
sys.path.append('')
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

sampling_rate = 44100
duration = 1
f1 = 110
f2 = 20

t = np.linspace(0, duration, int(sampling_rate * duration))
signal = np.where(t < duration / 2, np.sin(2 * np.pi * f1 * t), np.sin(2 * np.pi * f2 * t))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

ax1.plot(t, signal, color='navy')
ax1.set_title('signal waveform')
ax1.set_xlabel('time [s]')
ax1.set_ylabel('amplitude')
ax1.grid(True)
ax1.set_xlim(0, 1)

librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max), sr=sampling_rate, x_axis='time', y_axis='hz', ax=ax2)
ax2.set_title('signal spectrogram')
ax2.set_xlabel('time [s]')
ax2.set_ylabel('frequency [Hz]')
ax2.set_ylim(0, 2000)
ax2.grid(True)

plt.tight_layout()
plt.savefig('figure_plots/sin_2_freq_with_spectrogram.png')
plt.close()
