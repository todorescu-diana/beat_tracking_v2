import sys
sys.path.append('')
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

sampling_rate = 44100
duration = 1
f1 = 110
f2 = 20

t = np.linspace(0, duration, int(sampling_rate * duration))

signal = np.where(t < duration / 2, np.sin(2 * np.pi * f1 * t), np.sin(2 * np.pi * f2 * t))

N = len(signal)
yf = fft(signal)
xf = fftfreq(N, 1 / sampling_rate)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(t, signal, color='navy')
ax1.set_title('signal waveform')
ax1.set_xlabel('time [s]')
ax1.set_ylabel('amplitude')
ax1.grid(True)
ax1.set_xlim(0, 1)

ax2.plot(xf[:N // 2], 2.0 / N * np.abs(yf[:N // 2]), color='navy')
ax2.set_title('spectrum of the signal')
ax2.set_xlabel('frequency [Hz]')
ax2.set_ylabel('amplitude')
ax2.set_xlim(0, 200)
ax2.grid(True)

plt.tight_layout()
plt.savefig('figure_plots/sin_2_freq_with_spectrum.png')
plt.close()
