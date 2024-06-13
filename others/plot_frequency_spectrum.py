import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 8000  # Sampling frequency (samples per second)
T = 0.3     # Duration (seconds)
t = np.arange(0, T, 1/fs)  # Time vector

# Generate signals
f1 = 10  # Frequency of first sine wave (Hz)
f2 = 50  # Frequency of second sine wave (Hz)
y1 = np.sin(2 * np.pi * f1 * t)
y2 = np.sin(2 * np.pi * f2 * t)

# Calculate sum of signals
y_sum = y1 + y2

# Plot time-domain signals
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t, y1, label=f'signal 1: {f1} Hz', linestyle='--', color='lightcoral')
plt.plot(t, y2, label=f'signal 2: {f2} Hz', linestyle='--', color='darkcyan')
plt.plot(t, y_sum, label='sum', linewidth=2, color='darkslateblue')
plt.title('time-domain signals')
plt.xlabel('time [s]')
plt.ylabel('amplitude')
plt.legend()
plt.grid()

# Calculate and plot spectrum of the summed signal
plt.subplot(2, 1, 2)
freqs = np.fft.fftfreq(len(t), 1/fs)  # Frequency bins
Y_sum = np.fft.fft(y_sum) / len(t)   # FFT and normalize

plt.plot(freqs[:len(t) // 2], np.abs(Y_sum[:len(t) // 2]) , color='navy')

plt.xlim(0, 500)
plt.title('spectrum of summed signal')
plt.xlabel('frequency [Hz]')
plt.ylabel('magnitude')
plt.grid()
existing_ticks = plt.gca().get_xticks()  # Get current x-axis ticks
plt.xticks(np.unique(np.concatenate([existing_ticks, [10, 50]])))  # Concatenate existing ticks with custom ticks
plt.tight_layout()

plt.savefig('figure_plots/summed_signal_with_spectrum.png')
plt.close()
