import numpy as np
import matplotlib.pyplot as plt

fs = 8000
T = 0.3
t = np.arange(0, T, 1/fs)

f1 = 10 
f2 = 50 
y1 = np.sin(2 * np.pi * f1 * t)
y2 = np.sin(2 * np.pi * f2 * t)

y_sum = y1 + y2

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

plt.subplot(2, 1, 2)
freqs = np.fft.fftfreq(len(t), 1/fs) 
Y_sum = np.fft.fft(y_sum) / len(t) 

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
