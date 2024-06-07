import numpy as np
import matplotlib.pyplot as plt

sampling_rate = 1000  # samples per second
duration = 1  # duration in seconds
frequencies = [5, 50, 120]  # frequencies of the sine waves in Hz
amplitudes = [1, 0.5, 0.2]  # amplitudes of the sine waves

t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

signal = np.zeros_like(t)
for frequency, amplitude in zip(frequencies, amplitudes):
    signal += amplitude * np.sin(2 * np.pi * frequency * t)

fft_values = np.fft.fft(signal)
fft_frequencies = np.fft.fftfreq(len(t), 1 / sampling_rate)

fft_magnitude = np.abs(fft_values) / len(t)

plt.figure(figsize=(10, 6))
plt.plot(fft_frequencies, fft_magnitude)
plt.title('Frequency Spectrum of the Signal')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.grid(True)
plt.xlim(0, 150)  # limit x-axis to relevant frequency range
plt.show()
