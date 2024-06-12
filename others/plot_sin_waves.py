import numpy as np
import matplotlib.pyplot as plt

sampling_rate = 1000  # samples per second
duration = 1  # duration in seconds
frequency_1 = 5  #fFrequency of the first sine wave in Hz
frequency_2 = 10  # frequency of the second sine wave in Hz

t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

sine_wave_1 = np.sin(2 * np.pi * frequency_1 * t)
sine_wave_2 = np.sin(2 * np.pi * frequency_2 * t)

plt.figure(figsize=(10, 6))
plt.plot(t, sine_wave_1, label=f'Sine wave 1: {frequency_1} Hz', color='blue')
plt.plot(t, sine_wave_1, label=f'Sine wave 2: {frequency_1} Hz', color='red', linestyle='--')

plt.title('sine waves with different frequencies')
plt.xlabel('time [s]')
plt.ylabel('amplitude')
plt.legend()

plt.grid(True)
plt.show()
