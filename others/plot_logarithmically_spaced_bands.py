import numpy as np
import matplotlib.pyplot as plt

f_min = 30  # minimum frequency in Hz
f_max = 15360  # maximum frequency in Hz
n_octaves = np.log2(f_max / f_min)
print(n_octaves)
n_bands_per_octave = 12
n_frequency_bins = int(n_octaves * n_bands_per_octave) + 1

# generate logarithmically spaced frequency bins
frequency_bins = np.logspace(np.log2(f_min), np.log2(f_max), num=n_frequency_bins, base=2)

# plot the frequency bins
plt.figure(figsize=(10, 6))
# add vertical lines at the boundaries of each octave
octave_boundaries = [octave for octave in range (0, int(n_octaves + 1))]
for octave in range(0, int(n_octaves + 1)):
    octave_boundary = f_min * 2 ** octave
    print(octave_boundary, " ", octave)
    plt.axvline(octave_boundary, color='gray', linestyle='--', linewidth=0.5)
    plt.text(octave_boundary, 2.1, f'{octave_boundary:.0f}', color='black', ha='center')
plt.plot(frequency_bins, np.ones_like(frequency_bins), '|', markersize=10)
plt.xlabel('Frequency [Hz]')
plt.title('Logarithmically Spaced Frequency Bins with 12 bands per octave')
plt.show()