import matplotlib.pyplot as plt
from scipy.signal.windows import hann

window_length = 50

hann_window = hann(window_length + 1)

plt.figure(figsize=(8, 4))
plt.plot(hann_window, label='hann window', color='teal')
plt.title('hann window')
plt.xlabel('sample')
plt.ylabel('amplitude')
plt.grid(True)
plt.legend()

plt.savefig('figure_plots/hann_window.png')
plt.close()
