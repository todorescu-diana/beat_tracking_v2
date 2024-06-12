import numpy as np
import matplotlib.pyplot as plt

duration_seconds = 1
fps = 100
total_frames = duration_seconds * fps

beat_probabilities = np.random.rand(total_frames)
time_axis = np.linspace(0, duration_seconds, total_frames)

plt.plot(time_axis, beat_probabilities)
plt.title('beat activation function')
plt.xlabel('time (s)')
plt.ylabel('probability')
plt.grid(True)
plt.show()

