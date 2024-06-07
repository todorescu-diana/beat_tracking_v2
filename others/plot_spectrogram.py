import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

fs = 1000  # sampling frequency (Hz)
t = np.arange(0, 2, 1/fs)  # time vector (2 seconds)

# signal composed of two sine waves: 50 Hz and 120 Hz
x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)

D = librosa.stft(x)

S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# plot the spectrogram
plt.figure(figsize=(10, 6))
librosa.display.specshow(np.abs(D), sr=fs, x_axis='time', y_axis='linear')
# # dB conversion for better visualisation
# librosa.display.specshow(S_db, sr=fs, x_axis='time', y_axis='linear')
plt.colorbar(format='%+2.0f')
plt.title('Spectrogram')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.show()