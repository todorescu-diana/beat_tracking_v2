import librosa
import matplotlib.pyplot as plt

# Path to your MP3 file
audio_path = ''

# Load the audio file
y, sr = librosa.load(audio_path, sr=None)  # sr=None to preserve the original sampling rate

# Define the start and end times in seconds
start_time_minutes = 2
start_time_seconds = 27
start_time = start_time_minutes * 60 + start_time_seconds  # Convert to total seconds
end_time = start_time + 5  # 30 seconds from 2:29

# Calculate the sample indices for the desired time window
start_sample = int(start_time * sr)
end_sample = int(end_time * sr)

# Extract the segment of the audio
y_segment = y[start_sample:end_sample]

# Plot the waveform of the extracted segment
plt.figure(figsize=(10, 5))
librosa.display.waveshow(y_segment, sr=sr, color='darkslategray')
plt.title(f'Rammstein - Zeit, {start_time_minutes}:{start_time_seconds} - {start_time_minutes}:{start_time_seconds + 30}')
plt.xlabel('time [s]')
plt.ylabel('amplitude')
plt.xlim(0, 5)
plt.tight_layout()
plt.savefig('figure_plots/audio_wave.png')
plt.close()
