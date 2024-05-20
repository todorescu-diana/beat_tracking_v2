from classes.data_sequence import DataSequence
from classes.spectrogram_processor import SpectrogramProcessor
from constants.constants import PAD_FRAMES

import mirdata

gtzan = mirdata.initialize('gtzan_genre', version='mini')
tracks = gtzan.load_tracks()

pre_processor = SpectrogramProcessor()

# assuming tracks is a dictionary
first_key, first_value = next(iter(tracks.items()), (None, None))
data_sequence_tracks = {first_key: first_value}

train = DataSequence(
    data_sequence_tracks=data_sequence_tracks,
    data_sequence_pre_processor=pre_processor,
    data_sequence_pad_frames=PAD_FRAMES
)
train.widen_beat_targets()

print("Train DataSequence: ", train)
print("Train[0] DataSequence: ", train[0]) # [spectrogram, beats]
print("Train[0][0] DataSequence: ", train[0][0]) # spectrogram
print("Train[0][1] DataSequence: ", train[0][1]) # beats
for idx, spectrogram_slice in enumerate(train[0][0]):
    print(f"Spectrogram slice {idx+1}/{len(train[0][0])} shape: ", spectrogram_slice.shape) # beats
for idx, beats_slice in enumerate(train[0][1]):
    print(f"Beats slice {idx+1}/{len(train[0][1])} shape: ", beats_slice.shape) # beats