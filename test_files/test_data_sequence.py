import sys
sys.path.append('')
from classes.sequences.data_sequence import DataSequence
from constants.constants import PAD_FRAMES
from classes.spectrograms.SpectrogramProcessorFactory import SpectrogramProcessorFactory

import mirdata

gtzan = mirdata.initialize('gtzan_genre', version='mini')
tracks = gtzan.load_tracks()

spectrogram_processor_factory = SpectrogramProcessorFactory()
db_preprocessor = spectrogram_processor_factory.create_spectrogram_processor('db')
# log_preprocessor = spectrogram_processor_factory.create_spectrogram_processor('log')
mel_preprocessor = spectrogram_processor_factory.create_spectrogram_processor('mel')

# assuming tracks is a dictionary
first_key, first_value = next(iter(tracks.items()), (None, None))
tracks = {first_key: first_value}

train_db = DataSequence(
    tracks=tracks,
    pre_processor=db_preprocessor,
    pad_frames=PAD_FRAMES
)
train_db.widen_beat_targets()

# train_log = DataSequence(
#     tracks=tracks,
#     pre_processor=log_preprocessor,
#     pad_frames=PAD_FRAMES
# )
# train_log.widen_beat_targets()

train_mel = DataSequence(
    tracks=tracks,
    pre_processor=mel_preprocessor,
    pad_frames=PAD_FRAMES
)
train_mel.widen_beat_targets()

print("------------------ DB")
print("Train DataSequence: ", train_db)
print("Train[0][0] DataSequence: ", train_db[0][0].shape) # spectrogram
print("Train[0][1] DataSequence: ", train_db[0][1]['beats'].shape) # beats


# print("------------------ LOG")
# print("Train DataSequence: ", train_log)
# print("Train[0][0] DataSequence: ", train_log[0][0].shape) # spectrogram
# print("Train[0][1] DataSequence: ", train_log[0][1]['beats'].shape) # beats

print("------------------ MEL")
print("Train DataSequence: ", train_mel)
print("Train[0][0] DataSequence: ", train_mel[0][0].shape) # spectrogram
print("Train[0][1] DataSequence: ", train_mel[0][1]['beats'].shape) # beats


db_spectrogram = db_preprocessor.process(first_value.audio_path)
db_preprocessor.plot_spectrogram(db_spectrogram)

# log_spectrogram = log_preprocessor.process(first_value.audio_path)
# log_preprocessor.plot_spectrogram(log_spectrogram)

mel_spectrogram = mel_preprocessor.process(first_value.audio_path)
mel_preprocessor.plot_spectrogram(mel_spectrogram)