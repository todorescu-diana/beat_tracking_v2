import sys
sys.path.append('')
from classes.sequences.data_sequence import DataSequence
from constants.constants import PAD_FRAMES
from classes.spectrograms.SpectrogramProcessorFactory import SpectrogramProcessorFactory

import mirdata

gtzan = mirdata.initialize('gtzan_genre', version='mini')
tracks = gtzan.load_tracks()

spectrogram_processor_factory = SpectrogramProcessorFactory()
mel_preprocessor = spectrogram_processor_factory.create_spectrogram_processor('mel')
cqt_preprocessor = spectrogram_processor_factory.create_spectrogram_processor('cqt')
log_preprocessor = spectrogram_processor_factory.create_spectrogram_processor('log')

# assuming tracks is a dictionary
first_key, first_value = next(iter(tracks.items()), (None, None))
tracks = {first_key: first_value}

train_mel = DataSequence(
    tracks=tracks,
    pre_processor=mel_preprocessor,
    pad_frames=PAD_FRAMES
)
train_mel.widen_beat_targets()

train_cqt = DataSequence(
    tracks=tracks,
    pre_processor=cqt_preprocessor,
    pad_frames=PAD_FRAMES
)
train_cqt.widen_beat_targets()

train_log = DataSequence(
    tracks=tracks,
    pre_processor=log_preprocessor,
    pad_frames=PAD_FRAMES
)
train_log.widen_beat_targets()

print("------------------ MEL")
print("Train DataSequence: ", train_mel)
# print("Train[0] DataSequence: ", train_mel[0]) # [spectrogram, beats]
print("Train[0][0] DataSequence: ", train_mel[0][0].shape) # spectrogram
print("Train[0][1] DataSequence: ", train_mel[0][1]['beats'].shape) # beats

# print("------------------ CQT")
# print("Train DataSequence: ", train_cqt)
# # print("Train[0] DataSequence: ", train_cqt[0]) # [spectrogram, beats]
# print("Train[0][0] DataSequence: ", train_cqt[0][0].shape) # spectrogram
# print("Train[0][1] DataSequence: ", train_cqt[0][1]['beats'].shape) # beats

print("------------------ LOG")
print("Train DataSequence: ", train_log)
# print("Train[0] DataSequence: ", train_cqt[0]) # [spectrogram, beats]
print("Train[0][0] DataSequence: ", train_log[0][0].shape) # spectrogram
print("Train[0][1] DataSequence: ", train_log[0][1]['beats'].shape) # beats

mel_spectrogram = mel_preprocessor.process(first_value.audio_path)
mel_preprocessor.plot_spectrogram(mel_spectrogram)

# cqt_spectrogram = cqt_preprocessor.process(first_value.audio_path)
# cqt_preprocessor.plot_spectrogram(cqt_spectrogram)

log_spectrogram = log_preprocessor.process(first_value.audio_path)
log_preprocessor.plot_spectrogram(log_spectrogram)