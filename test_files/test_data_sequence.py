import sys
sys.path.append('')
from utils.dataset_utils import get_load_dataset_params, load_dataset

from classes.sequences.data_sequence import DataSequence
from constants.constants import PAD_FRAMES, VALID_DATASET_NAMES
from classes.spectrograms.SpectrogramProcessorFactory import SpectrogramProcessorFactory

import mirdata

dataset_name = VALID_DATASET_NAMES[10]

replace_dots_with_underline, tiny_aam, harmonix_set, gtzan_rhythm = get_load_dataset_params(dataset_name)

spectrogram_processor_factory = SpectrogramProcessorFactory()
mel_preprocessor = spectrogram_processor_factory.create_spectrogram_processor('mel')
pre_processor = mel_preprocessor
pre_processor_type = pre_processor.spectrogram_type()

tracks = load_dataset(dataset_name, replace_dots_with_underline, tiny_aam, harmonix_set, gtzan_rhythm)

spectrogram_processor_factory = SpectrogramProcessorFactory()
db_preprocessor = spectrogram_processor_factory.create_spectrogram_processor('db')
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

train_mel = DataSequence(
    tracks=tracks,
    pre_processor=mel_preprocessor,
    pad_frames=PAD_FRAMES
)
train_mel.widen_beat_targets()

print("------------------ MEL")
print("Train DataSequence: ", train_mel)
print("Train[0][0] DataSequence: ", train_mel[0][0].shape) # spectrogram
print("Train[0][1] DataSequence: ", train_mel[0][1]['beats'].shape) # beats

mel_spectrogram = mel_preprocessor.process(first_value.audio_path)
mel_preprocessor.plot_spectrogram(mel_spectrogram)